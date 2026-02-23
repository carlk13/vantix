import os
import time
import sys
import glob
import psutil
import statistics
import threading
import multiprocessing
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

# Ensure we can import the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure output directory exists
IMG_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(IMG_OUT_DIR, exist_ok=True)

try:
    from vantix import VantixLoader
    print("vantix detected and imported.")
except ImportError:
    print("vantix NOT found. The benchmark will fail for vantix cases.")
    print("   Run 'maturin develop --release' first.")
    sys.exit(1)

class ResourceMonitor(threading.Thread):
    """
    Runs in the background to track total CPU/RAM usage across 
    the main process AND all child processes (PyTorch workers).
    """
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.cpu_stats = []
        self.mem_stats = []
        self.stop_event = threading.Event()
        self.main_process = psutil.Process(os.getpid())

    def run(self):
        # Initial call to cpu_percent is always 0 or meaningless, so we prime it.
        self.main_process.cpu_percent()
        
        while not self.stop_event.is_set():
            try:
                # Main Process Usage
                total_cpu = self.main_process.cpu_percent()
                total_mem = self.main_process.memory_info().rss
                
                # Child Processes (PyTorch Workers)
                # We must fetch children every tick because workers might die/respawn
                children = self.main_process.children(recursive=True)
                for child in children:
                    try:
                        # Summing cpu_percent across all cores
                        total_cpu += child.cpu_percent()
                        total_mem += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass # Process died between listing and query
                
                self.cpu_stats.append(total_cpu)
                self.mem_stats.append(total_mem / 1024 / 1024) # Convert to MB
            except:
                pass # robust against process tear-down
            
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
        self.join()

    def get_avg(self):
        avg_cpu = statistics.mean(self.cpu_stats) if self.cpu_stats else 0
        avg_mem = statistics.mean(self.mem_stats) if self.mem_stats else 0
        return avg_cpu, avg_mem

def prepare_robust_dataset(name, target_resolution, min_images=2000):
    """Ensures a LARGE dataset exists on disk."""
    root_base = f"./bench_data/scientific_{name}_{target_resolution}px"
    img_dir = os.path.join(root_base, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    existing_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    if len(existing_files) >= min_images:
        print(f"   [Data] Found existing set: {name} ({len(existing_files)} imgs)")
        return sorted(existing_files)[:min_images]

    print(f"   [Data] Generating {min_images} images for '{name}'...")
    
    # Download Source (Use STL10 for high quality base)
    os.makedirs("./bench_data/downloads", exist_ok=True)
    try:
        ds = datasets.STL10(root="./bench_data/downloads", split='train+unlabeled', download=True)
    except:
        ds = datasets.CIFAR10(root="./bench_data/downloads", train=True, download=True)

    paths = []
    source_iter = iter(ds)
    
    for i in range(min_images):
        try:
            img, _ = next(source_iter)
        except StopIteration:
            source_iter = iter(ds)
            img, _ = next(source_iter)

        if not isinstance(img, Image.Image): 
            img = Image.fromarray(img)
        
        if img.size[0] != target_resolution:
            img = img.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
        
        p = os.path.join(img_dir, f"img_{i:05d}.jpg")
        img.save(p, quality=90, subsampling=0)
        paths.append(os.path.abspath(p))
        
        if i % 500 == 0:
            print(f"          Progress: {i}/{min_images}", end="\r")
            
    return paths

class VanillaDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        with open(self.paths[idx], 'rb') as f:
            img = Image.open(f).convert('RGB')
            return self.transform(img)

def get_pytorch_loader(paths, size, batch, workers):
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return DataLoader(
        VanillaDataset(paths, t), 
        batch_size=batch, 
        num_workers=workers,
        pin_memory=True, 
        persistent_workers=True if workers > 0 else False,
        drop_last=True
    )

def run_trial(name, loader, monitor):
    # Hardware Warmup
    try:
        iterator = iter(loader)
        _ = next(iterator) 
    except StopIteration:
        return 0, 0, 0

    # Start Measurement (Background Thread)
    monitor = ResourceMonitor(interval=0.05) # Sample every 50ms
    monitor.start()
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_t = time.perf_counter()
    
    count = 0
    for batch in loader:
        count += batch.shape[0]
        if torch.cuda.is_available():
            batch = batch.cuda(non_blocking=True)

    if torch.cuda.is_available(): torch.cuda.synchronize()

    duration = time.perf_counter() - start_t
    monitor.stop() # Stop sampling
    
    cpu, mem = monitor.get_avg()
    return count / duration, cpu, mem

def rigorous_benchmark(scenario):
    print(f"\n🔎 ANALYZING: {scenario['name']}")
    print("-" * 60)
    
    paths = prepare_robust_dataset(scenario['name'], scenario['src_res'], min_images=scenario['n_images'])
    BATCH = scenario['batch']
    TRIALS = 3  
    WORKERS = min(8, multiprocessing.cpu_count())
    
    results = {"PyTorch": [], "vantix": []}
    
    # --- PYTORCH ---
    print(f"   [PyTorch] Initializing (Workers: {WORKERS})...")
    pt_loader = get_pytorch_loader(paths, scenario['target_res'], BATCH, WORKERS)
    
    print("   [PyTorch] Warmup pass...", end="\r")
    run_trial("PyTorch-Warmup", pt_loader, ResourceMonitor())
    
    for i in range(TRIALS):
        speed, cpu, mem = run_trial(f"PT-{i}", pt_loader, ResourceMonitor())
        print(f"   [PyTorch] Trial {i+1}: {speed:.1f} img/s | Total CPU: {cpu:.0f}% ({cpu/100:.1f} cores)")
        results["PyTorch"].append({"speed": speed, "cpu": cpu})
        
    # --- VANTIX ---
    print(f"   [vantix] Initializing...")
    rs_loader = VantixLoader(
        image_paths=paths, 
        width=scenario['target_res'], 
        height=scenario['target_res'], 
        batch_size=BATCH,
        augment=False,
        shuffle=False
    )
    
    print("   [vantix] Warmup pass...", end="\r")
    run_trial("vantix-Warmup", rs_loader, ResourceMonitor())
    
    for i in range(TRIALS):
        speed, cpu, mem = run_trial(f"RS-{i}", rs_loader, ResourceMonitor())
        print(f"   [vantix] Trial {i+1}: {speed:.1f} img/s | Total CPU: {cpu:.0f}% ({cpu/100:.1f} cores)")
        results["vantix"].append({"speed": speed, "cpu": cpu})

    # --- STATISTICS ---
    pt_avg = statistics.mean([r['speed'] for r in results["PyTorch"]])
    rs_avg = statistics.mean([r['speed'] for r in results["vantix"]])
    
    # Note: Adding small epsilon to prevent division by zero
    pt_cpu = statistics.mean([r['cpu'] for r in results["PyTorch"]]) + 1e-6
    rs_cpu = statistics.mean([r['cpu'] for r in results["vantix"]]) + 1e-6
    
    return {
        "Scenario": scenario['name'],
        "PT_Mean": pt_avg, "PT_CPU": pt_cpu,
        "RS_Mean": rs_avg, "RS_CPU": rs_cpu,
        "Speedup": rs_avg / pt_avg if pt_avg > 0 else 0
    }

SCENARIOS = [
    {
        "name": "Medical-2K",
        "n_images": 500, 
        "src_res": 2048, 
        "target_res": 224, 
        "batch": 16
    },
    {
        "name": "ImageNet-Std",
        "n_images": 2000,
        "src_res": 500,   
        "target_res": 224,
        "batch": 128
    }
]

if __name__ == "__main__":
    print("===============================================================")
    print("     BENCHMARK: Vantix vs PyTorch (Total Resource Usage)")
    print("===============================================================")
    
    final_stats = []
    for scen in SCENARIOS:
        stats = rigorous_benchmark(scen)
        final_stats.append(stats)
        
    df = pd.DataFrame(final_stats)
    
    print("\n\nFINAL EFFICIENCY REPORT")
    print("=" * 100)
    print(f"{'Scenario':<15} | {'PT Speed':<12} | {'RS Speed':<12} | {'Speedup':<8} | {'PT CPU':<10} | {'RS CPU':<10} | {'Eff. Gain'}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        # Efficiency = Images per second / Total CPU utilization
        # Example: 1000 imgs / 800% CPU = 1.25 efficiency score
        pt_eff = row['PT_Mean'] / row['PT_CPU']
        rs_eff = row['RS_Mean'] / row['RS_CPU']
        eff_gain = rs_eff / pt_eff if pt_eff > 0 else 0
        
        print(f"{row['Scenario']:<15} | {row['PT_Mean']:<6.0f} img/s | {row['RS_Mean']:<6.0f} img/s | {row['Speedup']:<5.2f}x   | {row['PT_CPU']:<4.0f}%      | {row['RS_CPU']:<4.0f}%      | {eff_gain:.2f}x")
    
    print("=" * 100)
    print("* PT CPU: Sum of Main Process + 8 Worker Processes")
    print("* RS CPU: Main Process (Multithreaded)")
    print("* Eff. Gain: (Vantix Images per CPU Cycle) / (PyTorch Images per CPU Cycle)")