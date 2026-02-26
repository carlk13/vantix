import os
import time
import sys
import glob
import psutil
import statistics
import threading
import multiprocessing
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
    sys.exit(1)


class ResourceMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.cpu_stats = []
        self.stop_event = threading.Event()
        self.main_process = psutil.Process(os.getpid())

    def run(self):
        self.main_process.cpu_percent()
        while not self.stop_event.is_set():
            try:
                total_cpu = self.main_process.cpu_percent()
                children = self.main_process.children(recursive=True)
                for child in children:
                    try:
                        total_cpu += child.cpu_percent()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.cpu_stats.append(total_cpu)
            except:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
        self.join()

    def get_avg_cores(self):
        avg_cpu = statistics.mean(self.cpu_stats) if self.cpu_stats else 0
        return avg_cpu / 100.0


def prepare_robust_dataset(name, target_resolution, min_images=2000):
    root_base = f"./bench_data/scientific_{name}-Proxy_{target_resolution}px"
    img_dir = os.path.join(root_base, "images")
    os.makedirs(img_dir, exist_ok=True)

    existing_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    if len(existing_files) >= min_images:
        return sorted(existing_files)[:min_images]

    print(f"    [Data] Generating {min_images} images for '{name}'...")
    ds = datasets.STL10(
        root="./bench_data/downloads", split="train+unlabeled", download=True
    )
    paths = []
    source_iter = iter(ds)
    for i in range(min_images):
        img, _ = next(source_iter)
        if img.size[0] != target_resolution:
            img = img.resize(
                (target_resolution, target_resolution), Image.Resampling.LANCZOS
            )
        p = os.path.join(img_dir, f"img_{i:05d}.jpg")
        img.save(p, quality=90)
        paths.append(os.path.abspath(p))
    return paths


class VanillaDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], "rb") as f:
            img = Image.open(f).convert("RGB")
            return self.transform(img)


def get_pytorch_loader(paths, size, batch, workers):
    t = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return DataLoader(
        VanillaDataset(paths, t),
        batch_size=batch,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True if workers > 0 else False,
        drop_last=True,
    )


def run_trial(loader):
    monitor = ResourceMonitor(interval=0.05)
    monitor.start()
    start_t = time.perf_counter()
    count = 0
    for batch in loader:
        count += batch.shape[0] if hasattr(batch, "shape") else len(batch)
    duration = time.perf_counter() - start_t
    monitor.stop()
    return count / duration, monitor.get_avg_cores()


def save_time_saved_plots(df):
    """Generates a plot focusing on Throughput and Time Saved."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    labels = df["Scenario"]
    x = np.arange(len(labels))
    width = 0.35

    # Plot 1: Throughput
    ax1.bar(
        x - width / 2,
        df["PT_Speed"],
        width,
        label="PyTorch",
        color="#3498db",
        alpha=0.8,
    )
    ax1.bar(
        x + width / 2, df["RS_Speed"], width, label="Vantix", color="#2ecc71", alpha=0.8
    )
    ax1.set_ylabel("Images per Second")
    ax1.set_title("Throughput Comparison (Higher is Better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.legend()

    # Plot 2: Time to process 50,000 images
    # Calculation: (Total Images) / (Images Per Second)
    EPOCH_SIZE = 50000
    pt_time = EPOCH_SIZE / df["PT_Speed"]
    rs_time = EPOCH_SIZE / df["RS_Speed"]

    ax2.bar(x - width / 2, pt_time, width, label="PyTorch", color="#3498db", alpha=0.8)
    ax2.bar(x + width / 2, rs_time, width, label="Vantix", color="#2ecc71", alpha=0.8)
    ax2.set_ylabel("Seconds per 50k Images")
    ax2.set_title(f"Projected Time for {EPOCH_SIZE} images (Lower is Better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(IMG_OUT_DIR, "benchmark_throughput.png")
    plt.savefig(out_path)
    print(f"\n[Plot] Benchmark diagram saved to: {out_path}")


def rigorous_benchmark(scenario, num_runs=5):
    print(f"\n>>> Running Scenario: {scenario['name']} (Averaging {num_runs} runs)")
    paths = prepare_robust_dataset(
        scenario["name"], scenario["src_res"], min_images=scenario["n_images"]
    )

    # CRITICAL CHECK: Ensure paths are not empty
    if not paths:
        print("FATAL ERROR: No images found at path check. Ensure directories exist.")
        sys.exit(1)

    WORKERS = min(8, multiprocessing.cpu_count())

    # PyTorch Run
    print(f"    [PyTorch] Testing with {WORKERS} workers...")
    pt_loader = get_pytorch_loader(
        paths, scenario["target_res"], scenario["batch"], WORKERS
    )

    pt_speeds = []
    pt_cores_list = []
    for i in range(num_runs):
        speed, cores = run_trial(pt_loader)
        pt_speeds.append(speed)
        pt_cores_list.append(cores)
        print(f"      Run {i + 1}/{num_runs}: {speed:.2f} img/sec")

    avg_pt_speed = statistics.mean(pt_speeds)
    avg_pt_cores = statistics.mean(pt_cores_list)

    # Vantix Run
    print("    [Vantix] Testing...")
    rs_loader = VantixLoader(
        image_paths=paths,
        width=scenario["target_res"],
        height=scenario["target_res"],
        batch_size=scenario["batch"],
    )

    rs_speeds = []
    rs_cores_list = []
    for i in range(num_runs):
        speed, cores = run_trial(rs_loader)
        rs_speeds.append(speed)
        rs_cores_list.append(cores)
        print(f"      Run {i + 1}/{num_runs}: {speed:.2f} img/sec")

    avg_rs_speed = statistics.mean(rs_speeds)
    avg_rs_cores = statistics.mean(rs_cores_list)

    return {
        "Scenario": scenario["name"],
        "PT_Speed": avg_pt_speed,
        "PT_Cores": avg_pt_cores,
        "RS_Speed": avg_rs_speed,
        "RS_Cores": avg_rs_cores,
        "Speedup": avg_rs_speed / avg_pt_speed if avg_pt_speed > 0 else 0,
    }


SCENARIOS = [
    {
        "name": "Medical-2K",
        "n_images": 500,
        "src_res": 2048,
        "target_res": 224,
        "batch": 16,
    },
    {
        "name": "ImageNet-Std",
        "n_images": 2000,
        "src_res": 500,
        "target_res": 224,
        "batch": 128,
    },
]

if __name__ == "__main__":
    print("===============================================================")
    print("      VANTIX THROUGHPUT BENCHMARK")
    print("===============================================================")

    results = []
    for s in SCENARIOS:
        results.append(rigorous_benchmark(s, num_runs=5))

    df = pd.DataFrame(results)

    print("\nBENCHMARK SUMMARY")
    print("-" * 80)
    for _, r in df.iterrows():
        # Calculate time saved for a standard epoch
        pt_time = 50000 / r["PT_Speed"] if r["PT_Speed"] > 0 else 0
        rs_time = 50000 / r["RS_Speed"] if r["RS_Speed"] > 0 else 0
        time_saved = pt_time - rs_time

        print(
            f"{r['Scenario']:<15} | Speedup: {r['Speedup']:.2f}x | Time Saved/50k: {time_saved:.1f}s"
        )

    save_time_saved_plots(df)
