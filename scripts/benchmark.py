import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Import your Rust package
import vantix 

# Configuration
N_IMAGES = 500          # Number of images to test
IMG_SIZE = 1024         # Large images emphasize the speed difference (1024x1024)
TARGET_SIZE = 224       # Resize target (standard for ResNet/ViT)
BATCH_SIZE = 100        # Process in chunks
DATA_DIR = "./bench_data"

# ==========================================
# 1. SETUP: Generate Dummy Data
# ==========================================
def setup_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"Generating {N_IMAGES} dummy images ({IMG_SIZE}x{IMG_SIZE})...")
    # We generate random noise images to simulate real IO work
    for i in range(N_IMAGES):
        fname = os.path.join(DATA_DIR, f"img_{i}.jpg")
        if not os.path.exists(fname):
            # Create a random image
            arr = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(fname, quality=80)
    
    # Return list of absolute paths
    return [os.path.abspath(os.path.join(DATA_DIR, f"img_{i}.jpg")) for i in range(N_IMAGES)]

# ==========================================
# 2. COMPETITOR A: Standard PyTorch
# ==========================================
class PyTorchDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)

def run_pytorch(paths):
    # Standard transform: Resize -> Tensor -> Normalize
    transform = transforms.Compose([
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        transforms.ToTensor(),
    ])
    
    dataset = PyTorchDataset(paths, transform)
    
    # num_workers=4 is standard optimization
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    
    start = time.time()
    count = 0
    for batch in loader:
        count += batch.shape[0] # Just iterate to force loading
    
    return time.time() - start

# ==========================================
# 3. COMPETITOR B: vantix (Rust)
# ==========================================
def run_vantix(paths):
    start = time.time()
    
    # We process in batches manually since our Rust function takes a list of paths
    # (In a real app, you'd wrap this in a simple Python iterator)
    total_len = len(paths)
    for i in range(0, total_len, BATCH_SIZE):
        batch_paths = paths[i : i + BATCH_SIZE]
        
        # --- THE RUST CALL ---
        # Returns a Numpy array, which we flip to Torch (zero-copy usually)
        np_batch = vantix.load_images_fast(batch_paths, TARGET_SIZE, TARGET_SIZE)
        tensor_batch = torch.from_numpy(np_batch)
        
    return time.time() - start

# ==========================================
# 4. THE SHOWDOWN
# ==========================================
if __name__ == "__main__":
    files = setup_data()
    print(f"\n⚔️  BEGINNING BENCHMARK ({N_IMAGES} images) ⚔️\n")

    # --- Run PyTorch ---
    print(f"1. Running Standard PyTorch (Workers=4)...")
    try:
        pt_time = run_pytorch(files)
        print(f"   ⏱️  Time: {pt_time:.4f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        pt_time = 9999

    # --- Run vantix ---
    print(f"2. Running vantix (Rust Native)...")
    try:
        rs_time = run_vantix(files)
        print(f"   ⏱️  Time: {rs_time:.4f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        rs_time = 9999

    # --- Results ---
    print("\n" + "="*30)
    print(f"🏆 WINNER: {'vantix' if rs_time < pt_time else 'PYTORCH'}")
    print("="*30)
    
    if rs_time < pt_time:
        speedup = pt_time / rs_time
        print(f"🚀 vantix was {speedup:.2f}x faster than PyTorch!")
    else:
        print(f"⚠️ PyTorch was faster. Check your threading logic.")