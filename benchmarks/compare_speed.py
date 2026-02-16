import os
import time
import torch
import logging 
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

logger = logging.getLogger(__name__)

# Try to import vantix properly installed
try:
    from vantix import VantixLoader

    logger.info("Vantix package imported successfully.")
except ImportError:
    logger.info("Error: Could not import 'vantix'.")
    logger.info(
        "Please run 'maturin develop --release' to install the package in editable mode."
    )
    sys.exit(1)

# --- Configuration ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CIFAR_IMAGES_DIR = os.path.join(DATA_ROOT, "cifar10_extracted")
BATCH_SIZE = 64
EPOCHS = 3
IMG_SIZE = 128
DEVICE = "cpu"


def get_image_paths(root_dir):
    paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".png"):
                paths.append(os.path.join(root, file))
    return paths


def get_lightweight_model():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 10)
    return model


def train_loop(loader_name, loader, model, criterion, optimizer, epochs):
    model.train()
    total_images = 0

    logger.info(f"Starting benchmark for {loader_name}...")
    start_time = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        limit_batches = 50

        for i, batch in enumerate(loader):
            if i >= limit_batches:
                break

            if isinstance(batch, (list, tuple)):
                data = batch[0]
            else:
                data = batch

            data = data.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            target = torch.randint(0, 10, (data.size(0),), device=DEVICE)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_images += data.size(0)

        epoch_end = time.perf_counter()
        logger.info(
            f"  Epoch {epoch + 1}: {epoch_end - epoch_start:.4f}s ({total_images} images total)"
        )

    total_time = time.perf_counter() - start_time
    throughput = total_images / total_time
    return throughput


def main():
    logger.info(f"Running Benchmark (MobileNetV3 Small @ {IMG_SIZE}x{IMG_SIZE})")

    train_dir = os.path.join(CIFAR_IMAGES_DIR, "train")
    if not os.path.exists(train_dir):
        logger.info(f"Error: Data directory {train_dir} not found.")
        logger.info("Please run 'python benchmarks/prepare_data.py' first.")
        sys.exit(1)

    all_file_paths = get_image_paths(train_dir)
    logger.info(f"Found {len(all_file_paths)} training images.")

    criterion = nn.CrossEntropyLoss()

    # --- A. Standard PyTorch ImageFolder ---
    logger.info("\n--- Testing PyTorch ImageFolder ---")
    model_pt = get_lightweight_model().to(DEVICE)
    optimizer_pt = optim.Adam(model_pt.parameters(), lr=0.001)

    pt_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    pt_dataset = datasets.ImageFolder(train_dir, transform=pt_transform)
    pt_loader = DataLoader(
        pt_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=False,
    )

    pt_speed = train_loop(
        "PyTorch", pt_loader, model_pt, criterion, optimizer_pt, EPOCHS
    )
    logger.info(f"PyTorch Speed: {pt_speed:.2f} img/sec")

    # --- B. Vantix Loader ---
    logger.info("\n--- Testing Vantix Loader ---")
    model_vx = get_lightweight_model().to(DEVICE)
    optimizer_vx = optim.Adam(model_vx.parameters(), lr=0.001)

    try:
        vx_loader = VantixLoader(
            all_file_paths,
            width=IMG_SIZE,
            height=IMG_SIZE,
            batch_size=BATCH_SIZE,
            augment=True,
            shuffle=True,
            queue_size=5,
        )

        vx_speed = train_loop(
            "Vantix", vx_loader, model_vx, criterion, optimizer_vx, EPOCHS
        )
        logger.info(f"Vantix Speed: {vx_speed:.2f} img/sec")
    except Exception as e:
        logger.info(f"Vantix Loader Failed: {e}")
        vx_speed = 0.0

    logger.info("\n" + "=" * 30)
    logger.info("SUMMARY (Higher is better)")
    logger.info("=" * 30)
    logger.info(f"PyTorch: {pt_speed:.2f} img/s")
    logger.info(f"Vantix : {vx_speed:.2f} img/s")

    if vx_speed > pt_speed:
        logger.info(f"Winner: Vantix ({vx_speed / pt_speed:.2f}x faster)")
    else:
        logger.info("Winner: PyTorch")


if __name__ == "__main__":
    main()
