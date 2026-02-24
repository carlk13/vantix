import os
import time
import torch
import logging
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CIFAR_IMAGES_DIR = os.path.join(DATA_ROOT, "cifar10_extracted")
BATCH_SIZE = 64
IMG_SIZE = 128
NUM_RUNS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure output directory for images exists
IMG_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(IMG_OUT_DIR, exist_ok=True)

# Try to import vantix
try:
    from vantix import VantixLoader

    logger.info("Vantix package imported successfully.")
except ImportError:
    logger.error(
        "Error: Could not import 'vantix'. Please install it (e.g., pip install -e .)."
    )
    sys.exit(1)


def get_image_paths(root_dir):
    paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".png"):
                paths.append(os.path.join(root, file))
    return paths


# --- Model Definitions ---
class SimpleMLP(nn.Module):
    def __init__(self, input_size=128 * 128 * 3, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 16 * 16, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_mobilenet():
    # Helper to get a fresh model instance each time
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 10)
    return model


# Define model configurations
MODEL_CONFIGS = [
    {"name": "Small_MLP", "epochs": 10, "factory": lambda: SimpleMLP()},
    {"name": "Medium_CNN", "epochs": 5, "factory": lambda: SimpleCNN()},
    {"name": "Large_MobileNetV3", "epochs": 2, "factory": get_mobilenet},
]

from tqdm import tqdm


def train_loop(loader_name, loader, model, criterion, optimizer, epochs):
    model.train()
    total_images_processed = 0

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    # Calculate total batches if possible for tqdm
    total_batches = len(loader) if hasattr(loader, "__len__") else None

    for epoch in range(epochs):
        with tqdm(
            loader,
            total=total_batches,
            desc=f"{loader_name} Epoch {epoch + 1}/{epochs}",
            unit="batch",
        ) as pbar:
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch

                data = data.to(DEVICE)

                optimizer.zero_grad()
                output = model(data)
                # Create dummy targets since we are benchmarking data loading speed mostly
                target = torch.randint(0, 10, (data.size(0),), device=DEVICE)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_images_processed += data.size(0)

                # Optional: Update postfix with loss or valid accuracy if we were calculating it
                # pbar.set_postfix({'loss': loss.item()})

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    throughput = total_images_processed / total_time if total_time > 0 else 0

    return throughput, total_time


def plot_throughput(pt_stats, vx_stats, model_name):

    labels = ["PyTorch", "Vantix"]
    medians = [pt_stats["median"], vx_stats["median"]]
    stds = [pt_stats["std"], vx_stats["std"]]

    plt.figure(figsize=(8, 6))
    plt.bar(
        labels, medians, yerr=stds, capsize=10, color=["skyblue", "salmon"], alpha=0.7
    )
    plt.title(f"Throughput: {model_name} (Median over {NUM_RUNS} runs)")
    plt.ylabel("Images / Second")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i, v in enumerate(medians):
        plt.text(i, v + stds[i] + 5, f"{v:.1f}", ha="center", fontweight="bold")

    out_path = os.path.join(IMG_OUT_DIR, f"throughput_{model_name}.png")
    plt.savefig(out_path)
    logger.info(f"Saved throughput chart to {out_path}")
    plt.close()


def plot_time_savings(pt_times, vx_times, model_name):

    # Calculate savings
    savings = np.array(pt_times) - np.array(vx_times)

    plt.figure(figsize=(10, 6))

    # Create a boxplot for distribution of training times
    plt.subplot(1, 2, 1)
    plt.boxplot([pt_times, vx_times], labels=["PyTorch", "Vantix"], patch_artist=True)
    plt.title(f"Training Time: {model_name}")
    plt.ylabel("Time (seconds)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Create a bar chart for absolute savings per run
    plt.subplot(1, 2, 2)
    runs = range(1, NUM_RUNS + 1)
    plt.bar(runs, savings, color="green", alpha=0.7)
    plt.title(f"Savings per Run: {model_name}")
    plt.xlabel("Run ID")
    plt.ylabel("Seconds Saved (PyTorch - Vantix)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(IMG_OUT_DIR, f"time_savings_{model_name}.png")
    plt.savefig(out_path)
    logger.info(f"Saved time savings chart to {out_path}")
    plt.close()


def print_stats_table(pt_throughputs, vx_throughputs, pt_times, vx_times, model_name):
    pt_median = np.median(pt_throughputs)
    pt_var = np.var(pt_throughputs)
    pt_std = np.std(pt_throughputs)

    vx_median = np.median(vx_throughputs)
    vx_var = np.var(vx_throughputs)
    vx_std = np.std(vx_throughputs)

    speedup = vx_median / pt_median if pt_median > 0 else 0

    total_pt_time = np.sum(pt_times)
    total_vx_time = np.sum(vx_times)
    total_savings = total_pt_time - total_vx_time

    print("\n" + "=" * 80)
    print(f"RESULTS FOR MODEL: {model_name}")
    print(f"{'METRIC':<25} | {'PYTORCH':<15} | {'VANTIX':<15} | {'IMPROVEMENT':<10}")
    print("-" * 80)
    print(
        f"{'Throughput (img/s)':<25} | {pt_median:<15.2f} | {vx_median:<15.2f} | {speedup:.2f}x"
    )
    print(f"{'  (Median)':<25} | {'':<15} | {'':<15} |")
    print(f"{'Variance':<25} | {pt_var:<15.2f} | {vx_var:<15.2f} | -")
    print(f"{'Std Dev':<25} | {pt_std:<15.2f} | {vx_std:<15.2f} | -")
    print("-" * 80)
    print(
        f"{'Total Time (s)':<25} | {total_pt_time:<15.2f} | {total_vx_time:<15.2f} | -{total_savings:.2f}s"
    )
    print("=" * 80 + "\n")

    return {
        "pytorch": {"median": pt_median, "std": pt_std},
        "vantix": {"median": vx_median, "std": vx_std},
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Model Benchmark Suite")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[c["name"] for c in MODEL_CONFIGS],
        help="Specify which models to evaluate. If omitted, all models are evaluated. Supports one or multiple models (e.g. --models Small_MLP Medium_CNN).",
    )
    args = parser.parse_args()

    active_configs = MODEL_CONFIGS
    if args.models:
        active_configs = [c for c in MODEL_CONFIGS if c["name"] in args.models]
        logger.info(f"Filtered models to run: {args.models}")

    logger.info("Running Multi-Model Benchmark Suite")
    logger.info(f"Runs per model: {NUM_RUNS}, Device: {DEVICE}")

    train_dir = os.path.join(CIFAR_IMAGES_DIR, "train")
    if not os.path.exists(train_dir):
        logger.error(f"Error: Data directory {train_dir} not found.")
        logger.error("Please run 'python benchmarks/prepare_data.py' first.")
        sys.exit(1)

    all_file_paths = get_image_paths(train_dir)
    logger.info(f"Found {len(all_file_paths)} training images.")

    criterion = nn.CrossEntropyLoss()

    # Define common transform for PyTorch
    pt_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    for config in active_configs:
        model_name = config["name"]
        epochs = config["epochs"]
        model_factory = config["factory"]

        logger.info(f"\n{'#' * 40}")
        logger.info(f"Starting Benchmark for: {model_name}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"{'#' * 40}")

        # Store results for this model
        pt_throughputs = []
        vx_throughputs = []
        pt_times = []
        vx_times = []

        for run in range(NUM_RUNS):
            logger.info(f"\n--- {model_name} Run {run + 1}/{NUM_RUNS} ---")

            # Measure PyTorch
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            model_pt = model_factory().to(DEVICE)
            optimizer_pt = optim.Adam(model_pt.parameters(), lr=0.001)

            pt_dataset = datasets.ImageFolder(train_dir, transform=pt_transform)
            pt_loader = DataLoader(
                pt_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=os.cpu_count(),  # Use all cores
                pin_memory=(DEVICE == "cuda"),
            )

            logger.info(f"Run {run + 1}: Evaluating PyTorch...")
            pt_tp, pt_time = train_loop(
                f"PyTorch_{model_name}",
                pt_loader,
                model_pt,
                criterion,
                optimizer_pt,
                epochs,
            )
            pt_throughputs.append(pt_tp)
            pt_times.append(pt_time)
            logger.info(f"  -> PyTorch: {pt_tp:.2f} img/s ({pt_time:.2f}s)")

            del model_pt, optimizer_pt, pt_loader  # Free resources

            # Measure Vantix
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            model_vx = model_factory().to(DEVICE)
            optimizer_vx = optim.Adam(model_vx.parameters(), lr=0.001)

            try:
                # Re-initialize loader for fairness
                vx_loader = VantixLoader(
                    all_file_paths,
                    width=IMG_SIZE,
                    height=IMG_SIZE,
                    batch_size=BATCH_SIZE,
                    augment=True,
                    shuffle=True,
                    queue_size=5,
                )

                logger.info(f"Run {run + 1}: Evaluating Vantix...")
                vx_tp, vx_time = train_loop(
                    f"Vantix_{model_name}",
                    vx_loader,
                    model_vx,
                    criterion,
                    optimizer_vx,
                    epochs,
                )
                vx_throughputs.append(vx_tp)
                vx_times.append(vx_time)
                logger.info(f"  -> Vantix : {vx_tp:.2f} img/s ({vx_time:.2f}s)")

                del model_vx, optimizer_vx, vx_loader

            except Exception as e:
                logger.error(f"Vantix Loader Failed: {e}")
                vx_throughputs.append(0.0)
                vx_times.append(0.0)
                if "model_vx" in locals():
                    del model_vx
                if "optimizer_vx" in locals():
                    del optimizer_vx

        # Process Statistics and Output for this model
        stats = print_stats_table(
            pt_throughputs, vx_throughputs, pt_times, vx_times, model_name
        )

        # Visualize for this model
        logger.info(f"Generating plots for {model_name}...")
        plot_throughput(stats["pytorch"], stats["vantix"], model_name)
        plot_time_savings(pt_times, vx_times, model_name)

    logger.info("\nAll Benchmarks complete.")


if __name__ == "__main__":
    main()
