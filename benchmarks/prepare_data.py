import os
import logging
from torchvision import datasets
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CIFAR_IMAGES_DIR = os.path.join(DATA_ROOT, "cifar10_extracted")


def setup_cifar10_files(download_root, extract_root):
    """
    Downloads CIFAR-10 and extracts it to PNG files so we can test file I/O.
    """
    if os.path.exists(extract_root) and len(os.listdir(extract_root)) > 0:
        logger.info(f"Dataset seemingly ready at {extract_root}. Skipping extraction.")
        return

    logger.info("Downloading CIFAR-10...")
    train_set = datasets.CIFAR10(root=download_root, train=True, download=True)

    logger.info(f"Extracting images to {extract_root}...")
    os.makedirs(extract_root, exist_ok=True)
    train_dir = os.path.join(extract_root, "train")

    # Extract training data only for benchmark
    for idx, (img, label) in enumerate(tqdm(train_set, desc="Extracting")):
        class_dir = os.path.join(train_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f"{idx}.png"))


if __name__ == "__main__":
    setup_cifar10_files(DATA_ROOT, CIFAR_IMAGES_DIR)
    logger.info("Data preparation complete.")
