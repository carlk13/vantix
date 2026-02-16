import pytest
import torch
from PIL import Image
from vantix import VantixLoader


@pytest.fixture
def dummy_data(tmp_path):
    """Creates a temporary directory with dummy images."""
    data_dir = tmp_path / "images"
    data_dir.mkdir()

    paths = []
    # Create 10 dummy images
    for i in range(10):
        img = Image.new("RGB", (100, 100), color=(i * 10, i * 10, i * 10))
        path = data_dir / f"img_{i}.png"
        img.save(path)
        paths.append(str(path))

    return paths


def test_loader_basic(dummy_data):
    """Test that VantixLoader initializes and returns batches with correct shape."""
    batch_size = 4
    width = 64
    height = 64

    loader = VantixLoader(
        dummy_data,
        width=width,
        height=height,
        batch_size=batch_size,
        augment=False,
        shuffle=False,
    )

    # Iterate once
    batch_count = 0
    for batch in loader:
        assert isinstance(batch, torch.Tensor)
        # Check shape: [batch_size, 3, height, width]
        assert batch.shape[1] == 3
        assert batch.shape[2] == height
        assert batch.shape[3] == width
        batch_count += 1

    expected_batches = (len(dummy_data) + batch_size - 1) // batch_size
    assert batch_count == expected_batches


def test_loader_augmentation(dummy_data):
    """Test that loader runs with augmentation enabled."""
    loader = VantixLoader(dummy_data, width=32, height=32, batch_size=2, augment=True)

    for batch in loader:
        assert batch.shape[1:] == (3, 32, 32)
        break


def test_loader_len(dummy_data):
    """Test __len__ method."""
    batch_size = 3
    loader = VantixLoader(dummy_data, batch_size=batch_size)
    expected = (len(dummy_data) + 2) // 3
    assert len(loader) == expected
