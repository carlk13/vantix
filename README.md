# Vantix

**A fast image loader for Python, built in Rust.**

PyTorch's `DataLoader` is fine, but it hits a wall pretty fast when you have a lot of images — the Python GIL and slow JPEG decoding eat up a huge chunk of your GPU's potential throughput. Vantix moves all of that work (decoding, resizing, normalizing) into a multi-threaded Rust backend, hands you a ready-to-use `float32` tensor, and gets out of the way. On some workloads it's over **9× faster** than a standard PyTorch pipeline.

## How it works

The Rust core (`src/lib.rs`) uses:
- **`rayon`** — spawns one thread per CPU core, each decoding and resizing one image in parallel
- **`fast_image_resize`** — SIMD-accelerated resizing (AVX2 on x86, NEON on ARM) so bilinear resize is basically free
- **`pyo3` + `numpy`** — the output buffer is handed back to Python as a NumPy array with zero extra copies

On the Python side, `VantixLoader` (`python/vantix/loader.py`) wraps everything in a prefetch queue: a background thread keeps up to 3 batches queued so your training loop never waits on I/O. It also supports PyTorch DDP (multi-GPU) out of the box by sharding indices across ranks.

## Features

- **True parallelism** — bypasses the Python GIL entirely, all decoding happens in Rust threads
- **SIMD resizing** — hardware-accelerated, way faster than Pillow
- **Zero-copy tensor handoff** — the NumPy array converts to a PyTorch tensor without duplicating memory
- **Fused pipeline** — resize + optional random horizontal flip + normalize all in one pass
- **DDP ready** — automatically shards the dataset across GPUs when using distributed training
- **Drop-in replacement** — same interface as a standard PyTorch `IterableDataset`

## Quick start

### Install (from source, for now)

You need Rust and `maturin` installed first:

```bash
pip install maturin
maturin develop --release
```

Once it's published on PyPI:

```bash
pip install vantixloader
```

### Basic usage

```python
from vantixloader import VantixLoader
import glob

paths = glob.glob("data/train/**/*.jpg", recursive=True)

loader = VantixLoader(
    image_paths=paths,
    width=224,
    height=224,
    batch_size=64,
    augment=True,   # random horizontal flip
    shuffle=True,
)

for batch in loader:
    # batch is a float32 torch.Tensor of shape [B, 3, H, W]
    outputs = model(batch.to("cuda"))
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `image_paths` | — | List of paths to images on disk |
| `width` | `224` | Target width after resize |
| `height` | `224` | Target height after resize |
| `batch_size` | `64` | Images per batch |
| `augment` | `False` | Random horizontal flip (50% chance per image) |
| `shuffle` | `True` | Shuffle order every epoch |
| `transform` | `None` | Optional Python transform applied after loading |
| `queue_size` | `3` | Number of batches to prefetch in the background |

## Benchmarks

See [`BENCHMARK.md`](BENCHMARK.md) for full results. Short version:

| Dataset | PyTorch (img/s) | Vantix (img/s) | Speedup |
|---|---|---|---|
| CIFAR-Like (32 px) | ~1 750 | ~16 400 | **~9.4×** |
| ImageNet (224 px) | ~420 | ~1 020 | **~2.4×** |
| 4K Medical (2048→224) | ~65 | ~180 | **~2.8×** |

## Running the benchmarks yourself

```bash
# 1. Build the Rust extension
maturin develop --release

# 2. Download + prep CIFAR-10 test data
python benchmarks/prepare_data.py

# 3. Raw throughput (images/sec)
python benchmarks/benchmark_throughput.py

# 4. End-to-end training speedup
python benchmarks/benchmark_training.py
```

Charts are saved to `assets/`.

## Project structure

```
vantix/
├── src/lib.rs                  # Rust core: parallel decode, resize, normalize
├── python/vantixloader/
│   ├── __init__.py
│   └── loader.py               # VantixLoader — Python wrapper + prefetch queue
├── benchmarks/
│   ├── benchmark_throughput.py # Raw img/s benchmark
│   └── benchmark_training.py  # Full training loop speedup benchmark
├── assets/                     # Benchmark charts
├── Cargo.toml                  # Rust dependencies
└── pyproject.toml              # Python packaging (maturin)
```

## License

MIT — Carl Kemmerich
