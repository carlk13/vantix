# Vantix Benchmarks

This directory contains benchmarking scripts to compare `VantixLoader` against standard PyTorch `DataLoader` with `PIL` backend.
We use real CIFAR-10 data (extracted to `.png` files on disk) to simulate a realistic I/O bottleneck scenario.

## Prerequisites

Ensure you have installed the package in release mode:

```bash
maturin develop --release
```

## Running the Benchmark

### 1. Prepare Data
First, download and extract the CIFAR-10 dataset. This script will save images as individual PNG files in `data/cifar10_extracted`.

```bash
python benchmarks/prepare_data.py
```

### 2. Run Comparison
Run the benchmark script. This will train a lightweight MobileNetV3 (on CPU) for a few epochs using both loaders and report the throughput (Images/sec).

```bash
python benchmarks/compare_speed.py
```
