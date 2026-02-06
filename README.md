# Vantix

**A blazingly fast Rust-powered image loader for Python.**

Vantix replaces standard PyTorch DataLoaders by shifting image decoding, resizing, and normalization to a multi-threaded Rust backend. It produces ready-to-use float32 tensors, bypassing the Python GIL and maximizing GPU throughput.

## Features
- **True Parallelism:** Uses `rayon` to saturate all CPU cores for image decoding.
- **SIMD Resizing:** Hardware-accelerated resizing via AVX2/NEON.
- **Zero-Copy:** Returns NumPy arrays that convert to PyTorch tensors without memory duplication.
- **Fused Pipeline:** Performs resizing, flipping (augmentation), and normalization in a single pass.

## Installation
Available soon via 
```bash
pip install vantix