# Changelog

All notable changes to this project will be documented in this file.
Style based on [this](https://raw.githubusercontent.com/markdown-it/markdown-it/refs/heads/master/CHANGELOG.md)


## [Unreleased]
### Added
- Analyzer `--gpus` option to run the analysis on several GPUs at once, plus `--cpu`, `--batch-size` and `--loaders` options.
- `test/benchmark_batching.py` to time the analysis for different batch sizes.
- `test/compare_leaf_masks.py` to compare the compiled and Python leaf masking.
- Analyzer `--exp` option to run without the GUI (Linux/HPC), with `--overwrite` and `--gpus all` (Slurm `CUDA_VISIBLE_DEVICES`).
- NVIDIA CUDA support: the GPU backend is picked from the installed onnxruntime package (CUDA or DirectML).
- `requirements-linux.txt`.

### Changed
- Analyzer classifies sub-images in batches instead of one by one, and loads/masks the next images in parallel while the GPU is busy.
- Models with a fixed batch size of 1 are made batch-dynamic at load time (new `onnx` dependency).
- Analyzer falls back to the pure Python leaf masking (`leaf_mask.py`) when the compiled module can't be loaded (non-Windows or Python other than 3.11).
- `leaf_mask.py` is ~10x faster (no per-pixel Python loops, same output) and works with NumPy 2.

## [0.5.0] - 2025-01-11
### Added
- First beta release of the analyzer and the thresholder scripts.