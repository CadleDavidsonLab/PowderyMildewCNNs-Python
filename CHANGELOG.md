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
- Analyzer accepts several models in headless mode: each image is loaded and masked once and classified by all of them. Results are saved per model as `results_<MODEL_NAME>.msgpack`.
- Analyzer `--repo` option to analyze every experiment in a folder, skipping models that already have results (resumable).
- `summarize.py` to collect the results of many experiments and models into one CSV, with a union severity combining the models (e.g. sporulation + hyphae).

### Changed
- Analyzer classifies sub-images in batches instead of one by one, and loads/masks the next images in parallel while the GPU is busy.
- Models with a fixed batch size of 1 are made batch-dynamic at load time (new `onnx` dependency).
- Analyzer falls back to the pure Python leaf masking (`leaf_mask.py`) when the compiled module can't be loaded (non-Windows or Python other than 3.11).
- Headless (`--exp`) results are saved as `results_<MODEL_NAME>.msgpack` instead of `results.msgpack` (the GUI still writes `results.msgpack`).
- The analyzer reports invalid experiment folders with a clear message (error dialog in the GUI) instead of failing silently, and ignores hidden files and folders.
- `leaf_mask.py` is ~10x faster (no per-pixel Python loops, same output) and works with NumPy 2.

## [0.5.0] - 2025-01-11
### Added
- First beta release of the analyzer and the thresholder scripts.