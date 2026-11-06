# Blackbird Analyzer scripts
[![Python 3.11 (Windows)](https://img.shields.io/badge/python-3.11_(Windows)-blue.svg)](https://www.python.org/downloads/release/python-31111/)
[![Python 3.10+ (Linux)](https://img.shields.io/badge/python-3.10+_(Linux)-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)



This repository contains a set of Python-based scripts for Blackbird image analysis based on the original MATLAB scripts. The OpenCV library is used on some of the scripts found in this repository.  Details of the original MATLAB pipeline can be found [in this publication.](https://doi.org/10.34133/2019/9209727)

The trained models must be converted to ONNX in order to be able to be used with the scripts in this repository. If you need to use any pre-trained model, contact the Grape Genetics Research Unit (GGRU) of the USDA-ARS. In addition, contact Dani Martinez for the model conversion of your own MATLAB legacy models. In the **models** folder, there are two image classifier models: one (Hyphal2019Net2) is used to detect grape powdery mildew (*Erysiphe necator*) hyphae, and the other (SporeNet3) to detect grape powdery mildew conidia. Both require Blackbird-taken images as input. For more questions, please contact the GGRU.

> Use the "Issues" tab to publicly post any issue or question you might have about this repository.

## Scripts
The following list enumerates the scripts and tools that are packed within this repository.
* **Analyzer:** The Analyzer script uses a previously-trained CNN model to quantify disease from Blackbird experiment folders containing the stacked images. The script predicts the "Infection" score (0.0 to 1.0) for each sub-image which is a sub-division of the large sample image.
* **Thresholder:** The Thresholder processes the results obtained from the Analyzer script following a custom thresholding policy in order to deliver the final results in an Excel file.
* **Summarizer:** The Summarizer collects the results of many experiments (and several models, e.g. sporulation and hyphae) into a single CSV table with one row per sample, including a combined severity from the union of the models' infected areas.
* **Labeler:** FUTURE WORK
* **Trainer:** FUTURE WORK



## How the analysis works
For every sample image (one leaf disk), the Analyzer:
1. Loads the image and computes a **leaf mask** that separates the in-focus leaf tissue from the background (see [Leaf masking](#leaf-masking)).
2. Divides the image into a grid of non-overlapping **224x224 pixel sub-images** (any leftover pixels are split evenly between both borders). A full 8254x5502 Blackbird image gives a 36x24 grid (864 sub-images).
3. Skips the sub-images where less than 70% of the pixels belong to the leaf mask (score set to NaN).
4. Classifies the remaining sub-images with the CNN, in batches, and keeps the **"Infected" probability** (0.0 to 1.0) of each one.

The result of each sample is a score map with one value per sub-image, which the Thresholder then turns into infection percentages per sample.

>**NOTE:** The models output two class probabilities, but the class order is not stored in the ONNX files. The Analyzer currently identifies the "Infected" output by running a black image through the model and taking the class with the lowest probability. Check this assumption when using a new model (e.g. with `analyzeNetwork()` in MATLAB).

### Experiment folder structure
The Analyzer expects one folder per experiment, with one sub-folder per timepoint and one sub-folder per tray inside it:
```
experiment_name/
├── 12-06-2022_5dpi/            Timepoint folder: name starts with the date as MM-DD-YYYY, optionally followed by "_<anything>"
│   ├── 1/                      Tray folder (any name)
│   │   ├── 1-<anything>.png    Sample images: name starts with the sample number (1 to 351) followed by "-"
│   │   ├── 2-<anything>.png
│   │   └── ...
│   └── 2/
│       └── ...
├── 12-08-2022_7dpi/
│   └── ...
└── results.msgpack             Written by the Analyzer
```
* Timepoints are processed in date order. Every sub-folder of the experiment folder must be a timepoint folder named as above, otherwise the experiment can't be loaded.
* Only `.png` files are analyzed. Each tray has room for up to 351 samples; missing sample numbers are simply skipped.


## Installation steps
Strictly follow the following steps to install all required software to correctly run the scripts in this repository.

### 1 - Install Python
Download **Python 3.11** from the official webpage: https://www.python.org/downloads/windows/ .<br />
Or [click here](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) to download it directly.

> **IMPORTANT:** On Windows, use exactly Python 3.11 so that the Pybind11-compiled leaf masking function in **cpp_functions** can be loaded. With other Python versions (or on Linux/macOS) the analyzer falls back to the pure Python implementation in `leaf_mask.py`, which needs Python 3.10 or newer. See [Leaf masking](#leaf-masking).

1. Start the installation by running the downloaded Python installer.
2. Check the "Add python.exe to PATH".
3. Click on "Install Now".

> **NOTE:**
> If you previously installed another Python version, you might need to uninstall it first or manually modify the PATH environmental variable to point to the correct Python version.

### 2 - Set up a Python virtual environment

A Python virtual environment is convenient in order to create isolated workspaces where you want to install specific Python packages that might need to fulfill specific version dependencies. This is very convenient to avoid conflicts between packages that require different versions of the same dependency. Therefore, it is recommended to create a Python virtual environment to use these scripts. In a nutshell, a Python virtual environment is a folder where all the packages are installed. Therefore, if something "breaks" this environment, the user can easily delete it by simply deleting the folder.

* Click on the Windows search bar and start typing "powershell", then click on the "Windows Powershell" icon to start the command terminal.
* When the terminal opens, create a Python virtual environment by running the following command: ```python -m venv C:\path\to\new\virtual\environment_name```
    - For example: ```python -m venv C:\blackbird_env```
* Once created, activate the environment by running the following command: ```C:\path\to\new\virtual\environment_name\Scripts\Activate.ps1```
    - For example: ```C:\blackbird_env\Scripts\Activate.ps1```
* Once activated, the environment name will appear in parenthesis at the beginning of the prompt.

> **NOTE:**
> If you got an error when activating the environment, you might need to give execution permissions by calling this (only required in first time):

```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

> **NOTE:** The created virtual enviroment will need to be activated every time you need to use the scripts from this repository.

### 3 - Download this repository
In the left margin of this Github page, find the **Releases** section, and download the latest ZIP package with all the necessary files to run the scripts. Then, extract the files in a local folder (ex: C:\blackbird-analyzer).

> **NOTE:** The analyzer script requires a compiled OpenCV library file in the **cpp_functions** folder. This file is not included in this code repository as it should be compiled by each user. Nevertheless, in order to "keep things easy", this library file has been included in the release package. In the future we might provide scripts to download and compile OpenCV since it is the correct way to do it.


### 4 - Install dependencies
The scripts found in this repository require the installation of very specific Python packages as dependencies. These dependencies are listed in the "requirements.txt" file in the repository folder. Users can use this file to install them all at once by calling the following command: ```pip install -r requirements.txt```

First of all, make sure to change your current directory in the Powershell terminal to the folder where the scripts are located by using the **"cd"** command.
- For example (Windows): ```cd C:\blackbird-analyzer```

>**IMPORTANT:** Make sure you have activated the Python virtual environment. You should see the virtual environment name in parentheses at the begining of the prompt before calling this command.
```
(blackbird_env) PS C:\blackbird-analyzer> pip install -r requirements.txt
``` 

## Usage


### Analyzer
The Blackbird analyzer script is a Python carbon-copy version or the original MATLAB "PM_analyzer.m" script with some added improvements on performance. Make sure to have the correct Python virtual environment activated before calling the script. The path to the CNN models used for the analysis must be passed explicitly when calling the script:```python analyzer.py <C:\path\to\model.onnx>```

For example:
```
(blackbird_env) PS C:\blackbird-analyzer> python analyzer.py models\Hyphal2019Net2.onnx
```

When the graphical user interface (GUI) is shown, the Powershell terminal will remain blocked until the window is closed.

Each sample image is cut into 224x224 sub-images that are classified in batches on the GPU, while the next images are loaded and masked in parallel on the CPU. The following optional arguments control how the analysis runs:

```
usage: analyzer.py [-h] [-e <EXP_DIR> ...] [-r <REPO_DIR> ...] [--overwrite] [-g <IDS>] [--cpu] [-b <N>] [-l <N>] <ONNX_MODEL> [<ONNX_MODEL> ...]

  <ONNX_MODEL> ...                        One model for the GUI; one or more models with --exp/--repo
  -e <EXP_DIR> ..., --exp <EXP_DIR> ...   Experiment folder(s) to analyze without the GUI (see Linux / HPC below)
  -r <REPO_DIR> ..., --repo <REPO_DIR> .. Folder(s) containing experiment folders, all analyzed without the GUI
  --overwrite                             With --exp/--repo, redo models that already have results instead of skipping them
  -g <IDS>, --gpus <IDS>                  Comma-separated GPU ids, e.g. 0,1, or 'all' for every GPU in CUDA_VISIBLE_DEVICES (default: 0)
  --cpu                                   Run the analysis on the CPU instead of the GPU(s)
  -b <N>, --batch-size <N>                Sub-images classified per inference run (default: 64)
  -l <N>, --loaders <N>                   Image loading/masking threads per device (default: 2)
```

In the GUI, click **Open** to select an experiment folder, check the number of images, trays and timepoints found, and click **Start analysis**. The progress bar shows the estimated time to completion, and **Stop analysis** cancels the run (no results are saved). When the analysis finishes, the results are saved as `results.msgpack` inside the experiment folder. If the folder already has results, the GUI asks before overwriting them.

To use several GPUs, list their ids. One worker process is started per GPU and the sample images are shared between them:
```
(blackbird_env) PS C:\blackbird-analyzer> python analyzer.py models\Hyphal2019Net2.onnx --gpus 0,1
```

On Windows, GPU ids follow the DirectX adapter order (usually the order in the Windows Task Manager "Performance" tab). If a GPU runs out of memory, lower the batch size. To find the best batch size for a given GPU, run `python test\benchmark_batching.py <MODEL> <IMAGE> --gpu 0`.

>**NOTE:** Models exported from MATLAB with a fixed batch size of 1 are converted automatically at load time to accept batches (requires the `onnx` package from requirements.txt).

>**IMPORTANT:**
> In order to run the models on the GPU (DirectML), you need to have Windows 11 or Windows 10 with the latest updates installed in your system.

### Analyzer on Linux / HPC (no GUI)
The analyzer can also run without the GUI on Linux, e.g. on an HPC cluster with NVIDIA GPUs. On systems where the compiled leaf masking module (Windows, Python 3.11) can't be loaded, the pure Python implementation in `leaf_mask.py` is used instead, so any recent Python 3 version works.

Install the Linux dependencies in a virtual environment (`onnxruntime-gpu` replaces `onnxruntime-directml`, and must match the CUDA/cuDNN versions available on the cluster, see the [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)):
```
python3 -m venv ~/blackbird_env
source ~/blackbird_env/bin/activate
pip install -r requirements-linux.txt
```

Then pass one or more experiment folders with `--exp`, or a folder that contains many experiments with `--repo`:
```
python analyzer.py models/SporeNet3.onnx --exp /path/to/experiment1 /path/to/experiment2 --gpus 0,1
python analyzer.py models/SporeNet3.onnx models/Hyphal2019Net2.onnx --repo /path/to/repository --gpus 0,1
```
* Without the GUI, the results of each model are written to `results_<MODEL_NAME>.msgpack` inside each experiment folder (e.g. `results_SporeNet3.msgpack`), so the results of different models don't overwrite each other. The format is the same as the GUI's `results.msgpack`, and the Thresholder works with either.
* When several models are given, each image is loaded and masked only once and its sub-images are classified by all the models. This is faster than running the models separately, and all the models score exactly the same sub-images (needed for the [union severity](#summarizer)).
* Models that already have results in an experiment are skipped unless `--overwrite` is given, so an interrupted run can simply be started again and it continues where it stopped.
* Sub-folders that are not valid experiments are reported and skipped, and the run continues with the next experiment.

Example Slurm job using every GPU allocated to it (`--gpus all` reads `CUDA_VISIBLE_DEVICES`, set by Slurm):
```bash
#!/bin/bash
#SBATCH --job-name=blackbird
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

module load cuda cudnn          # Module names depend on the cluster
source ~/blackbird_env/bin/activate
cd ~/PowderyMildewCNNs-Python
python analyzer.py models/SporeNet3.onnx models/Hyphal2019Net2.onnx --repo /path/to/repository --gpus all --loaders 3
python summarize.py /path/to/repository -o /path/to/repository/summary.csv
```

Request about `--loaders` CPU cores per GPU (plus one), since image loading and leaf masking run on the CPU. The script exits with a non-zero code if any experiment fails.

### Results file
`results.msgpack` (GUI) and `results_<MODEL_NAME>.msgpack` (`--exp`/`--repo`) are [MessagePack](https://msgpack.org/) files with the score maps of all samples, nested by timepoint and tray: `results[timepoint][tray][sample_number - 1]`. Each entry is:
* `None` if there is no image for that sample number.
* `[sample_id, score_map]` otherwise, where `sample_id` is the image filename without extension, and `score_map` is a 2D `float32` array (one value per sub-image, e.g. 24x36) with the "Infected" probability, or NaN for sub-images that were not analyzed. `score_map` is `None` if no leaf was found in the image.

The score maps are stored with `msgpack-numpy`, so it must be enabled to read them back in Python:
```python
import msgpack
import msgpack_numpy as mnp
import numpy as np
mnp.patch()

with open("results.msgpack", "rb") as f:
    results = msgpack.unpackb(f.read())

for timepoint, trays in results.items():
    for tray, samples in trays.items():
        for sample in samples:
            if sample is None or sample[1] is None:
                continue
            sample_id, score_map = sample
            print(timepoint, tray, sample_id, "analyzed sub-images:", np.count_nonzero(~np.isnan(score_map)))
```

### Leaf masking
The leaf mask marks the in-focus leaf tissue of each image. It measures the local sharpness of the image (standard deviation of 50x50 pixel windows), thresholds it relative to the sharpest region (`MASKING_RL_TH = 0.2` in `analyzer.py`), fills holes, erodes the borders and keeps only the largest connected region. Images where that region covers less than 15% of the image are reported as "No sample found" and skipped.

There are two implementations of the same algorithm:
* `cpp_functions/leaf_masking.cp311-win_amd64.pyd`: compiled C++ (Pybind11 + OpenCV), only loads on Windows with Python 3.11. Used when available.
* `leaf_mask.py`: pure Python + OpenCV, used on any other system. It takes about 1-2 seconds per full image.

To check that both give the same mask on your images, run this on a Windows machine with Python 3.11:
```
(blackbird_env) PS C:\blackbird-analyzer> python test\compare_leaf_masks.py <IMAGE> [<IMAGE> ...]
```
It reports the number of differing mask pixels and of differing analyzed sub-images.

### Performance tuning
* **Batch size (`-b`)**: larger batches keep the GPU busier, up to the GPU memory limit. Use the benchmark script to measure the throughput for several batch sizes on a given GPU:
  ```
  python test/benchmark_batching.py models/SporeNet3.onnx <IMAGE> --gpu 0 --batch-sizes 16,64,128,256
  ```
* **Loaders (`-l`)**: each image takes ~1-2 s of CPU time to load and mask. If the GPU is idle part of the time (e.g. low utilization in `nvidia-smi` or the Task Manager), increase the number of loaders, as long as there are enough CPU cores.
* **Multiple GPUs (`-g`)**: images are shared dynamically between GPUs, so the total throughput grows with the number of GPUs as long as the disk and CPU keep up.
* **Memory**: each GPU worker keeps a few full images in memory at the same time (roughly 1 GB per GPU with the default settings). Lower `-l` if the system runs out of RAM.

### Troubleshooting
* **"The installed onnxruntime has no GPU support"**: the installed `onnxruntime` package is CPU-only. Install `onnxruntime-directml` (Windows, from requirements.txt) or `onnxruntime-gpu` (Linux, from requirements-linux.txt) in the active virtual environment, and make sure the plain `onnxruntime` package is not installed alongside it (`pip uninstall onnxruntime`).
* **"... could not start on GPU N"**: the GPU id doesn't exist, or (on Linux) the CUDA/cuDNN libraries don't match the installed `onnxruntime-gpu` version. Check the GPUs with `nvidia-smi` and the loaded modules with `module list`.
* **GPU out of memory**: lower the batch size, e.g. `-b 16`.
* **"Tkinter is not available on this system"**: the Python installation has no GUI support (common on HPC nodes). Use `--exp` to run without the GUI.
* **"No sample found in ..."**: no leaf was detected in that image (empty well, out of focus or very small leaf disk). Its result is stored as `None` and it is ignored by the Thresholder.
* **Experiment can't be loaded / date error**: some sub-folder of the experiment folder is not named `MM-DD-YYYY` or `MM-DD-YYYY_<anything>`. See [Experiment folder structure](#experiment-folder-structure).


### Thresholder
The thresholder script does not have a graphical interface and is easily called by command line.

```
usage: thresholder.py [-h] [-o <OUT_XLSX>] [-lo <LOW_TH>] [-hi <HIGH_TH>] <RES_PATH>

Blackbird Results Thresholder

positional arguments:
  <RES_PATH>            Path to experiment results as a *.msgpack file.

options:
  -h, --help            show this help message and exit
  -o <OUT_XLSX>, --out <OUT_XLSX>
                        Path where the Excel file will be placed and its filename (incl. .xlsx extension).
  -lo <LOW_TH>, --low <LOW_TH>
                        Specify lower threshold between 0 and 1
  -hi <HIGH_TH>, --high <HIGH_TH>
                        Specify higher threshold between 0 and 1
```

Example usage:
```
(blackbird_env) PS C:\blackbird-analyzer> python thresholder.py D:\stacked\test_experiment\results.msgpack -lo 0.2 -hi 0.8 -o .\output.xlsx
```

The thresholder works with both `results.msgpack` (GUI) and `results_<MODEL_NAME>.msgpack` (`--exp`/`--repo`) files. This new thresholder script outputs different data compared to the original MATLAB script. For each timepoint and sample ID, there are four given values:
* Percentage of infected-rated sub-images (%): INF / INF + CLR
* Absolute number of sub-images labeled as infected (INF): ```"score" >= HIGH_TH```
* Absolute number of sub-images labeled as clear (CLR): ```score < LOW_TH```
* Absolute number of total analyzed sub-images (ALL) including the ones rated in between both given thresholds.

### Summarizer
The summarizer collects the `results_<MODEL_NAME>.msgpack` files of one or many experiments into a single CSV table, ready for R, pandas or Excel. It takes repository folders (containing experiment folders) or experiment folders:

```
usage: summarize.py [-h] [-o <OUT_CSV>] [-m <MODEL_NAME> ...] [-u <MODEL_NAME> ...] [-lo <LOW_TH>] [-hi <HIGH_TH>] <DIR> [<DIR> ...]

  -o <OUT_CSV>, --out <OUT_CSV>          Path of the resulting CSV file (default: summary.csv)
  -m <MODEL_NAME> ..., --models ...      Models to include, by name (default: all models found)
  -u <MODEL_NAME> ..., --union ...       Models combined in the union severity (default: all included models, if two or more)
  -lo <LOW_TH>, --low <LOW_TH>           Sub-images with a score below this are clear (default: 0.5)
  -hi <HIGH_TH>, --high <HIGH_TH>        Sub-images with a score equal or above this are infected (default: 0.5)
```

Example, after analyzing a repository with both models:
```
python summarize.py /path/to/repository -o summary.csv
```

The table has one row per sample image, with its metadata (`experiment`, `timepoint` folder name, `date`, `tray`, `sample_number`, `sample_id` = filename without extension), and for each model:
* `<MODEL>_analyzed`: number of analyzed (on-leaf) sub-images.
* `<MODEL>_infected` / `<MODEL>_clear`: sub-images with `score >= HIGH_TH` / `score < LOW_TH`.
* `<MODEL>_pct_infected`: `infected / (infected + clear) * 100`, the same percentage as the Thresholder (empty if there is nothing to rate, e.g. no leaf found).
* `<MODEL>_mean_score`: mean score of the analyzed sub-images.

When there are two or more models (e.g. `SporeNet3` for sporulation and `Hyphal2019Net2` for hyphae), the **union severity** combines them: a sub-image counts as infected if **any** of the models rates it as infected, and as clear only if **all** of them rate it as clear. It is reported as `union_analyzed`, `union_infected`, `union_clear` and `union_pct_infected`, computed the same way as above. In other words, `union_infected = spore_infected + hyphal_infected - (sub-images infected by both)`, so the union is the proportion of the leaf area showing sporulation, hyphae or both. Since both models are run on the same images and leaf mask, they rate exactly the same sub-images.

Experiments where a model has not been run yet are included with that model's (and the union's) columns left empty.
