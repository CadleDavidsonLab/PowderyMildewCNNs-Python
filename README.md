# Blackbird Analyzer scripts
[![Python 3.11](https://img.shields.io/badge/python-3.11_ONLY-red.svg)](https://www.python.org/downloads/release/python-31111/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)



This repository contains a set of Python-based scripts for Blackbird images analysis based on the original MATLAB scripts. The OpenCV library is used on some of the scripts found in this repository.

The trained models must be converted to ONNX in order to be able to be used with the scripts in this repository. If you need to use any pre-trained model, contact the Grape Genetics Research Unit (GGRU) of USDA-ARS. In addition, contact Dani Martinez for the model conversion of your own MATLAB legacy models. In the **models** folder, there are two image classifier models: one is used to detect Powdery Mildew, and the other to detect spores presence. Both require Blackbird-taken images as input. For more questions, please contact GGRU.

> Use the "Issues" tab to publicly post any issue or question you might have about this repository.

## Scripts
The following list enumerates the scripts and tools that are packed within this repository.
* **Analyzer:** The Analyzer script uses a previously trained CNN model to quantify disease from Blackbird experiment folders containing the stacked images. The script predicts the "Infection" score (0.0 to 1.0) for each sub-image which is a sub-division of the large sample image.
* **Thresholder:** The Tresholder process the results obtained from the Analyzer script following a custom tresholding policy in order to deliver the final results in an Excel file.
* **Labeler:** FUTURE WORK
* **Trainer:** FUTURE WORK



## Installation steps
Strictly follow the following steps to install all required software required to correctly run the scripts in this repository.

### 1 - Install Python
Download **Python 3.11** from the official webpage: https://www.python.org/downloads/windows/ .<br />
Or [click here](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) to download it directly.

> **IMPORTANT:** For the moment, these scripts required the exact version of Python 3.11 due to the Pybind11-compiled leaf masking function. Support for other Python versions may be added in the future.

1. Start the installation by running the downloaded Python installer.
2. Check the "Add python.exe to PATH".
3. Click on "Install Now".

> **NOTE:**
> If you previously installed another Python version, you might need to uninstall it first or manually modify the PATH environmental variable to point to the correct Python version.

### 2 - Set up a Python virtual environment

A Python virtual environment is convenient in order to create isolated workspaces where you want to install specific Python packages that might need to fulfill specific version dependencies. This is very convenient to avoid conflicts between packages that require different versions of the same dependency. Therefore, it is recommende to create a Python virtual environment to use these scripts. In a nutshell, a Python virtual environment is a folder where all the packages are installed. Therefore, if something "breaks" this environment, the user can easily delete it by simply deleting the folder.

* Click on Windows search bar and start typing "powershell", then click on the "Windows Powershell" icon to start the command terminal.
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
The scripts found in this repository requires to install very specific Python packages as dependencies. These dependencies are listed in the "requirements.txt" file in the repository folder. The users can use this file to install them all at once by calling the following command: ```pip install -r requirements.txt```

First of all, make sure to change your current directory in the Powershell terminal to the folder where the scripts are located by using the **"cd"** command.
- For example (Windows): ```cd C:\blackbird-analyzer```

>**IMPORTANT:** Make sure you have activated the Python virtual environment. You should see the virtual environment name in parenthesis at the begining of the prompt bedore calling this command.
```
(blackbird_env) PS C:\blackbird-analyzer> pip install -r requirements.txt
``` 

## Usage


### Analyzer
The Blackbird analyzer script is a Python carbon-copy version or the original MATLAB "PM_analyzer.m" script with some added improvements on performance. Make sure to have the correct Python virtual environment activated before calling the script. The path to the CNN models used for the analysis must be passed explicitly when calling the script:```python analyzer.py <C:\path\to\model.onnx>```

For example:
```
(blackbird_env) PS C:\blackbird-analyzer> python analyzer.py models\PMnet.onnx
```

When the graphical user interface (GUI) is shown, the powershell terminal will remain blocked until the window is closed.

>**IMPORTANT:**
> In order to run the models on the GPU (DirectML), you need to have Windows 11 or Windows 10 with the latest updates installed in your system.

### Thresholder
The thresholder script does not have a graphical interface and it is easily called by command line.

```
usage: thresholder.py [-h] [-o <OUT_XLSX>] [-lo <LOW_TH>] [-hi <LOW_TH>] <RES_PATH>

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

This new tresholder script outputs different data compared to the original MATLAB script. For each timepoint and sample ID, there are four given values:
* Percentage of infected-rated sub-images (%): INF / INF + CLR
* Absolute number of sub-images labeled as infected (INF): ```"score" >= HIGH_TH```
* Absolute number of sub-images labeled as clear (CLR): ```score < LOW_TH```
* Absolute number of total analyzed sub-images (ALL) including the ones rated in between both given tresholds.
