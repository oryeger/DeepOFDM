# DeepOFDM

This repository contains the code for the ESCNN (Element wise scaled CNN) system, published in:

Ory Eger and Nir Shlezinger: "Learning to Refine LLRs: Modular Neural Augmentation for MIMO-OFDM Receivers"

The ESCNN is a DNN which can either run standalone, or run as an augmentation to a model based or a data driven detector. 

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [python_code](#python_code)
    + [channel](#channel)
    + [detectors](#detectors)
   + [utils](#utils)  
- [Execution](#execution)
- [Environment Installation](#environment-installation)
  
# Introduction

We can augment four different detectors with the ESCNN:
1. LMMSE
2. Sphere Decoder (SD)
3. DeepSIC - based on: https://github.com/ShlezingerLab/deepsic-official  
4. DeepRx - based on: https://github.com/j991222/MIMO_JCESD  

We can test the ESCNN with different types of impairments. Three typical scenarios that we checked:
1. cfo = 0,    iqmm_gain = 0, iqmm_phase = 0 - All detectors perform well, ESCNN does not help much.
2. cfo = 0.15, iqmm_gain = 0, iqmm_phase = 0 - Un-augmented LMMSE, SD and DeepSIC don't work well, ESCNN helps.
3. cfo = 0,    iqmm_gain = 1, iqmm_phase = 5 - Un-augmented LMMSE and SD don't work well, ESCNN helps.


Note: The basis for this code was also the DeepSIC repository 


# Folders Structure

## python_code 

The python simulations: symbols generation, channel transmission and detection.

### channel 


Two types of MIMO channel models:

<u>TDL_model = True:</u>  
- Channel model based on 3GPP 38.901 TDL channel implemented using the  Nvidia Sionna simulation 
- Impairments (CFO, IQMM and clliping)
- AWGN
  
<u>TDL_model = False:</u>  
Just AWGN and impairments

The folder also includes OFDM modulation and demodulation.

### detectors 

Includes the following trainers and detectors:

1. **escnn** - the element wise scaled CNN data driven detector. The CNN kernel is defined by conf.kernel_size.  
   \* Has an option to use several iterations defined in conf.iterations, uses sequential training  
   \* Can run either in standalone mode or as augmentation to other detectors
2. **lmmse** - model based Linear Minimum Square Error detector
3. **sphere** - model based Sphere decoder
4. **deepsic** - the DeepSIC data driven detector with sequential training, uses a separate instance for each subcarrier, where each instance only looks at the signal of the corresponding subcarrier
5. **deeprx** - data driven detectors based on DeepRx
6. **deepsice2e** - *not commonly used* - like escnn, but the training is done jointly all users and all iterations (conf.full_e2e = True) or jointly just for all users but separately for each iteration (conf.full_e2e = False) 
7. **deepsicmb** - *not commonly used* - data driven detector like deepsic, where each instance looks at the signal of conf.kernel_size subcarriers
8. **deepstag** - *not commonly used* - data driven detector which has staggered iterations: ESCNN1 / DeepSICMB1 / ESCNN2 / DeepSICMB2 ... number of subcarriers used is defined by conf.stag_re_kernel_size

**trainer.py** - Is the backbone trainer which holds the most basic functions for the data driven detectors

conf.which_augment select which primary detectors the escnn augments:  
'AUGMENT_LMMSE' -   Probabilities at the escnn input are the output of lmmse  
'AUGMENT_SPHERE' -  Probabilities at the escnn input are the output of sphere  
'AUGMENT_DEEPSIC' - Probabilities at the escnn input are the output of deepsic  
'AUGMENT_DEEPRX' -  Probabilities at the escnn input are the output of deeprx  
'NO_AUGMENT' -      Probabilities at the escnn input are initialized to a constant (half) and the escnn runs in standalone mode

For checking transferability between primary detectors conf.override_augment_with_lmmse can be used.
When it is set to True, in inference the escnn gets the lmmse probabilities regardless of which primary dector the escnn was trained for.  

### utils

General utils, some examples: 
- plot_multiple_csvs.py - Can aggregate multiple csv results for different SNR values into one plot and also save the results to a mat file. 
- constants.py - Defines the OFDM Numeroligy (sampling rate, etc.) and some other constants


## analog
Contains the analog front end modeling: IQMM impairments and PA

## coding
Includes LDPC channel coding and crc  
Table 5.1.3.1-2 from 3GPP 38.214 is used to defined the MCSs, with addition of custom MCSs 28-30 which are 16QAM with high code-rate 


# Execution

For executing the code a yaml configuration file needs to be created,
then the following command can be used to run the training or evaluation

python -m python_code.evaluate --config python_code/my_config.yaml

Default configuration file is python_code/config.yaml

The results in the form of csv files and jpeg plots will be saved is the Scratchpad sibling directory. If the Scratchpad directory does not exist it will be created.

## Environment Installation

Open git bash and cd to a working directory of you choice.  
Clone this repository to your local machine.  
Open Anaconda prompt and navigate to the cloned repository.  
  
<u>Then do this sequence:</u>

Linux:  
conda create --name DeepOFDMs python=3.9.16  
conda activate DeepOFDMs  
pip install git+https://github.com/NVlabs/sionna.git@v0.19.2  
conda env update --file deepsics_env_linux.yml  
pip install tensorflow==2.13.0  
pip install scikit-commpy  
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
conda install pandas  
conda install pyyaml  
conda install python-dateutil
conda install conda-forge::libllvm12
ln -s /home/egero/miniconda3/envs/DeepOFDMs/lib/libLLVM-12.so /home/egero/miniconda3/envs/DeepOFDMs/lib/libLLVM.so
export DRJIT_LIBLLVM_PATH=/home/egero/miniconda3/envs/DeepOFDMs/lib/libLLVM.so

  

Windows:  
conda create --name DeepOFDMs python=3.9.16=h6244533_2  
conda activate DeepOFDMs  
pip install git+https://github.com/NVlabs/sionna.git@v0.19.2  
conda env update --file deepsics_env_windows.yml  
pip install tensorflow==2.13.0  
pip install scikit-commpy  
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
conda install pandas  
conda install pyyaml  
conda remove numpy  
conda install numpy  
conda install python-dateutil
conda install conda-forge::libllvm12  
When running on pytorch set the environment variable in the debug configuration, for example:
DRJIT_LIBLLVM_PATH = C:\Projects\Utils\clang+llvm-20.1.3-x86_64-pc-windows-msvc\bin\LLVM-C.dll

