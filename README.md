This repository contains the code for the ESCNN (Element wise scaled CNN) system, published in:

Ory Eger and Nir Shlezinger: "Learning to Refine LLRs: Modular Neural Augmentation for MIMO-OFDM Receivers"

The ESCNN augments some model based or data driven  detectors. Among the detectors that have been used are DeepSIC and DeepRx.  

DeepSIC implementation is based on: https://github.com/ShlezingerLab/deepsic-official

DeepRx implementation is based on: https://github.com/j991222/MIMO_JCESD

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [python_code](#python_code)
    + [channel](#channel)
    + [detectors](#detectors)
    + [utils](#utils)
  * [dir_definitions](#dir_definitions)
- [Execution](#execution)
  * [Environment Installation](#environment-installation)
  
  # Introduction

Note that this python implementation deviates from the [basic one](https://arxiv.org/pdf/2002.03214.pdf) in the basic DNN implementaion: Here it is only (1) Linear 1X64, (2) ReLU, (3) Linear 64X64, as  opposed to the three layers in the paper for the sequential version. Also, the learning rate is 5e-3 instead of 1e-2. Note that these changes obtain even better results on this setup, than the ones reported in the paper. These hyperparameters should be chosen to fit the complexity of the simulated channel. 

Also, note that the simulated setup here is a sequential transmission of pilots + info in each block coherence, which is different than the one in the original paper (that has only unlabeled info bits and uses error correction codes to correct errors and train on the decoded packet). It is more convenient in our opinion for learning purposes. 

# Folders Structure

## python_code 

The python simulations of the simplified communication chain: symbols generation, channel transmission and detection.

### channel 

Includes the symbols generation and transmission part, up to the creation of the dataset composed of (transmitted, received) tuples in the channel_dataset wrapper class. The modulation is done in the modulator file.

### detectors 

Includes the next files:

(1) The backbone trainer.py which holds the most basic functions, including the network initialization and the sequential transmission in the channel and BER calculation. The trainer is a wrapper for the training and evaluation of the detector. Trainer holds the training, sequential evaluation of pilot + info blocks. It also holds the main function 'eval' that trains the detector and evaluates it, returning a list of coded ber/ser per block.

(2) The DeepSIC trainer, which focuses on the online sequential training part (layer-by-layer). Refer to Algorithm 3 in the paper for more details.

(3) The DeepSIC detector, which is the basic cell that runs the priors through the deep neural network, and later propagates these values through the iterations.

### utils

Extra utils for calculating the accuracy over the BER metric; several constants; and the config singleton class.
The config works by the [singleton design pattern](https://en.wikipedia.org/wiki/Singleton_pattern). Check the link if unfamiliar.

The config is accessible from every module in the package, featuring the next parameters:
1. seed - random number generator seed. Integer.
2. fading_in_channel - whether to use fading. Relevant only to the synthetic channel. Boolean flag.
3. snr - signal-to-noise ratio, determines the variance properties of the noise, in dB. Float.
4. block_length - number of coherence block bits, total size of pilot + data. Integer.
5. pilot_size - number of pilot bits. Integer.
6. blocks_num - number of blocks in the tranmission. Integer.

## dir_definitions 

Definitions of relative directories.

# Execution

For executing the code a yaml configuration file needs to be created,
then the following command can be used to run the training or evaluation

python -m python_code.evaluate --config python_code/my_config.yaml

Default configuration file is config.yaml

The results in the form of csv files and jpeg plot will be saved is the Scratchpad sibling directory. If the Scratchpad directory does not exist it will be created.

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

