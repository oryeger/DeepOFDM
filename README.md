# DeepOFDM

This repository contains the code for the ESCNN (Element wise scaled CNN) system.

The ESCNN is a DNN which can either run standalone, or run as an augmentation to a model based or a data driven detector.

# Papers Using This Code

| Paper | Status | Entry point | Workflow |
|---|---|---|---|
| O. Eger and N. Shlezinger, "Learning to Refine LLRs: Modular Neural Augmentation for MIMO-OFDM Receivers" ([IEEE Xplore](https://ieeexplore.ieee.org/document/11587582)) | Published, ICC 2026 | `evaluate.py` | Train and test in one run |
| O. Eger, G. C. Alexandropoulos, and N. Shlezinger, "Protocol-Aware Label-Free Online Adaptation of Neural-Augmented Receivers" | Submitted to ICC 2027 | `evaluate.py` → `ekf.py` | Pretrain, then track |

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [python_code](#python_code)
    + [channel](#channel)
    + [detectors](#detectors)
    + [utils](#utils)
  * [analog](#analog)
  * [coding](#coding)
- [Execution](#execution)
  * [evaluate.py: block-based train and test](#evaluatepy-block-based-train-and-test)
  * [ekf.py: streaming tracking from a pretrained checkpoint](#ekfpy-streaming-tracking-from-a-pretrained-checkpoint)
  * [Configuration keys per script](#configuration-keys-per-script)
- [Running the EKF Tracking](#running-the-ekf-tracking)
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

There are two entry points. They share the detectors, channel, coding and configuration code.
Both take a yaml configuration file. The default is `python_code/config.yaml`.

Always run from the repository root. The output and weights folders are resolved relative to the working directory.
Results (csv files and jpeg plots) are saved in the `Scratchpad` sibling directory (`../Scratchpad`). If it does not exist it is created.

## evaluate.py: block-based train and test

The channel is fixed per block. The ESCNN is trained on pilots (training loss set by `pretrain_loss`: `bce`, `tent`, `tsyn`, `gfmi`), then tested against the un-augmented detectors.
This is the entry point for the "Learning to Refine LLRs" paper.

```
python -m python_code.evaluate --config python_code/config.yaml
```

## ekf.py: streaming tracking from a pretrained checkpoint

The channel drifts over groups of slots. Each slot carries embedded 5G-style DMRS and CRC+LDPC-coded payload.
`ekf.py` loads ESCNN weights saved by `evaluate.py` and tracks them online. The tracking method is set by `weights_track_mode`.
`ekf.py` never trains from scratch.
This is the entry point for the "Protocol-Aware Label-Free Online Adaptation" paper. See [Running the EKF Tracking](#running-the-ekf-tracking).

```
python -m python_code.ekf --config python_code/config.yaml
```

## Configuration keys per script

A single `config.yaml` serves both scripts. Most keys are shared. These are script-specific:

| Key | evaluate.py | ekf.py |
|---|---|---|
| `pretrain_loss` | Training loss | Ignored |
| `save_escnn_weights` | Saves the trained weights | Ignored |
| `load_escnn_weights_tag` | Optional (load instead of train) | Required |
| `weights_track_mode`, `slots_per_group`, `calib_slots_per_group`, `escnn_ekf_*` | Ignored | Used |


# Running the EKF Tracking

## Step 1: Pretrain and save weights with evaluate.py

In the config set:

```yaml
save_escnn_weights: True
pretrain_loss: 'bce'
```

Run:

```
python -m python_code.evaluate --config python_code/config.yaml
```

One checkpoint is saved per SNR, to:

```
../Scratchpad/weights/<params>_s=<channel_seed>_SNR=<snr>_<tag>.pt
```

- `<params>` encodes the run configuration: channel, modulation, number of users, epochs, iterations, code rate, pilot ratio, augment mode, clipping and spatial correlation.
- `<tag>` is a 6-character hash of `<params>_s=<channel_seed>`. The SNR is not part of the hash, so all SNRs of one sweep share the same tag.
- The tag is printed when the weights are saved:
  `[ESCNN] saved weights, tag=<tag> -> <path>`
- With `AUGMENT_DEEPSIC` or `AUGMENT_DEEPRX`, the primary detector's weights are stored in the same `.pt` file.

## Step 2: Point ekf.py at the checkpoint

```yaml
load_escnn_weights_tag: '<tag>'   # or 'auto'
```

- `'auto'` searches `../Scratchpad/weights` for a checkpoint matching this run's channel model, channel seed, `which_augment`, number of users, number of antennas, `num_res`, modulation, IQMM, CFO and clipping. It fails if none matches.
- By default the checkpoint trained at the run's own `snr` is loaded. To load a different one:
  - `load_escnn_weights_snr_override: <snr>` always loads the checkpoint trained at this SNR.
  - `load_escnn_weights_snr_max_override: <snr>` loads the checkpoint trained at this SNR for any higher run SNR.
- If several files match, the most recent one is loaded.
- `which_augment` and the clipping level are checked against the checkpoint filename.

## Step 3: Choose what is tracked and how

```yaml
escnn_load_freeze: 'first_conv_and_scale_only'   # ESCNN parameters that stay frozen. The unfrozen ones form the tracked state.
weights_track_mode: 'ekfht'                      # Tracking method
```

Main tracking modes:

| Mode | Description |
|---|---|
| `ekfht` | Proposed hybrid: slots whose decode passes CRC are re-encoded and used as labels. Slots where decoding fails use the soft LDPC syndrome measurement. One EKF update per slot, then the failed slots are re-detected and re-decoded. |
| `ekfcrc` | EKF with CRC-verified labels only. No update when decoding fails. |
| `sgdbce` | SGD with BCE on the embedded DMRS pilots. |
| `notrack` | No adaptation. The loaded weights stay frozen (forces `escnn_load_freeze: 'all'`). |

The full list of modes is documented in the `ekf.py` docstring and in the `weights_track_mode` comment in `config.yaml`.

Run:

```
python -m python_code.ekf --config python_code/config.yaml
```


# Environment Installation

Open git bash and cd to a working directory of you choice.
Clone this repository to your local machine.
Open Anaconda prompt and navigate to the cloned repository.

Then do this sequence:

Linux:
```
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
```

Windows:
```
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
```

When running on pytorch set the environment variable in the debug configuration, for example:
```
DRJIT_LIBLLVM_PATH = C:\Projects\Utils\clang+llvm-20.1.3-x86_64-pc-windows-msvc\bin\LLVM-C.dll
```
