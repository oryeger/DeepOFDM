# DeepOFDM
This repository contains the code for the DeepOFDM system, which includes 

DeepSIC code is based on: https://github.com/ShlezingerLab/deepsic-official

DeepRx code is based on: https://github.com/j991222/MIMO_JCESD

For running the code a yaml configuration file needs to be created,
then the following command can be used to run the training or evaluation
python -m

python_code.evaluate --config python_code/my_config.yaml

Default configuration file is config.yaml

The results will be saved in the folder scratchpad as csv files
