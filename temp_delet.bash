#!/bin/bash

BASE="/home/egero/Projects/DeepOFDM"

rm -f $BASE/python_code/TDL_A_16QAM.yaml
rm -f $BASE/python_code/TLD_channel.py
rm -f $BASE/python_code/TLD_channel_example.py
rm -f $BASE/python_code/config_orig.yaml
rm -f $BASE/python_code/config_temp.yaml
rm -f $BASE/python_code/detectors/sphere/sphere_64qam_fourbits.py
rm -f $BASE/python_code/mimo_channel_dataset.py
rm -f $BASE/python_code/modulator.py
rm -f $BASE/python_code/sed_channel.py
rm -f $BASE/python_code/temp.py
rm -f $BASE/python_code/utils/run_deepsic_batch_back.bash
rm -f $BASE/python_code/utils/test_temp
rm -f $BASE/python_code/utils/test_temp.py

rm -rf $BASE/python_code/detectors/deepsicsb
rm -rf $BASE/python_code/detectors/vsdnn

echo "Cleanup complete."

