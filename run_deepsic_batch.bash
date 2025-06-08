#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 1-10:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name deepsic			### name of the job
#SBATCH --array=0-103                 ### run parallel 5 times
#SBATCH --output logs/job-%J.out			### output log for running job - %J for job number

# Note: the following 4 lines are commented out
##SBATCH --gpus=rtx_3090:1			### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=rtx_3090:1
##SBATCH --mail-user=egero@post.bgu.ac.il	### user's email for sending job status messages
##SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
##SBATCH --mem=24G				### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################
#my_values=(0.01 0.02 0.03 0.04 0.05)
#my_value=${my_values[$SLURM_ARRAY_TASK_ID]}
config_files=("TDL_A_seed41_5.yaml" "TDL_A_seed41_6.yaml" "TDL_A_seed41_7.yaml" "TDL_A_seed41_8.yaml" "TDL_A_seed41_9.yaml" "TDL_A_seed41_10.yaml" "TDL_A_seed41_11.yaml" "TDL_A_seed41_12.yaml" "TDL_A_seed41_13.yaml" "TDL_A_seed41_14.yaml" "TDL_A_seed41_15.yaml" "TDL_A_seed41_16.yaml" "TDL_A_seed41_17.yaml" "TDL_A_seed41_18.yaml" "TDL_A_seed41_19.yaml" "TDL_A_seed41_20.yaml" "TDL_A_seed41_21.yaml" "TDL_A_seed41_22.yaml" "TDL_A_seed41_23.yaml" "TDL_A_seed41_24.yaml" "TDL_A_seed41_25.yaml" "TDL_A_seed41_26.yaml" "TDL_A_seed41_27.yaml" "TDL_A_seed41_28.yaml" "TDL_A_seed41_29.yaml" "TDL_A_seed41_30.yaml" "TDL_A_seed123_5.yaml" "TDL_A_seed123_6.yaml" "TDL_A_seed123_7.yaml" "TDL_A_seed123_8.yaml" "TDL_A_seed123_9.yaml" "TDL_A_seed123_10.yaml" "TDL_A_seed123_11.yaml" "TDL_A_seed123_12.yaml" "TDL_A_seed123_13.yaml" "TDL_A_seed123_14.yaml" "TDL_A_seed123_15.yaml" "TDL_A_seed123_16.yaml" "TDL_A_seed123_17.yaml" "TDL_A_seed123_18.yaml" "TDL_A_seed123_19.yaml" "TDL_A_seed123_20.yaml" "TDL_A_seed123_21.yaml" "TDL_A_seed123_22.yaml" "TDL_A_seed123_23.yaml" "TDL_A_seed123_24.yaml" "TDL_A_seed123_25.yaml" "TDL_A_seed123_26.yaml" "TDL_A_seed123_27.yaml" "TDL_A_seed123_28.yaml" "TDL_A_seed123_29.yaml" "TDL_A_seed123_30.yaml" "TDL_A_seed17_5.yaml" "TDL_A_seed17_6.yaml" "TDL_A_seed17_7.yaml" "TDL_A_seed17_8.yaml" "TDL_A_seed17_9.yaml" "TDL_A_seed17_10.yaml" "TDL_A_seed17_11.yaml" "TDL_A_seed17_12.yaml" "TDL_A_seed17_13.yaml" "TDL_A_seed17_14.yaml" "TDL_A_seed17_15.yaml" "TDL_A_seed17_16.yaml" "TDL_A_seed17_17.yaml" "TDL_A_seed17_18.yaml" "TDL_A_seed17_19.yaml" "TDL_A_seed17_20.yaml" "TDL_A_seed17_21.yaml" "TDL_A_seed17_22.yaml" "TDL_A_seed17_23.yaml" "TDL_A_seed17_24.yaml" "TDL_A_seed17_25.yaml" "TDL_A_seed17_26.yaml" "TDL_A_seed17_27.yaml" "TDL_A_seed17_28.yaml" "TDL_A_seed17_29.yaml" "TDL_A_seed17_30.yaml" "TDL_A_seed58_5.yaml" "TDL_A_seed58_6.yaml" "TDL_A_seed58_7.yaml" "TDL_A_seed58_8.yaml" "TDL_A_seed58_9.yaml" "TDL_A_seed58_10.yaml" "TDL_A_seed58_11.yaml" "TDL_A_seed58_12.yaml" "TDL_A_seed58_13.yaml" "TDL_A_seed58_14.yaml" "TDL_A_seed58_15.yaml" "TDL_A_seed58_16.yaml" "TDL_A_seed58_17.yaml" "TDL_A_seed58_18.yaml" "TDL_A_seed58_19.yaml" "TDL_A_seed58_20.yaml" "TDL_A_seed58_21.yaml" "TDL_A_seed58_22.yaml" "TDL_A_seed58_23.yaml" "TDL_A_seed58_24.yaml" "TDL_A_seed58_25.yaml" "TDL_A_seed58_26.yaml" "TDL_A_seed58_27.yaml" "TDL_A_seed58_28.yaml" "TDL_A_seed58_29.yaml" "TDL_A_seed58_30.yaml")
config_file=${config_files[$SLURM_ARRAY_TASK_ID]}


### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
echo -e "Running job with value parameter: " $my_value "\n\n" 


### Start your code below ####
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/egero/miniconda3/envs/DeepOFDMs
echo "Using Python at: $(which python)"
python -c "import sys; print('Python version:', sys.version)"


#jupyter lab					### this command executes jupyter lab – replace with your own command
#python -u DCD_MUSIC/main.py -s 0 -n 15 -ft near -mt SubspaceNet -to angle,range -ss 4096 -ttr 0 -m 3 -sn coherent -snap 100 -eta $my_value -w -t
cd /home/egero/Projects/DeepOFDM
python -m python_code.evaluate --config config_files/$config_file



