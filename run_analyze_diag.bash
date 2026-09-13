#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
################################################################################################

#SBATCH --partition main                        ### all nodes - this is a plain LDPC-decode replay, no GPU needed
#SBATCH --time 0-01:00:00                       ### quick offline analysis, 1h is generous
#SBATCH --job-name analyze_diag                 ### name of the job
#SBATCH --output logs/analyze_diag-%j.out       ### output log - %j for this job's ID
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

################  Following lines will be executed by a compute node    #######################

### Hardcoded to the SNR=29 diag H5 the debug logging just produced - see analyze_diag_h5.py's
### docstring for why this doesn't need a config file (no training/EKF, just an LDPC decode
### replay against the H5's own arrays).
H5_FILE="/home/egero/Projects/Scratchpad/20260912_1704_TC_sp=0_QPSK_REs=96_UEs=1_ant=1_cfo=0.27_cfod=0.75_iqg=0_iqp=0_Clp=100_C=Lo_R=0.30_r=d2572a_frz=a_tf=0_dyn=ar1_a=0.99_sp0=0.1_sq=0.01_sr=0.5_spg=1_trk=ekf_lr=0.005_ps=20k_pltdcfo_s=123_SNR=29_diag.h5"

echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
echo "Analyzing: $H5_FILE"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/egero/miniconda3/envs/DeepOFDMs
echo "Using Python at: $(which python)"

cd /home/egero/Projects/DeepOFDM
python -m python_code.utils.analyze_diag_h5 "$H5_FILE"
