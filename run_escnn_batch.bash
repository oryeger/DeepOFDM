#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main                        ### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 7-00:00:00                       ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name deepsic                      ### name of the job
#SBATCH --array=0-31                 ### run parallel 5 times
#SBATCH --output logs/job-%J.out                        ### output log for running job - %J for job number

# Note: the following 4 lines are commented out
##SBATCH --gpus=rtx_3090:1                      ### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=rtx_3090:1
##SBATCH --mail-user=egero@post.bgu.ac.il       ### user's email for sending job status messages
##SBATCH --mail-type=ALL                        ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=8G                                ### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################
#my_values=(0.01 0.02 0.03 0.04 0.05)
#my_value=${my_values[$SLURM_ARRAY_TASK_ID]}
config_files=("Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0_s123_snr1.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0_s123_snr2.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p25_s123_snr1.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p25_s123_snr2.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p5_s123_snr1.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p5_s123_snr2.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg1_s123_snr1.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg1_s123_snr2.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0_s123_snr1.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0_s123_snr2.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p25_s123_snr1.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p25_s123_snr2.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p5_s123_snr1.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p5_s123_snr2.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg1_s123_snr1.yaml" "Mult_cfo0_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg1_s123_snr2.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0_s123_snr1.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0_s123_snr2.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p25_s123_snr1.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p25_s123_snr2.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p5_s123_snr1.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p5_s123_snr2.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg1_s123_snr1.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frnone_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg1_s123_snr2.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0_s123_snr1.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0_s123_snr2.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p25_s123_snr1.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p25_s123_snr2.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p5_s123_snr1.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg0p5_s123_snr2.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg1_s123_snr1.yaml" "Mult_cfo0.4_clip100_f0_NOAUG_CMC_sclow_k3_p10k_m4_u4_mpm1_mix0_ipm0_bs-1_ep100_dr0p0_wd0p0_lr5em3_frfirstconvonly_sh0_sap0_blf3_ov0_td0_nllgf_tlgf_bg1_s123_snr2.yaml")
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


#jupyter lab                                    ### this command executes jupyter lab – replace with your own command
#python -u DCD_MUSIC/main.py -s 0 -n 15 -ft near -mt SubspaceNet -to angle,range -ss 4096 -ttr 0 -m 3 -sn coherent -snap 100 -eta $my_value -w -t
cd /home/egero/Projects/DeepOFDM
if [[ "$1" == "--drift" ]]; then
  python -m python_code.run_drift --config config_files/$config_file
else
  python -m python_code.evaluate --config config_files/$config_file
fi