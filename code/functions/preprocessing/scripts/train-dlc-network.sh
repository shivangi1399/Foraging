#!/bin/bash

#SBATCH --job-name=dlc-train-network
#SBATCH --partition=GPUshort,GPUlong
#SBATCH --gpus=1
#SBATCH --mem=16000
#SBATCH --cpus-per-gpu=4
#SBATCH --nodelist=esi-svhpc107
#SBATCH --mail-user=muad.abd-el-hay@esi-frankfurt.de
#SBATCH --mail-type=END,FAIL

source /mnt/hpc_slurm/opt/env/python/x86_64/conda/etc/profile.d/conda.sh

conda activate /mnt/hpc_slurm/departmentN5/conda_envs/dlc_rtx6000

srun python /mnt/hpc_slurm/home/abdelhaym/train-dlc-network.py $1

exit 0
