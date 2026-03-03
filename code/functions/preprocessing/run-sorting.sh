#!/bin/bash

#SBATCH --job-name=kilosort
#SBATCH --partition=GPUshort
#SBATCH --gpus=1
#SBATCH --mem=16000
#SBATCH --cpus-per-gpu=4
#SBATCH --nodelist=esi-svhpc107
#SBATCH --mail-user=muad.abd-el-hay@esi-frankfurt.de,robert.taylor@esi-frankfurt.de
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/mnt/hpc_slurm/departmentN5/logs/spike-sorting/%j.out

source /mnt/hpc_slurm/opt/env/python/x86_64/conda/etc/profile.d/conda.sh

conda activate /mnt/hpc_slurm/departmentN5/conda_envs/spikeinterface

export PATH=$PATH:/mnt/hpc_slurm/opt/matlab-2021b/bin/

srun python ../preprocessing/sort-recording.py $1

exit 0
