#!/bin/bash

#SBATCH --job-name=dlc-mouse-pupil
#SBATCH --partition=GPUshort
#SBATCH --gpus=1
#SBATCH --mem=16000
#SBATCH --cpus-per-gpu=4
#SBATCH --mail-user=muad.abd-el-hay@esi-frankfurt.de
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/mnt/hpc_slurm/departmentN5/logs/dlc-mouse-pupil/%j.out

## Check if the path given is valid

echo "Running script on following input: $1"

validate_folder_structure() {
    local path=$1

    # Split the path into an array using '/' as delimiter
    IFS='/' read -ra PARTS <<< "$path"

    # Debug: print out the parts of the path
    #echo "Debug: Entire path array: ${PARTS[@]}"
    ##echo "Debug: Species part: ${PARTS[-6]}"
    ##echo "Debug: Date part: ${PARTS[-3]}"

    # Check for multiple starting paths
    if ! { [[ ${PARTS[1]} == "mnt" ]] && [[ ${PARTS[2]} =~ ^(as|cs|hpc|hpc_slurm)$ ]] && [[ ${PARTS[3]} == "projects" ]]; } &&
         ! { [[ ${PARTS[1]} =~ ^(as|cs|hpc|hpc_slurm)$ ]] && [[ ${PARTS[2]} == "projects" ]]; }; then
        echo "Invalid path: Does not start with a valid prefix"
        return 1
    fi

# Check species (either 'OWzeronoise', 'MWzeronoise', 'ZeroNoise_MOUSE', or 'ZeroNoise_MONKEY')
    if [[ ${PARTS[-6]} != "OWzeronoise" && ${PARTS[-6]} != "MWzeronoise" && ${PARTS[-6]} != "ZeroNoise_MOUSE" && ${PARTS[-6]} != "ZeroNoise_MONKEY" ]]; then
    echo "Invalid path: Species must be 'OWzeronoise', 'MWzeronoise', 'ZeroNoise_MOUSE', or 'ZeroNoise_MONKEY'"
    return 1
    fi


    # Check if date is in the correct format (YYYYMMDD)
    if ! [[ ${PARTS[-3]} =~ ^[0-9]{4}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])$ ]]; then
        echo "Invalid path: Date must be in YYYYMMDD format"
        return 1
    fi

    # If all checks pass
    echo "Valid folder structure"
    return 0
}

directory_path=$(dirname "$1")

validate_folder_structure "$directory_path"
validation_result=$?

# Check if the validation failed
if [ $validation_result -ne 0 ]; then
    echo "Folder structure validation failed. Exiting script."
    exit 1
fi


# Determine the node on which the job is running
current_node=$SLURMD_NODENAME

echo $current_node


# Corrected conditional code
if [ "$current_node" == "esi-svhpc107" ]; then
    echo "Running on node esi-svhpc107."
    source /mnt/hpc_slurm/opt/env/python/x86_64/conda/bin/activate
    conda activate /mnt/hpc_slurm/departmentN5/conda_envs/dlc_rtx6000

elif [[ "$current_node" == "esi-sv922-01" || "$current_node" == "esi-sv922-02" ]]; then
    echo "Running on POWER node."
    source /mnt/hpc_slurm/opt/env/python/power8/conda/bin/activate
    conda activate /mnt/hpc_slurm/departmentN5/conda_envs/dlc_v100

else
    echo "Running on an unexpected node. Exiting."
    exit 1
fi

srun python ./scripts/run-mouse-pupil.py $1

exit 0
