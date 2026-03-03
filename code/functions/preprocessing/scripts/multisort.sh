#!/bin/bash

set +o posix

# Define the base directory
base_dir="/mnt/as/projects/OWzeronoise/"

# Find folders meeting the specified criteria
mapfile -d $'\0' array < <(find "$base_dir" -maxdepth 5 -mindepth 5 -cmin +30 -cmin -1440 -print0 | while IFS= read -r -d '' file; do if [ $(du -sh -BM "$file" | cut -f1 | tr -d 'M') -ge 1000 ]; then echo -ne "$file\0"; fi; done)

# Check if the array is empty
if [ ${#array[@]} -eq 0 ]; then
    echo "No folders found. Exiting."
    exit 0
fi

# Check if all folders have a .spy folder
all_have_spy=true
for folder in "${array[@]}"; do
    if [ ! -d "$folder/.spy" ]; then
        all_have_spy=false
        break
    fi
done

# Proceed if all folders have .spy directories
if [ "$all_have_spy" = true ]; then
    echo "All folders have .spy directories. Proceeding with processing."

    sbatch ./scripts/multisort-sbatch.sh "${array[@]}"
    
else
    echo "Not all folders have .spy directories. Exiting."
    exit 1
fi
