#!/bin/bash

set +o posix

mapfile -d $'\0' array < <(find /mnt/as/projects/MWzeronoise/ /mnt/as/projects/OWzeronoise/ -maxdepth 5 -mindepth 5 -cmin +30 -cmin -180 -print0)

non_migrated_array=() # This will store the non-migrated folders

for i in "${array[@]}"
do
    fls=$(ls -ls "$i")
    if [[ ! -z $fls ]]; then
       fblk=$(echo $fls | awk '{print $1}')
       fsize=$(echo $fls | awk '{print $6}')
       if (( $fblk != 0 || $fsize == 0 )); then
          non_migrated_array+=("$i") # The folder is non-migrated
       fi
    fi
done

if [ ${#non_migrated_array[@]} -eq 0 ]; then
    echo "Nothing to do... NEXT!"
    exit
fi

for i in "${non_migrated_array[@]}"
do
    compfold=${i/\/as\//\/hpc\/}
    if [ -d $compfold ]; then	
	    echo "Already ran"
    else
	    recordfind=$(find $i -type d -name "Record*")
	    if ! [[ -d $recordfind ]]; then
	        echo "walking behavior only"
	        bash /mnt/hpc_slurm/departmentN5/code/preprocessing/preprocess_session.sh -b $i
	    else
	        echo "walking everything"
	        bash /mnt/hpc_slurm/departmentN5/code/preprocessing/preprocess_session.sh $i
	    fi
    fi
done
