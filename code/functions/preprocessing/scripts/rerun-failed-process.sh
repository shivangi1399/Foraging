#!/bin/bash

set +o posix

if test $(find /mnt/as/projects/MWzeronoise/ /mnt/as/projects/OWzeronoise/ -maxdepth 5 -mindepth 5 -cmin +30 -cmin -1440 -print0 | while IFS= read -r -d '' file; do if [ $(du -sh -BM "$file" | cut -f1 | tr -d 'M') -ge 1000 ]; then echo -ne "$file\0"; fi; done | wc -c) -eq 0
then
    echo "Nothing to do... NEXT!"
    exit
fi


mapfile -d $'\0' array < <(find /mnt/as/projects/MWzeronoise/ /mnt/as/projects/OWzeronoise/ -maxdepth 5 -mindepth 5 -cmin +30 -cmin -1440 -print0 | while IFS= read -r -d '' file; do if [ $(du -sh -BM "$file" | cut -f1 | tr -d 'M') -ge 1000 ]; then echo -ne "$file\0"; fi; done)


for i in "${array[@]}"
do
    :
    compfold=${i/\/as\//\/hpc\/}
    if [ -d $compfold ]; then	
	echo "checking for already ran processing in: $compfold"
	checkdlc=$(find $compfold -maxdepth 1 -regex '.*\.\(dlc\)')
	checkephys=$(find $compfold -maxdepth 1 -regex '.*\.\(nwb\)')
	if [[ -d $checkdlc && -d $checkephys ]]; then
	    echo "running both-erino"
	    bash /mnt/hpc_slurm/departmentN5/code/preprocessing/preprocess_session.sh $i
	elif [[ -d $checkdlc || -d $checkephys ]]; then
	    if [[ -d $checkephys ]]; then
		echo "running ephys only"
		bash /mnt/hpc_slurm/departmentN5/code/preprocessing/preprocess_session.sh -e $i
	    elif [[ -d $checkdlc ]]; then
		echo "running dlc only"
		bash /mnt/hpc_slurm/departmentN5/code/preprocessing/preprocess_session.sh -v $i
	    fi
	else
	    checkspy=$(find $compfold -maxdepth 1 -type d -regex '.*\.\(spy\)')

	    if ! [[ -d $checkspy ]]; then
		
		spikes=$(find $spyfolder -maxdepth 1 -name '*spikes.spike')
		lfps=$(find $spyfolder -maxdepth 1 -name '*lfp.analog')
		emuas=$(find $spyfolder -maxdepth 1 -name '*eMUA.analog')

		if [[ -e $spikes || -e $lfps || -e $emuas ]]; then
		    echo "Rerunning ephys"
		    bash /mnt/hpc_slurm/departmentN5/code/preprocessing/preprocess_session.sh -e $i
		fi
	    fi
	    echo "already analyzed"
	fi
    else
	echo "walking everything"
	bash /mnt/hpc_slurm/departmentN5/code/preprocessing/preprocess_session.sh $i
    fi
done
