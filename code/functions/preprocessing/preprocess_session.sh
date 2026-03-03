#!/bin/bash

# Ensure script fails on any error
set -uo pipefail

# Cleanup function definition
cleanup() {
    [[ -n "${tmppipe:-}" && -e "$tmppipe" ]] && rm -f "$tmppipe"
}

# Register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

source /mnt/hpc_slurm/opt/env/python/x86_64/conda/etc/profile.d/conda.sh
# Only works on Linux (or systems that have rwadlink with the -f option)

# Initialize processing flags to true for default behavior
run_behaviour=false
run_ephys=false
run_videos=false
# Default behavior for sending emails is true, can be explicitly disabled with -m
send_emails=true


# Initialize a variable to keep track of process outcomes
process_outcomes=""

main() {
    
    # Parse and handle options
    while getopts "bevm" option; do
        case $option in
            b)
                run_behaviour=true
                ;;
            e)
                run_ephys=true
                ;;
            v)
                run_videos=true
                ;;
            m)
                send_emails=false
                ;;
            *)
                echo "Incorrect options provided" >&2
                exit 1
                ;;
        esac
    done

    # If no specific options are provided, run all tasks
    if [ $OPTIND -eq 1 ]; then
	echo "No specific options provided. Running everything"
        run_behaviour=true
        run_ephys=true
        run_videos=true
    fi


    shift $((OPTIND - 1))
    
    if [ $# -eq 0 ]
    then
	echo "No folder supplied"
	tmppipe=$(mktemp)
	python find_as_folder.py $tmppipe

	fold=$( cat $tmppipe )

	validate_folder_structure $fold
	
	rm $tmppipe
    else
	echo $1
	# Attempt to normalize and validate the path
	validate_folder_structure $1
    fi

    echo $normalized_path

    # Setup logging
    setup_logging "$normalized_path"

    # run session alignment
    align_sessions "$normalized_path"
    
    # Process based on options
    [[ $run_videos == true ]] && process_videos $normalized_path
    [[ $run_ephys == true ]] && process_ephys $normalized_path
    [[ $run_behaviour == true ]] && process_behaviour $normalized_path

    compfold=${normalized_path/\/as\//\/hpc\/}

    if [[ -n "$compfold" ]]; then
        change_permissions "$compfold"
    else
        printf "compfold is not defined or is empty. Provide a directory to change permissions.\n" >&2
    fi
    
    echo "Waiting for things to finish up..."
    sleep 30
    
    # Send email if required
    [[ $send_emails == true ]] && send_email


}

setup_logging() {
    local fold=$1
    local prefix=$(echo "${fold//\//-}" | grep -Po 'zeronoise-\K.*')
    local currtime=$(date +"%Y%m%d-%H%M%S")
    outlog="/mnt/hpc_slurm/departmentN5/logs/preprocessing/${prefix}-${currtime}.log"
    outerror="/mnt/hpc_slurm/departmentN5/logs/preprocessing/${prefix}-${currtime}-errors.log"
    
    exec 1> >(tee -a "$outlog")
    exec 2> >(tee -a "$outerror")

    echo "Setup loggging"
}


change_permissions() {
    local directory=$1
    local required_permissions="2770"

    if [[ -n "$directory" ]]; then
        local current_permissions
        current_permissions=$(stat -c "%a" "$directory")

        if [[ "$current_permissions" == "$required_permissions" ]]; then
            printf "Permissions for '%s' are already set to %s.\n" "$directory" "$required_permissions"
        else
            printf "Changing permissions for '%s' and its contents to %s...\n" "$directory" "$required_permissions"
            chmod -R "$required_permissions" "$directory"
            
            if [[ $? -eq 0 ]]; then
                printf "Permissions changed successfully.\n"
            else
                printf "Failed to change permissions.\n" >&2
                return 2
            fi
        fi
    else
        printf "Directory variable is not defined or is empty. Skipping permissions change.\n" >&2
        return 1
    fi
}


normalize_path() {
    local input_path=$1
    local normalized_path

    # Ensure the path starts with a leading slash
    if [[ $input_path != /* ]]; then
        normalized_path="/$input_path"
    else
        normalized_path="$input_path"
    fi

    # If the path starts with /as/, prepend /mnt to make it /mnt/as/
    if [[ $normalized_path == /as/* ]]; then
        normalized_path="/mnt$normalized_path"
    fi

    # If validation succeeds, echo the normalized path
    echo "$normalized_path"
}

validate_folder_structure() {
    local path=$1
    local -a parts
    local -a normalized_parts
    
    IFS='/' read -ra parts <<< "$path"

    ## For debugging
    ##echo "${parts[@]}"
    ##echo "${#parts[@]}"

    if [[ -d "$path" ]]; then
	echo "The path is a directory and it exists."
    else
	echo "The path is either not a directory or it does not exist."
	return 1
    fi

    # Corrected check for multiple starting paths
    if ! { [[ ${parts[1]} == "mnt" ]] && [[ ${parts[2]} =~ ^(as)$ ]] && [[ ${parts[3]} == "projects" ]]; } &&
         ! { [[ ${parts[1]} =~ ^(as)$ ]] && [[ ${parts[2]} == "projects" ]]; }; then
        echo "Invalid path: Should start with /mnt/ or /as/" >&2
        return 1
    fi

    normalized_path=$(normalize_path "$path")

    IFS='/' read -ra normalized_parts <<< "$normalized_path"

    ##echo "${normalized_parts[@]}"

    if ! [[ ${#normalized_parts[@]} -eq 10 ]]; then
	echo "The path is too short, it needs to point to the individual recording session"
	return 1
    fi
    
    # Check for the additional directory level after session type
    if ! [[ ${normalized_parts[-1]} =~ ^[0-9]+$ ]]; then
        echo "Invalid path: Expected session number such as 010 but instead got ${parts[-2]}" >&2
        return 1
    fi


    # Check species
    if ! [[ ${normalized_parts[-6]} =~ ^(OWzeronoise|MWzeronoise|ZeroNoise_MOUSE|ZeroNoise_MONKEY)$ ]]; then
        printf "Invalid path: Species must be 'OWzeronoise', 'MWzeronoise', 'ZeroNoise_MOUSE', or 'ZeroNoise_MONKEY'\n" >&2
        return 1
    fi

    # Validate the date format and check if it's a real date and not in the future
    validate_date_format "${normalized_parts[-3]}"


}

validate_date_format() {
    local date_part=$1
    
    if ! [[ $date_part =~ ^[0-9]{8}$ ]]; then
        echo "Invalid path: Date part must be 8 consecutive numbers" >&2
        return 1
    fi

    # Convert date_part to YYYY-MM-DD format for date command compatibility
    local formatted_date="${date_part:0:4}-${date_part:4:2}-${date_part:6:2}"

    # Check if it's a real date
    if ! date -d "$formatted_date" "+%Y%m%d" &>/dev/null; then
        echo "Invalid path: Not a real date" >&2
        return 1
    fi

    # Check if the date is not in the future
    local current_date=$(date "+%Y%m%d")
    if [[ $date_part -gt $current_date ]]; then
        echo "Invalid path: Date cannot be in the future" >&2
        return 1
    fi

    return 0
}

run_conda_command() {
    local environment_path="$1"
    shift
    conda run --verbose --no-capture-output -p "$environment_path" "$@"
}

submit_job_and_get_id() {
    local job_script=$1
    shift
    local sbatch_output=$(sbatch "$job_script" "$@")
    if [[ "$sbatch_output" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
	return 0
    else
        echo "Failed to submit job: $job_script" >&2
        return 1
    fi
}


align_sessions() {
    local fold=$1
    echo "Aligning sessions in $fold"
    
    if run_conda_command "/mnt/hpc_slurm/departmentN5/conda_envs/oesyncopy/" python session_alignments.py "$1"; then
        process_outcomes+="Alignment SUCCESS\n"
    else
        process_outcomes+="Alignment FAILED\n"
    fi
}


process_videos() {
    local fold=$1
    echo "Processing videos in $fold"
    local video
    local job_id
    for video in "$fold"/*.avi; do
        if [[ $video == *"cam35"* ]]; then
            if grep -q -e "/OWzeronoise/" -e "/ZeroNoise_MOUSE/" <<< "$fold"; then
                echo "Running mouse face and pupil analysis."
                job_id=$(submit_job_and_get_id ./scripts/run-dlc-mouse-face.sh "$video") && \
                    process_outcomes+="Mouse face analysis job ID: $job_id submitted successfully.\n" || \
			process_outcomes+="Failed to submit mouse face analysis job for video $video.\n"
                
                echo "Running mouse pupil analysis."
                job_id=$(submit_job_and_get_id --dependency=afterok:$job_id ./scripts/run-dlc-mouse-pupil.sh "$video") && \
                    process_outcomes+="Mouse pupil analysis job ID: $job_id submitted successfully.\n" || \
			process_outcomes+="Failed to submit mouse pupil analysis job for video $video.\n"

		echo "Running flash extraction:"
		job_id=$(submit_job_and_get_id --dependency=afterany:$job_id ./scripts/run-flash-extraction.sh "$video") && \
                    process_outcomes+="Flash extraction job ID: $job_id submitted successfully.\n" || \
			process_outcomes+="Failed to submit flash extraction job for video $video.\n"
		
            elif grep -q -e "/MWzeronoise/" -e "/ZeroNoise_MONKEY/" <<< "$fold"; then
                echo "Running monkey face analysis."
                job_id=$(submit_job_and_get_id ./scripts/run-dlc-monkey-face.sh "$video") && \
                    process_outcomes+="Monkey face analysis job ID: $job_id submitted successfully.\n" || \
			process_outcomes+="Failed to submit monkey face analysis job for video $video.\n"
		echo "Running flash extraction:"
		job_id=$(submit_job_and_get_id --dependency=afterany:$job_id ./scripts/run-flash-extraction.sh "$video") && \
                    process_outcomes+="Flash extraction job ID: $job_id submitted successfully.\n" || \
			process_outcomes+="Failed to submit flash extraction job for video $video.\n"
            fi
        elif [[ $video == *"cam34"* ]] && ! grep -q -e "/OWzeronoise/" -e "/ZeroNoise_MOUSE/" <<< "$fold"; then
            echo "Running monkey hand analysis."
            job_id=$(submit_job_and_get_id ./scripts/run-dlc-monkey-hands.sh "$video") && \
		process_outcomes+="Monkey hand analysis job ID: $job_id submitted successfully.\n" || \
		    process_outcomes=+="Failed to submit monkey hand analysis job for video $video.\n"
	    echo "Running flash extraction:"
	    job_id=$(submit_job_and_get_id --dependency=afterany:$job_id ./scripts/run-flash-extraction.sh "$video") && \
                process_outcomes+="Flash extraction job ID: $job_id submitted successfully.\n" || \
		    process_outcomes+="Failed to submit flash extraction job for video $video.\n"
        fi
    done
}


process_ephys() {
    local fold=$1
    echo "Running ephys analysis on $fold"
    
    if run_conda_command "/mnt/hpc_slurm/departmentN5/conda_envs/oesyncopy/" python ephys.py "$fold"; then
            process_outcomes+="EPhys SUCCESS\n"
    else
        process_outcomes+="Ephys FAILED\n"
    fi
}



process_behaviour() {
    local fold=$1
    echo "Running behaviour analysis on $fold"
    local log
    for log in "$fold"/*_Cont.log; do
        if [[ $log != *"Start_Cont.log" ]]; then
            # Behaviour analysis
            if run_conda_command "/mnt/hpc_slurm/departmentN5/conda_envs/oesyncopy/" python -m EndOfDayFinal "$log"; then
                process_outcomes+="Behaviour analysis for log $log SUCCESS\n"
            else
                process_outcomes+="Behaviour analysis for log $log FAILED\n"
            fi
            
            echo "Extracting metadata and detecting approximate flashes"
            # Metadata extraction
            if run_conda_command "/mnt/hpc_slurm/departmentN5/conda_envs/oesyncv2/" python create-metadata.py "$fold"; then
                process_outcomes+="Metadata extraction for $fold SUCCESS\n"
            else
                process_outcomes+="Metadata extraction for $fold FAILED\n"
            fi
        fi
    done
}


send_email() {
    echo "Sending email notification"
    local compfold=${normalized_path/\/as\//\/hpc\/}
    
    # Determine the recipients based on the folder content
    if grep -q -e "/OWzeronoise/" -e "/ZeroNoise_MOUSE/" <<< "$normalized_path"; then
        recip="muad.abd-el-hay@esi-frankfurt.de robert.taylor@esi-frankfurt.de"
    else
        recip="katharine.shapcott@esi-frankfurt.de shivangi.patel@esi-frankfurt.de"
    fi

    local email_content="Processing Summary for $normalized_path\n\n"

    # Directly include the dynamically populated process_outcomes
    email_content+="Outcome of Processing Steps:\n$process_outcomes\n"

    
    if [[ $run_ephys == true ]]; then
        email_content+="Ephys Processing Outcomes:\n"
        local alignmentfile=$(find "$compfold" -maxdepth 1 -name '*events*.npy')
        local nwbfile=$(find "$compfold" -maxdepth 1 -name '*.nwb')
        local spyfolder=$(find "$compfold" -maxdepth 1 -type d -name '*.spy')
        
        ##email_content+=$( [[ -n "$alignmentfile" ]] && echo "Alignment SUCCESS\n" || echo "Alignment FAILED\n" )
        email_content+=$( [[ -n "$nwbfile" ]] && echo "NWB-creation SUCCESS\n" || echo "NWB-creation FAILED\n" )
        
        if [[ -n "$spyfolder" ]]; then
            local ephys_summary="Ephys data found. Checking details...\n"
            local spikes=$(find "$spyfolder" -maxdepth 1 -name '*spikes.spike')
            local lfps=$(find "$spyfolder" -maxdepth 1 -name '*lfp.analog')
            local emuas=$(find "$spyfolder" -maxdepth 1 -name '*eMUA.analog')
            
            ephys_summary+=$( [[ -n "$spikes" ]] && echo "Spikes data present.\n" || echo "Spikes data MISSING.\n" )
            ephys_summary+=$( [[ -n "$lfps" ]] && echo "LFPs data present.\n" || echo "LFPs data MISSING.\n" )
            ephys_summary+=$( [[ -n "$emuas" ]] && echo "eMUA data present.\n" || echo "eMUA data MISSING.\n" )
            email_content+="$ephys_summary"
        else
            email_content+="NO Ephys data found.\n"
        fi
    fi

    # Send the email
    echo -e "$email_content" | /usr/bin/mail -s "Preprocessing summary for $normalized_path" "$recip"
}


# Ensure the script directory is the working directory
cd "$(dirname -- "$(readlink -f -- "$0")")"

# Call main with all the arguments
main "$@"
