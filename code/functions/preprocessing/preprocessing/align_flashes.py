import os
import sys
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import pathlib as p
import shutil
import time

import quickspikes as qs

from scipy.ndimage import gaussian_filter

from preprocessing.align_ephys import find_logfiles, snippet_eventmarkers, find_all_alignments, find_oe_recording
from preprocessing.preprocessing_utilities import get_session_info, create_standard_container, change_permissions_recursively, is_compute_storage, is_video_file, is_archive_storage, check_recalls

from parse_logfile import TextLog



def calculate_differences(video_path, pixel_jump, ratio):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Error: Could not open video {video_path}.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    success, previous_frame = cap.read()

    if success:
        previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)[::pixel_jump, ::pixel_jump]
    else:
        raise IOError("Error: Could not read the first frame.")

    diff_list = []
    for _ in tqdm(range(1, total_frames), desc="Detecting significant differences in video"):
        success, current_frame = cap.read()
        if not success:
            break
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)[::pixel_jump, ::pixel_jump]
        differences = (current_frame > ratio * previous_frame).astype(int)
        mean_difference = np.mean(differences)
        diff_list.append(mean_difference)
        previous_frame = current_frame

    cap.release()

    return diff_list

def detect_flashes(diff_list,  smoothing_sigma):
    """Detect flashes based on the processed frame differences."""
    smoothed_baseline = gaussian_filter(diff_list, sigma=smoothing_sigma)
    corrected_diffs = np.array(diff_list) - smoothed_baseline
    reldet = qs.detector(2.5, 2)
    reldet.scale_thresh(corrected_diffs.mean(), corrected_diffs.std())
    flash_times = reldet.send(corrected_diffs)
    return flash_times, corrected_diffs


def save_plot(diff_list, corrected_diffs, flash_times, video_path):
    """Save the plot showing the detected flashes."""
    plot_path = os.path.join(os.path.dirname(video_path), os.path.splitext(os.path.basename(video_path))[0] + "-flashplot.png")
    fig = plt.figure()
    plt.plot(diff_list, label='Differences', color="black", linewidth=0.1)  # Thinner line for differences
    plt.plot(corrected_diffs, label='Corrected Differences', color='blue', linewidth=0.1)  # Thinner line for corrected differences
    plt.scatter(flash_times, corrected_diffs[flash_times], color='red', label='Detected Flashes', marker='.', s=1)  # Smaller marker size for flashes
    plt.legend()
    plt.title('Detected Flashes')
    plt.xlabel('Frame number')
    plt.ylabel('Difference Value')
    plt.savefig(plot_path)
    plt.close(fig)


def extract_flashes(video_path, pixel_jump=5, ratio=1.5, smoothing_sigma=50):

    diff_list = calculate_differences(video_path, pixel_jump, ratio)
    
    flash_times, corrected_diffs = detect_flashes(diff_list,  smoothing_sigma)
    
    print("Detected ",len(flash_times)," flashes.")
    
    detected_flashes = np.zeros(len(diff_list))
    detected_flashes[flash_times] = 1

    save_plot(diff_list, corrected_diffs, flash_times, video_path)

    return detected_flashes, diff_list

def rle(inarray):
        """ run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.asarray(inarray)                # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])

def find_sessions(flash_runs, tolerance=1):
    """ Identify sessions allowing for gaps in the consecutive flashes. """
    sessions = []
    count = 1
    for i in range(1, len(flash_runs)):
        if flash_runs[i] - flash_runs[i - 1] <= 5 + tolerance:
            count += 1
        else:
            if count >= 4:  # Expect at least 4 close flashes to consider it a session start
                sessions.append(flash_runs[i - count])
            count = 1
    if count >= 4:  # Check the last sequence
        sessions.append(flash_runs[-count])
    return np.array(sessions)

def split_flashes(binary_flashes, raw_flashes):
    """ Split the detected flashes into sessions, allowing for occasional missed flashes. """
    _, positions, values = rle(binary_flashes)

    # Identify potential session starts
    flash_runs = positions[values == 1]
    session_beginnings = find_sessions(flash_runs)

    # Split the flash data based on identified session starts
    session_flashes = np.split(binary_flashes, session_beginnings)[1:]
    session_raw_flashes = np.split(raw_flashes, session_beginnings)[1:]

    # Calculate the number of trials in each session
    n_start_trials = [int(np.sum(sf)) for sf in session_flashes]

    print("Detected ", len(session_beginnings), " session beginning signals with the following number of flashes per session:", n_start_trials)

    return session_beginnings, session_flashes, session_raw_flashes, n_start_trials

def get_folders(video_path):
    compute_folder = os.path.dirname(video_path).replace("/as/", "/hpc_slurm/")
    archive_folder = os.path.dirname(video_path).replace("/hpc_slurm/", "/as/")
    return compute_folder, archive_folder

def get_alignments_and_events(compute_folder, archive_folder):


    files2recall = check_recalls(archive_folder, recall=True, output_root='/mnt/pns/recalls/')
            
    if len(files2recall) != 0:
        print('Files are being recalled, please wait until the process is finished and then try again')
        print(files2recall)
            
    while len(check_recalls(archive_folder, recall=False)) != 0:
        time.sleep(5)
    
    try:
        alignments = find_all_alignments(compute_folder)

    except ValueError:
        
        alignments = snippet_eventmarkers(archive_folder)

    logfiles = find_logfiles(archive_folder)

    eventfiles = glob.glob(compute_folder + "/*_events*.npy")
    
    reordered_logfiles = [i for _, i in sorted(zip(np.argsort(alignments), logfiles))]

    reordered_eventfiles = [i for _, i in sorted(zip(np.argsort(alignments), eventfiles))]

    return alignments, reordered_logfiles, reordered_eventfiles


def align_flashes(video_path):
    if not os.path.isfile(video_path):
        raise ValueError(video_path + " either does not exist or is not a file.")

    compute_folder, archive_folder = get_folders(video_path)

    print("Detecting flashes in video: " + os.path.basename(video_path))
    binary_flashes, raw_flashes = extract_flashes(video_path)

    print("Splitting flashes by session beginnings (5 flashes).")
    session_beginnings, session_flashes, _ , n_start_trials  = split_flashes(binary_flashes, raw_flashes)

    alignments, logfiles, eventfiles = get_alignments_and_events(compute_folder, archive_folder)
    
    # Process each session based on alignments and logfiles
    for i, alignment in enumerate(alignments):

        trial_frames, _ = process_session(session_flashes, session_beginnings, n_start_trials, logfiles[i], eventfiles[i])
        
        save_session_results(trial_frames, eventfiles[i], compute_folder, video_path, alignment)
        

        
        
        

def verify_flash_alignment(session_flash_indices, logfile_flash_indices, logfile_flash_times, frame_tolerance=50):
    """
    Verifies that each detected flash is within a certain number of frames from the expected flash.

    :param detected_indices: Array of indices where flashes were detected in the video.
    :param expected_indices: Array of expected flash indices based on the logfile.
    :param frame_tolerance: The maximum number of frames the detected index can differ from the expected.
    :return: A tuple containing the array of trial frames and the count of corrected flashes.
    """
    trial_frames = np.zeros(len(logfile_flash_indices))
    
    corrected_counter = 0

    for j, expected_idx in enumerate(logfile_flash_indices):
        # Find the closest detected flash index
        distances = np.abs(session_flash_indices - expected_idx)
        closest_index = np.argmin(distances)
        
        if distances[closest_index] <= frame_tolerance:
            # If the closest detected flash is within the tolerance, it's considered correct
            trial_frames[j] = session_flash_indices[closest_index]
        else:
            # If not, increment the corrected_counter and potentially handle the correction
            corrected_counter += 1
            trial_frames[j] = interpolate_missing_flash(trial_frames, expected_idx, j, logfile_flash_times)

    return trial_frames, corrected_counter

def interpolate_missing_flash(trial_frames, expected_idx, current_idx, logfile_flash_times, frame_rate=60):
    """
    Interpolates the index of a missing flash based on surrounding flashes and the expected index.
    
    :param detected_indices: Array of indices where flashes were detected in the video.
    :param expected_idx: The current expected flash index based on the logfile.
    :param current_idx: The current index in the expected flash sequence being checked.
    :param framerate: The framerate of the video to calculate the hypothetical frame indices.
    :return: The interpolated index for the missing flash.
    """
    # This is a placeholder for the actual interpolation logic
    # You might interpolate based on the surrounding correct flashes or return the expected index directly
    if current_idx > 0:
        # If not the first flash, interpolate based on the previous flash
        last_correct_flash = trial_frames[current_idx - 1]
        
        last_correct_logfile_time = logfile_flash_times[current_idx - 1]

        next_correct_logfile_time = logfile_flash_times[current_idx]

        hypothetical_frames = np.round((next_correct_logfile_time - last_correct_logfile_time) / (1/frame_rate)).astype(int)
        
        return last_correct_flash + hypothetical_frames
    else:
        # If it's the first flash, there's nothing to interpolate from; return the expected index
        return expected_idx

def process_session(session_flashes, session_beginnings, n_start_trials,  logfile, eventfile,  frame_rate = 60):

    
    # Load event markers and parse log file
    ev_markers = np.load(eventfile)
    with TextLog(logfile) as log:
        evt, _ , _ , true_ts = log.parse_eventmarkers()

    # Determine the number of expected flashes from the logfile
    n_flashes_logfile = np.sum(ev_markers[:, 3] == 3000)

    print(f"Logfile has {n_flashes_logfile}.")

    # Find out which part of the video (which split) the flashes are most probably from
    session_idx = np.argmin(
        abs(np.sum(ev_markers[:, 3] == 3000) - n_start_trials))
    
    # Calculate the logfile flash times and indices
    logfile_flash_times = true_ts[evt == 3000] - true_ts[evt > 30000][0]
    logfile_flash_indices = np.round((true_ts[evt == 3000] - true_ts[evt > 30000][0]) / (1/frame_rate)).astype(int)

    # Retrieve the indices of the flashes detected in the session
    session_flash_indices = np.nonzero(session_flashes[session_idx])[0]

    n_flashes_detected = len(session_flash_indices)

    if n_flashes_detected != n_flashes_logfile:

        missing_percentage = abs(n_flashes_detected - n_flashes_logfile) / n_flashes_logfile
        
        if missing_percentage > 0.2:
            
            raise ValueError(f"Session {session_idx}: More than 20% of the flashes are missing. Detected: {n_flashes_detected}, Expected: {n_flashes_logfile}")

        else:

            trial_frames, corrected_counter = verify_flash_alignment(session_flash_indices, logfile_flash_indices, logfile_flash_times)
    else:
        
        trial_frames, corrected_counter = verify_flash_alignment(session_flash_indices, logfile_flash_indices, logfile_flash_times)

    trial_frames = trial_frames.astype(int)

    trial_frames_corrected = trial_frames + session_beginnings[session_idx]
        
    return trial_frames_corrected, corrected_counter





def save_session_results(trial_frames_corrected, eventfile, compute_folder, video_path, alignment):

    ## get the event file name
    event_filename = os.path.splitext(eventfile)[0].split("/")[-1]

    _ , project, subject, date, experiment, session = get_session_info(compute_folder)

    containername = create_standard_container(project, subject, date, experiment, session, extension=".dlc")

    combinedpath = os.path.join(compute_folder, containername)
    if not os.path.isdir(combinedpath):
        os.mkdir(combinedpath)

    videoname = os.path.splitext(video_path)[0].split("/")[-1]
    

    filename = "_".join(event_filename.split("_")[:-1])+"-"+videoname+"_snip"+alignment+".flashframes"

    print(f"Saving: {filename}")

    np.save(file = combinedpath+"/"+filename, arr = trial_frames_corrected)

    change_permissions_recursively(combinedpath+"/"+filename)



if __name__ == '__main__':
    
#    if len(sys.argv) != 1:
#        raise ValueError("No or too many arguments given.")

    if is_video_file(sys.argv[1]):
        video_path = sys.argv[1]
    else:
        raise ValueError("Not a video path.")
    
    if is_archive_storage(video_path):
        
        putPath = p.Path(video_path.replace("/mnt/as/","/mnt/hpc_slurm/"))
        
        if not putPath.exists() or putPath.stat().st_size == 0:

            files2recall = check_recalls(video_path, recall=True, output_root='/mnt/pns/recalls/')
            
            if len(files2recall) != 0:
                print('Files are being recalled, please wait until the process is finished and then try again')
                print(files2recall)
            
            while len(check_recalls(video_path, recall=False)) != 0:
                time.sleep(5)
            
        print("Copying video file to compute storage")
        
        cs_video_path = shutil.copyfile(video_path, putPath)

        change_permissions_recursively(cs_video_path)
        
    elif is_compute_storage(video_path):

        cs_video_path = video_path
            
    else:
        raise ValueError("Video is neither on archive nor on compute storage...")
        

    align_flashes(str(cs_video_path))

