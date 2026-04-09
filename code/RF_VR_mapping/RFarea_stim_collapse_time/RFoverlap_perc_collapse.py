
# This script check what percentages of RF centers fall inside the stimulus outlines accountinf for collapse
# and does it with a specific time window

"""
#### Code instructions #######################################################################################################
# 1. run mapping_corner.py for stimuli corner in cartesian coordinates
# 2. run prepare_RF.py for Rfs in cartseian coordinates
# 3. order the corner using corner_order.py and run warping_stim.py to warp the stimuli in the corners we got in step 1
# 4. Check if RF falls inside the warped stim accounting for collapse using RFarea_stim_collapse.py
# 5. Check how everything looks using RFoverlap_perc_collapse.py
##############################################################################################################################
"""

from collections import defaultdict
from typing import Optional, List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from skimage.transform import ProjectiveTransform
from skimage import io
from scipy import interpolate                       
import h5py
import csv
import pickle
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog

# aligned_stim_times ---------------------------------------------------------------------------------------------
folder = '//cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data/20230214/'
filename = folder + '2023_02_14-11_42_27_Cosmos_LeafForaging_001_PAF_GrassyLandscapeWithBackgroundDark_Cont.log'

stim_loc = []
stim_ts = []
stim_params = []

stim_width = np.zeros(0, dtype = np.uint16)
vertical_offset = np.zeros(0, dtype = np.uint16)

stim_name = 'ImageStimulus'

n_stimuli = 5 # Number of stimuli per trial (e.g. dividing placeholder, etc.)

with TextLog(filename) as log:
    start_trial_times = log.parse_all_state_times(state='StartTrial', times='StateStarted')
    end_trial_times = log.parse_all_state_times(state='EndTrial', times='StateStarted')
    evt, ts, evt_desc, true_ts = log.parse_eventmarkers()
    
    indx = [ii for ii, name in enumerate(log.all_ids['name']) if name.startswith(stim_name)] #finding all image stimulus and their IDs
    for ii, istim in enumerate(indx):
        ii
        istim
        if ii + n_stimuli == len(indx):
            break
        this_id = log.all_ids[istim]
        next_id = log.all_ids[indx[ii + n_stimuli]] # this should be the spawn time of the image in the next trial
        loc, pos_ts = log.parse_spherical(obj_id=this_id['id'],
                                              st=this_id['start'],
                                              end=next_id['start']) #location and time
        params = log.parse_initial_parameters(obj_id = this_id['id'], st = this_id['start'], end = next_id['start']) #stim parameters
        
        stim_loc.append(loc) #loc for each stim with time
        stim_ts.append(pos_ts) #corresponding time for each stim when it was on
        stim_params.append(params)

#Align the timepoints with just the time points when the stimulus is on
trial_onset = ts[np.where(evt == 3000)[0]] #time of trial start
target_onset = ts[np.where(evt == 3011)[0]] # time at which the stimulus comes on for the first time
trial_end = ts[np.where(evt == 3090)[0]]  
n_trials = trial_onset.shape[0] - 1 

aligned_stim_times_list = [] # times at which the stimulus is on the screen

for itrl in range(n_trials): 
    
    stim_times = stim_ts[itrl * n_stimuli + 3].T
    aligned_stim_times = stim_times - stim_times[0]  
    aligned_stim_times_list.append(aligned_stim_times)


# RF in stim percent plots 

output_hdf5_file =  '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/RFarea_stim/20230214/RF_stim_collapse.h5'
save_path = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/RF VR mapping/RFarea_VR_collapse(entire trial).pdf'
aligned_stim_times = aligned_stim_times_list

def process_file(file_path: str, aligned_stim_times: List[np.ndarray], max_trials: Optional[int] = None, max_time_points: Optional[int] = None) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    total_counts = defaultdict(int)
    inside_a_counts = defaultdict(int)
    inside_b_counts = defaultdict(int)
        
    with h5py.File(file_path, 'r') as file:
        for trial_index, trial_name in enumerate(file.keys()):
            if max_trials is not None and trial_index >= max_trials:
                break
            
            trial_group = file[trial_name]
            trial_times = aligned_stim_times[trial_index]
            valid_time_indices = np.arange(len(trial_times))
            if len(valid_time_indices) == 0:
                continue
            max_time_idx = valid_time_indices[-1]

            for time_point_index, time_point_name in enumerate(trial_group.keys()):
                if time_point_index > max_time_idx:
                    break

                point_index = 0
                time_point_group = trial_group[time_point_name]
                for point_name in time_point_group.keys():
                    point_group = time_point_group[point_name]
                    inside_a = point_group['inside_transformed_outline_A'][()]
                    inside_b = point_group['inside_transformed_outline_B'][()]
                    point_id = f'Point_{point_index}'
                    point_index += 1
                    total_counts[point_id] += 1
                    inside_a_counts[point_id] += int(inside_a)
                    inside_b_counts[point_id] += int(inside_b)

    return total_counts, inside_a_counts, inside_b_counts


def plot_percent_inside(file_path: str, save_path: str, aligned_stim_times: List[np.ndarray], max_trials: Optional[int] = None, max_time_points: Optional[int] = None) -> None:
    try:
        total_counts, inside_a_counts, inside_b_counts = process_file(
            file_path, aligned_stim_times, max_trials=max_trials, max_time_points=max_time_points
        )

        percent_inside_a = {point: (inside_a_counts[point] / total_counts[point]) * 100 for point in total_counts}
        percent_inside_b = {point: (inside_b_counts[point] / total_counts[point]) * 100 for point in total_counts}

        points = list(total_counts.keys())
        percentages_a = [percent_inside_a[point] for point in points]
        percentages_b = [percent_inside_b[point] for point in points]

        chunk_size = 32
        num_chunks = (len(points) + chunk_size - 1) // chunk_size

        fig, axes = plt.subplots(num_chunks, 1, figsize=(12, 6 * num_chunks), squeeze=False)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(points))
            chunk_points = points[start_idx:end_idx]
            chunk_percentages_a = percentages_a[start_idx:end_idx]
            chunk_percentages_b = percentages_b[start_idx:end_idx]

            x = np.arange(len(chunk_points))
            width = 0.35
            ax.bar(x - width / 2, chunk_percentages_a, width, label='InsideOutlineA (%)')
            ax.bar(x + width / 2, chunk_percentages_b, width, label='InsideOutlineB (%)')
            ax.set_xlabel('Test Points or channels')
            ax.set_ylabel('Percent of Time')
            ax.set_title(f'Percentage of Time Test Points Are Inside Outlines (Points {start_idx + 1} to {end_idx})') #points mean channel here
            ax.set_xticks(x)
            ax.set_xticklabels(chunk_points, rotation=45, ha='right')
            ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, format='pdf')
        print(f"Plot saved successfully to: {save_path}")

    except Exception as e:
        print(f"Error processing the file: {e}")

plot_percent_inside(
    file_path=output_hdf5_file,
    save_path=save_path,
    aligned_stim_times=aligned_stim_times,
    #max_trials=10 #None
)
