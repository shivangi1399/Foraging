
"""
#### Code instructions #######################################################################################################
# 1. run mapping_corner.py for stimuli corner in cartesian coordinates
# 2. run prepare_RF.py for Rfs in cartseian coordinates
# 3. order the corner using corner_order.py and run warping_stim.py to warp the stimuli in the corners we got in step 1
# 4. Check if RF falls inside the warped stim accounting for collapse using RFarea_stim_collapse.py
# 5. Check how everything looks using RFoverlap_perc_collapse.py
##############################################################################################################################
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy import interpolate 
import matplotlib.pyplot as plt
from skimage.transform import ProjectiveTransform
from skimage import io
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from shapely.geometry import Polygon
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog #run this in warping env


# Load in the stimuli corners for the entire session in cartesian coordinates 
sessions = ['20230203', '20230208', '20230209', '20230213'] #, '20230214']

for session in sessions:

    path = f'/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/corners/{session}/'
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    cartesian_trials = np.load('cartesian_trials.npy', allow_pickle=True)
    cartesian_trials_list = cartesian_trials.tolist()

    # Changing the order of the 4 corners from CBAD to DCBA
    def reorder_corners(corners_list):
        
        if len(corners_list) != 8: # Ensure there are exactly 8 corners
            raise ValueError("There must be exactly 8 corners to reorder (4 per stimulus).")
        
        # Separate the corners for each stimulus
        stimulus1_corners = corners_list[:4]  # First 4 corners
        stimulus2_corners = corners_list[4:]  # Last 4 corners

        # Reorder the corners for each stimulus
        reordered_stimulus1 = [stimulus1_corners[3], stimulus1_corners[0], stimulus1_corners[1], stimulus1_corners[2]]
        reordered_stimulus2 = [stimulus2_corners[3], stimulus2_corners[0], stimulus2_corners[1], stimulus2_corners[2]]
        
        # Combine the reordered corners
        reordered_list = reordered_stimulus1 + reordered_stimulus2
        
        return reordered_list

    def convert_to_dict_list(cartesian_trials_list):
        trials_reordered_list = []

        #This loop iterates over each trial in cartesian_trials_list
        for trial_index, trial in enumerate(cartesian_trials_list):
            trial_dict = {}  # Initialize an empty dictionary for the current trial

        #This inner loop iterates over each time point in the current trial
            for time_point_index, time_point in enumerate(trial):
                # Reorder the corners and store in the dictionary
                reordered_time_point = reorder_corners(time_point)
                trial_dict[time_point_index] = reordered_time_point
            
            # Append the dictionary for this trial to the list
            trials_reordered_list.append(trial_dict)

        return trials_reordered_list


    trials_reordered_list = convert_to_dict_list(cartesian_trials_list)

    print(f"Number of trials (dictionaries): {len(trials_reordered_list)}")
    # trials_reordered_list has the number of lists equal to the numbe of trials. Each list has as many 
    # dictionaries as time points with time point index as key and the values of the corners as 2D arrays
    # if you want to get the coordinates of the 4 corners of trial 1, time point 1 and mesh 1 in that time 
    # point : trials_reordered_list[0][0][:4]

    #saving the dictionary
    path = f'/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/reordered_corners/{session}/'
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    with open('trials_reordered_list.pkl', 'wb') as file: 
        pickle.dump(trials_reordered_list, file)

