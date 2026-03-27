import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog

### Data structure for RF VR mapping data #################################################################################################
"""
- `RF_stim_collapse` contains all the information needed to determine, for every RF and every time point, whether the RF overlaps with the 
   stimulus
- The data structure is as follows: 
        trial_X
        └── time_point_Y
            ├── attrs:
            │   └── collapsed_case
            │
            ├── Point_1_xx_yy  --> this corresponds to the different channels
            │   ├── RF_center
            │   ├── radius
            │   ├── inside_transformed_outline_A
            │   └── inside_transformed_outline_B
            ├── Point_2_xx_yy
            └── ...
- All RFs are iterated one-to-one between `center_coords` and `rf_radius`, and only RFs with non-NaN centers and radii are tested 
  and saved—any RF with NaNs is skipped entirely and does not appear in the output HDF5.

"""

### Aligning time between RF VR mapping data and neural data #################################################################################################
# getting stim corner info ----
folder = '//cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data/20230214/'
filename = folder + '2023_02_14-11_42_27_Cosmos_LeafForaging_001_PAF_GrassyLandscapeWithBackgroundDark_Cont.log'

# get stimulus path on each trial ----
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
    
    indx = [ii for ii, name in enumerate(log.all_ids['name']) if name.startswith(stim_name)] # finding all image stimulus and their IDs
    for ii, istim in enumerate(indx):
        if ii + n_stimuli == len(indx):
            break
        this_id = log.all_ids[istim]
        next_id = log.all_ids[indx[ii + n_stimuli]] # this should be the spawn time of the image in the next trial
        loc, pos_ts = log.parse_spherical(obj_id=this_id['id'],
                                              st=this_id['start'],
                                              end=next_id['start']) #location and time
        params = log.parse_initial_parameters(obj_id = this_id['id'], st = this_id['start'], end = next_id['start']) #stim parameters
        
        stim_loc.append(loc) #loc for each stim with time
        stim_ts.append(pos_ts) #corresponding time for each stim when it was on - includes all the display images
        stim_params.append(params)

vertical_offset = np.append(arr = vertical_offset, values = [np.uint16(stimulus['Height']) for i, stimulus in enumerate(stim_params)])
stim_width = np.append(arr = stim_width, values = [np.uint16(stimulus['Scale'] * 200) for i, stimulus in enumerate(stim_params)]) # In Unreal units


# Align the timepoints with just the time points when the stimulus is on ----
n_trials = int(stim_width.shape[0]/5) #number of trials

aligned_stim_times_list = [] 

for itrl in range(n_trials):
    
    stim_times = stim_ts[itrl * n_stimuli + 3].T
    aligned_stim_times = stim_times - stim_times[0]
    aligned_stim_times_list.append(aligned_stim_times)

print("Aligned Stimulus Times for Trial 0:", aligned_stim_times_list[9]) 
# Time zero corresponds to the first logged time point of the target stimulus in that trial. The neural data is aligned to stimulus onset
# which means that the time 0 in neural data and the RF VR mapping data is the same. In the RF VR mapping, each time point is numbered but
# you can find which exact time point it corresposnds to using aligned_stim_times_list[trial_number][time_pont_number]