#### Code instructions #######################################################################################################
# 1. run mapping_corner.py for stimuli corner in cartesian coordinates
# 2. run prepare_RF.py for Rfs in cartseian coordinates
# 3. order the corner using corner_order.py and run warping_stim.py to warp the stimuli in the corners we got in step 1
# 4. Check if RF falls inside the warped stim using RF_in_stim.py
# 5. Check how everything looks using check_RF_in_stim.ipynb
##############################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/eyetracking')
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/convert_unreal_coordinates')
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
import time_conversion as tc
import irec_conversion as irec
import dome_conversion as dc
from convert_unreal_coordinates import relative_spherical
from parse_logfile import TextLog
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter  #we can use warping env in this

folder = '//cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data/20230214/'
filename = folder + '2023_02_14-11_42_27_Cosmos_LeafForaging_001_PAF_GrassyLandscapeWithBackgroundDark_Cont.log'

# get stimulus path on each trial
stim_loc = []
stim_ts = []
stim_params = []

stim_width = np.zeros(0, dtype = np.uint16)
vertical_offset = np.zeros(0, dtype = np.uint16)

stim_name = 'ImageStimulus'#'MultipleStimul' 

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

vertical_offset = np.append(arr = vertical_offset, values = [np.uint16(stimulus['Height']) for i, stimulus in enumerate(stim_params)])
stim_width = np.append(arr = stim_width, values = [np.uint16(stimulus['Scale'] * 100) for i, stimulus in enumerate(stim_params)]) # In Unreal units


# eye data --------------------------------------------------------------------------------------------------------
eye_file = '114146'

log_irec_offset = tc.align_irec(filename, folder+eye_file+'net.csv')
irec_pos = np.genfromtxt(folder+eye_file+'.csv',delimiter=',',skip_header=1)
eye_evt = np.genfromtxt(folder+eye_file+'net.csv',delimiter=',',skip_header=1)

# get eye position on each trial
evt_start = 3000
trl_starts = eye_evt[:, 0][eye_evt[:, 1] == evt_start]
eye_trl = []
eye_trl_ts = []
eye_trl_evt = []
for ist, st in enumerate(trl_starts):
    ist
    st
    if ist * n_stimuli >= len(stim_ts):
        break
    
    # gives the eye positions and irec times for the leaf stimuli when the leaf stimuli comes on (in each trial) - does not contain the placeholders etc
    eye_x, eye_y, eye_ts = tc.irec2log(irec_pos[:,1],irec_pos[:,2],irec_pos[:,0], 
                                                    stim_ts[ist * n_stimuli+3], log_irec_offset) 
    
    eye_trl.append(np.stack((eye_x, eye_y), 1))
    eye_trl_ts.append(eye_ts) 
    
    stim_time = stim_ts[ist * n_stimuli+3][-1] - stim_ts[ist * n_stimuli+3][0]
    inc_evt = np.logical_and(eye_evt[:, 0] >= st, eye_evt[:, 0] < st + stim_time)
    eye_trl_evt.append(eye_evt[inc_evt, :]) #event data for the duration the leaf is on (maybe)
    
# convert stimulus to eye coordinates ------------------------------------------------------------------------------
# find_stimulus_corners changed here: with vertical offset added
def find_stimulus_corners(azimuth, elevation, radius, width = 100, height = 100, vertical_offset = 0):

    xorig, yorig, zorig = relative_spherical.spherical2relative(azimuth, elevation, radius)
    
    a0, e0, r = relative_spherical.relative2spherical(xorig, yorig + width / 2, zorig + vertical_offset) #bottom right
    a1, e1, r = relative_spherical.relative2spherical(xorig, yorig - width / 2, zorig + vertical_offset) #bottom left
    a2, e2, r = relative_spherical.relative2spherical(xorig, yorig + width / 2, zorig + height + vertical_offset) #top right
    a3, e3, r = relative_spherical.relative2spherical(xorig, yorig - width / 2, zorig + height + vertical_offset) #top left
    
    a = np.stack((a0, a2 ,a3, a1))  #br-tr-tl-bl
    e = np.stack((e0, e2 ,e3, e1))  #br-tr-tl-bl
    
    return a, e

eye_coords = np.array([1.5, 2.93, -13.77]) #this is head position
eccentricity_trl = []
polar_trl = []

for itrl, irec_xy in enumerate(eye_trl): #gives spherical coordinates of the four corners of both the stimuli
    
    x, y, z = stim_loc[itrl * n_stimuli + 3].T #x, y and z is a series of locations with time for the trial
    dome_x1, dome_y1 = find_stimulus_corners(azimuth = x,
                                            elevation = y,
                                            radius = z,
                                            width = stim_width[itrl * n_stimuli + 3],
                                            height = stim_width[itrl * n_stimuli + 3],
                                            vertical_offset = vertical_offset[itrl * n_stimuli + 3])
    
    x, y, z = stim_loc[itrl * n_stimuli + 4].T
    dome_x2, dome_y2 = find_stimulus_corners(azimuth = x,
                                            elevation = y,
                                            radius = z,
                                            width = stim_width[itrl * n_stimuli + 4],
                                            height = stim_width[itrl * n_stimuli + 4], 
                                            vertical_offset = vertical_offset[itrl * n_stimuli + 4])
    
    eccentricity, polar = dc.dome2eye(dome_x = np.concatenate((dome_x1.T, dome_x2.T), axis = 1),
                                        dome_y = np.concatenate((dome_y1.T, dome_y2.T), axis = 1),
                                        irec_x = irec_xy[:, 0],
                                        irec_y = irec_xy[:, 1],
                                        eye_coords = eye_coords) 
    eccentricity_trl.append(eccentricity) #each element is a trial, row is time and columns has corners of the two stimuli 
    polar_trl.append(polar)

# --------------------------------------------------------------------------------------------------------------------------------
### converting stimuli coordinates from retinal to cartesian - to check if RF lies inside meshes

def retinal2cartesian(eccentricity, polar, R=1): # Function to convert polar and eccentricity to Cartesian coordinates
    if np.isnan(eccentricity) or np.isnan(polar):
        return np.array([np.nan, np.nan, np.nan])
    
    lat = np.deg2rad(90 - eccentricity)
    lon = np.deg2rad(360 - polar)
    
    x = R * np.cos(lat) * np.cos(lon)
    y = -R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    
    return np.array([x, y, z])

# Initialize list to hold Cartesian coordinates
cartesian_trials = []

# Process each trial
for trial_polar, trial_eccentricity in zip(polar_trl, eccentricity_trl):
    trial_cartesian = []
    for time_point_polar, time_point_eccentricity in zip(trial_polar, trial_eccentricity):
        stimulus1_polar = time_point_polar[:4]  #This goes into a trial in polar_trl and eccentricity_trl and in one time point and takes the first 4 coordinates, which are the 4 corners of mesh 1. For example trial 1, time point 1 is (polar_trl[0][0][:4]). Same for eccentricity_trl
        stimulus1_eccentricity = time_point_eccentricity[:4]
        stimulus2_polar = time_point_polar[4:]
        stimulus2_eccentricity = time_point_eccentricity[4:] # In the same way this is (polar_trl[0][0][4:]) for trial 1, time point 1, mesh 2
        
        # Convert stimulus 1 coordinates. Stimulus 1 is basically the first in polar_trl[trial][time_point][:4], and eccentricity_trl[trial][time_point][4:]
        stimulus1_coords = [retinal2cartesian(e, p) for e, p in zip(stimulus1_eccentricity, stimulus1_polar)]
        
        # Convert stimulus 2 coordinates. 
        stimulus2_coords = [retinal2cartesian(e, p) for e, p in zip(stimulus2_eccentricity, stimulus2_polar)]
        
        # Combine both stimuli coordinates for the time point
        time_point_coords = stimulus1_coords + stimulus2_coords
        trial_cartesian.append(time_point_coords)
    
    cartesian_trials.append(trial_cartesian)

cartesian_trials = np.array(cartesian_trials, dtype=object)

print(f"Length of cartesian_trials: {len(cartesian_trials)}")  # This should be 1066, same as the total amount of trials

os.chdir('/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/corners/20230214')
np.save('cartesian_trials.npy', cartesian_trials)

#cartesian_trials is a list with length 1066, and it has the cartesian coordinates for all the meshes. the structure of the list is cartesian_trials[trial][time point][corners]
#cartesian_trials is a list of lists. It has 1066 lists for all the trials. The first list of them (trial 1), has lists inside it (as many as the time points of that trial). 
#Each time point has a 2D array with 8 1D arrays (8 corners) inside it 

# if you want to get the 4 corners of mesh 1 for trial 1, time point 1:  cartesian_trials[0][0][:4]  and for mesh 2: cartesian_trials[0][0][4:] 
print(cartesian_trials[0][0]) 



