import os
from preprocessing import align_ephys as align
from preprocessing import snippet_ephys as snip
from rf import rf_mapping as rfm
import syncopy as spy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

# Checking sessions and trials for cosmos data ----------------------------------------------------------------------------------------------------------------------------------
# Extracting all foraging sessions
dir_path = '/cs/projects/MWzeronoise/test_recordings/Cosmos'
sess_list = os.listdir(dir_path)
sess_path = []
for sess in sess_list:
    path = os.path.join(dir_path, sess)
    sess_path.append(path)

foraging_paths_all = []
for path in sess_path:
    if os.path.isdir(os.path.join(path, 'LeafForaging')):
        foraging_paths_all.append(path)

# Extracting foraging sessions after a certain date - to get clean signals
dir_path = '/cs/projects/MWzeronoise/test_recordings/Cosmos'
sess_list = os.listdir(dir_path)
sess_use_all = np.array([int(x) for x in sess_list])
sess_use = (sess_use_all[sess_use_all>20230201]) #after a certain data
sess_use.tolist()
sess_use = [str(element) for element in sess_use]

sess_path = []
for sess in sess_use:
    path = os.path.join(dir_path, sess)
    sess_path.append(path)

foraging_paths = []
for path in sess_path:
    if os.path.isdir(os.path.join(path, 'LeafForaging')):
        foraging_paths.append(path)

# checking number of trials in each sessions
tr = []
for path in foraging_paths:
    tr.append(os.listdir(os.path.join(path, 'LeafForaging')))

blk = []
T = np.asarray(tr)
for i in range(0,T.size):
    blk.append(tr[i][0]) #assuming the last block is the one with correct data

sess_names = []
for i in range(0,T.size):
    os.chdir(os.path.join(foraging_paths[i], 'LeafForaging', blk[i]))
    sess_names.append(glob.glob('*.spy'))

trial_no = []
for i in [0,1,2,3,5,6,7]: #range(0,T.size) 
    i
    spy_filename = os.path.join(foraging_paths[i], 'LeafForaging', blk[i], sess_names[i][0])
    compute_folder = os.path.dirname(spy_filename)
    alignments = align.find_all_alignments(compute_folder) 
    aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])
    
    trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[3011],stop=[5000,6000])
    eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)
    trial_no.append(eventData.sampleinfo.shape[0])

del sess_names[4] #still need to run preprocessing for sess 4
sess_info = ({
     'Session' :sess_names,
     'Trail Number':trial_no,
               })
sess_info_df = pd.DataFrame(sess_info)

os.chdir('/mnt/cs/departmentN5/4Shivangi/Train_Rec_Info/Session_Info')
sess_info_df.to_csv('Foraging_recording_info_Cosmos.csv')

# checking trial length distribution
trial_length = []
for i in [0,1,2,3,5,6,7]: #range(0,T.size) 
    i
    spy_filename = os.path.join(foraging_paths[i], 'LeafForaging', blk[i], sess_names[i][0])
    compute_folder = os.path.dirname(spy_filename)
    alignments = align.find_all_alignments(compute_folder) 
    aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])
    
    trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[3011],stop=[5000,6000])
    eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)
    
    for tr_time in eventData.time:
        trial_length.append(round(tr_time[-1],3))


counts, bins = np.histogram(trial_length)
plt.stairs(counts, bins)

data = trial_length
for i in data:
   i
   if i > 30:
      data.remove(i)

plt.figure()
w=0.1
plt.hist(data, bins=np.arange(min(data), max(data) + w, w))
plt.show(block = False)

# Checking training sessions for Karl ------------------------------------------------------------------------------------------------------------------------------------------
dir_path = '/as/projects/MWzeronoise/domeVRTraining/Karl'
sess_list = os.listdir(dir_path)
sess_path = []
for sess in sess_list:
    path = os.path.join(dir_path, sess)
    sess_path.append(path)

unwanted_list = ['._.DS_Store', '.DS_Store']
for i in unwanted_list:
    unwanted_path = os.path.join(dir_path, i)

sess_paths = [ele for ele in sess_path if ele not in unwanted_path]

Diff_Tasks = []
for path in sess_paths:
    task = os.listdir(path)
    unwanted_list = ['._.DS_Store', '.DS_Store', 'tmp']
    Task = [ele for ele in task if ele not in unwanted_list]
    Diff_Tasks.append(Task)

# Training Info
unwanted_list = ['._.DS_Store', '.DS_Store']
sess_lists = [ele for ele in sess_list if ele not in unwanted_list]
sess_info = ({
     'Session' :sess_lists,
     'Tasks':Diff_Tasks,
               })
sess_info_df = pd.DataFrame(sess_info)
sorted_sess_info_df = sess_info_df.sort_values(by='Session')

os.chdir('/mnt/cs/departmentN5/4Shivangi/Train_Rec_Info/Session_Info')
#sorted_sess_info_df.to_csv('Training_info_Karl.csv')