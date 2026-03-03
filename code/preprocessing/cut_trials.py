# Cut trials from neural data and saving trial info from log files using cut_trials.py
# We can convert all the data into matlab format using cosmos_data_matlab.m
# Use artifact_rejection.m to remove artifacts from LFP data using fieldtrip, at the end convert data to syncopy friendly format
# Load cleaned data back into syncopy using load_cleaned_data.py

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from datetime import datetime
import seaborn as sns
from pathlib import Path
from scipy.io import savemat

# Custom module paths
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog
from preprocessing import align_ephys as align
from preprocessing import snippet_ephys as snip
import syncopy as spy

# Cutting and saving neural data - centered at stim A spawn time -------------------------------------------
# The trial is cut between start and trial end
foldername = '/cs/projects/MWzeronoise/test_recordings/Cosmos'
filenames = [
    '20230202/LeafForaging/002/test_recordings-Cosmos-20230202-LeafForaging-002',
    '20230203/LeafForaging/001/test_recordings-Cosmos-20230203-LeafForaging-001',
    #'20230206/LeafForaging/001/test_recordings-Cosmos-20230206-LeafForaging-001'  #no lfp data in spy folder
    '20230208/LeafForaging/001/test_recordings-Cosmos-20230208-LeafForaging-001',
    '20230209/LeafForaging/001/test_recordings-Cosmos-20230209-LeafForaging-001',
    '20230213/LeafForaging/002/test_recordings-Cosmos-20230213-LeafForaging-002',
    '20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001'
]

data_directory = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/full_length'

# loading neural data and cutting trials----
for filename in filenames:
    filename
    # Construct the spy file path
    spy_filename = os.path.join(foldername, filename + '.spy')
    compute_folder = os.path.dirname(spy_filename)
    
    # Find and load alignments
    alignments = align.find_all_alignments(compute_folder)
    aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

    # method to create trial definition
    trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[3011], stop=[3090, 3091])
    eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample", "sample", "log_times", "eventid"], 
                               trialdefinition=trldef, samplerate=snip.evt_samplerate)

    subfolder_name = filename.split('/')[0]  # Extracts '20230202', '20230203', etc.
    target_directory = os.path.join(data_directory, subfolder_name)

    # Load LFP data from the original spy file
    datalfp = spy.load(spy_filename, tag='lfp')
    snip.make_trialdefinition(datalfp, eventData)
    datalfp_save_path = os.path.join(target_directory, 'datalfp.spy')
    datalfp.save(datalfp_save_path)


# loading the trial info----
raw_data_folder = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/raw_data'

def session_metadata(log):
    """
    Return header dict + BoxSize (reads first ImageStimulus).
    """
    log.make_id_struct()
    
    meta = log.read_log_header_with_return()
    
    # ImageStimulus params → box size
    indx = [ii for ii,name in enumerate(log.all_ids['name']) if name.startswith('ImageStimulus')]
    #ids= log.all_ids['id'][indx]

    for i in range(100):
        img = log.parse_initial_parameters(log.all_ids[indx[i]]['id'],log.all_ids[indx[i]]['start'],log.all_ids[indx[i]]['end'])
        if img['StimulusCategory']:

            boxsize=img['Scale']*img['StimulusOverlapCollisionBoxScale'][1]*100
            print(img['StimulusOverlapCollisionBoxScale'])
            break
    
    meta['BoxSize']=boxsize
    return meta


for file_rel in filenames:

    # Extract session date
    session_dfs = []
    session_date = file_rel.split('/')[0]
    # Find log file in the corresponding folder
    log_dir = Path(raw_data_folder) / session_date
    log_files = list(log_dir.glob("*.log"))
    
    if not log_files:
        print(f"No log file found for session {session_date}, skipping.")
        continue
    
    # Take the first log file (or refine pattern if multiple)
    log_file = log_files[0]

    with TextLog(log_file) as log:
        # 1. Parse trials
        trials = log.get_info_per_trial(
            return_eventmarkers=True,
            return_loc=True,
            start=3000
        )
        df = pd.DataFrame(trials)

        # 2. Metadata
        meta = session_metadata(log)
        df.attrs['Metadata'] = meta

        # 3. Session info
        df['Session'] = session_date
        df['Session_Info'] = f"{meta['Start Time'][:10]}_{meta['Subject']}_{meta['Session']}"

        session_dfs.append(df)

    # Create a list of only event columns from all sessions
    event_data = [df[['Event']].copy() for df in session_dfs]

    # add session info to keep track
    for i, df in enumerate(event_data):
        df['Session'] = session_dfs[i]['Session']
    
    # Defining trial info
    def classify_event_list(event_list):
        """Extract trial, reward, difficulty from a list of events."""
        trial = reward = difficulty = pd.NA
        for val in event_list:
            if 1000 <= val <= 2999:
                trial = val
            elif 5000 <= val <= 5999:
                reward = val
            elif 3400 <= val <= 3600:
                difficulty = val
        return pd.Series([trial, reward, difficulty])

    # Process all session dataframes
    trial_info = pd.concat([
        df['Event'].apply(classify_event_list).rename(columns={0:'Trial_Number', 1:'Reward', 2:'Difficulty'})
        .assign(
              Session = df['Session'],
        )
        for df in event_data
    ], ignore_index=True)

    trial_info_clean = trial_info.where(pd.notnull(trial_info), np.nan)

    target_directory = os.path.join(data_directory, session_date)
    trial_info_save_path = os.path.join(target_directory, 'Trial_Info.pkl')
    trial_info.to_pickle(trial_info_save_path)
    trial_info_mat_path = os.path.join(target_directory, 'Trial_Info.mat')
    savemat(trial_info_mat_path, {'trial_info': trial_info_clean.to_dict('list')})
    print(trial_info)


    # Save all event markers per trial into a CSV ----
    eventmarkers_list = []
    for trial_idx, trial in enumerate(df['Event']):
        eventmarkers_list.append({
            "Session": session_date,
            "Trial_Index": trial_idx + 1,
            "EventMarkers": ",".join(map(str, trial))  # store events as comma-separated list
        })

    events_df = pd.DataFrame(eventmarkers_list)
    eventmarkers_csv_path = os.path.join(target_directory, 'EventMarkers.csv')
    events_df.to_csv(eventmarkers_csv_path, index=False)
    print(f"Saved event markers CSV for session {session_date}")


"""
# Cutting and saving neural data - centered at stim A spawn time ------------------------------------------- don't use this codes
# The trial is cut between start and reward - didn't work out very well cuz could not reproduce this - use log file instead of eventdata

foldername = '/cs/projects/MWzeronoise/test_recordings/Cosmos'
filenames = [
    '20230202/LeafForaging/002/test_recordings-Cosmos-20230202-LeafForaging-002',
    '20230203/LeafForaging/001/test_recordings-Cosmos-20230203-LeafForaging-001',
    #'20230206/LeafForaging/001/test_recordings-Cosmos-20230206-LeafForaging-001'  #no lfp data in spy folder
    '20230208/LeafForaging/001/test_recordings-Cosmos-20230208-LeafForaging-001',
    '20230209/LeafForaging/001/test_recordings-Cosmos-20230209-LeafForaging-001',
    '20230213/LeafForaging/002/test_recordings-Cosmos-20230213-LeafForaging-002',
    '20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001'
]

data_directory = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/200_600'

for filename in filenames:
    filename
    # Construct the spy file path
    spy_filename = os.path.join(foldername, filename + '.spy')
    compute_folder = os.path.dirname(spy_filename)
    
    # Find and load alignments
    alignments = align.find_all_alignments(compute_folder)
    aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

    # method to create trial definition
    trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[3011], stop=[5000, 6000])
    eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample", "sample", "log_times", "eventid"], 
                               trialdefinition=trldef, samplerate=snip.evt_samplerate)

    # Determine the subdirectory name based on the first part of the filename
    subfolder_name = filename.split('/')[0]  # Extracts '20230202', '20230203', etc.
    target_directory = os.path.join(data_directory, subfolder_name)
    
    # Ensure the target directory exists
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    # Save eventData to the target directory
    event_data_path = os.path.join(target_directory, 'eventData.spy')
    eventData.save(event_data_path)

    # Load LFP data from the original spy file
    datalfp = spy.load(spy_filename, tag='lfp')
    snip.make_trialdefinition(datalfp, eventData)
    datalfp_save_path = os.path.join(target_directory, 'datalfp.spy')
    datalfp.save(datalfp_save_path)

    # Creating trial info table ----
    TrialNo = [] 
    for trial in eventData.trials:
        filterlist = np.isin(trial[:, 3], list(range(1000, 2999)))  # trial number
        TrialNo = np.append(TrialNo, trial[filterlist, 3])
        
    RewardStimuli = []
    Samples = []
    for trial in eventData.trials:
        filterlist = np.isin(trial[:, 3], list(range(5000, 5999)))  # reward parameter
        RewardStimuli = np.append(RewardStimuli, trial[filterlist, 3])
        sample = trial[filterlist, 1]
        Samples = np.append(Samples, sample)
        
    DifficultyLevel = []
    for trial in eventData.trials:
        filterlist = np.isin(trial[:, 3], list(range(3400, 3600)))  # difficulty level
        DifficultyLevel = np.append(DifficultyLevel, trial[filterlist, 3])
        
    # Create the trial_info DataFrame
    trial_info = {
        'Trial Number': TrialNo,
        'Reward': RewardStimuli,
        'Difficulty Level': DifficultyLevel,
    }
    
    trial_info_df = pd.DataFrame(trial_info)
    trial_info_save_path = os.path.join(target_directory, 'Trial_Info.pkl')
    trial_info_df.to_pickle(trial_info_save_path)
    print(trial_info_df)
"""


