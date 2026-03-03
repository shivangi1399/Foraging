# looks at effect of paraemters of that trial on the neural activity

import os
from preprocessing import align_ephys as align
from preprocessing import snippet_ephys as snip
from rf import rf_mapping as rfm
import syncopy as spy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## averaging across trials of different parameters - centered at stim A spawn time
"""
# centering trials at stim spawn A with a pre and a post time to be able to average across trials ----
spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder) 
aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[3011],stop=[5000,6000])
eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)
eventData.time[0]
spy.save(eventData, filename='/mnt/cs/departmentN5/4Shivangi/Data/eventData.spy')

datalfp = spy.load(spy_filename, tag='lfp')
snip.make_trialdefinition(datalfp, eventData)
datalfp.save('/mnt/cs/departmentN5/4Shivangi/Data/datalfp.spy')

# creating trial info table----
# Trial number or trial ID
TrialNo = [] 
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list(range(1000,2999))) #trial number
    TrialNo = np.append(TrialNo,trial[filterlist, 3]) 

# Reward parameter
RewardStimuli = [] #consecutive list of each - this is bascically eventcode or eventmarker
Samples = [] #can be used to obtain time - 
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list(range(5000,5999))) #5000 means no reward - happens when monkey didn't switch blocks. 5500 means 500ms pulse of reward
    RewardStimuli = np.append(RewardStimuli,trial[filterlist, 3]) #sometimes ther might be two rewards - use stimA or StimB reached eventmarkers along with exit button reached eventmarker to cross check
    sample = trial[filterlist,1]
    Samples = np.append(Samples, sample)          

# Difficulty level
DifficultyLevel = [] #consecutive list of each - this is bascically eventcode or eventmarker
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list(range(3400,3600))) #3470 means 70% morph for one stimulus and 30% morph for seccond stimulus - check log files to confirm
    DifficultyLevel = np.append(DifficultyLevel,trial[filterlist, 3])

# Trial Info
trial_info = ({
     'Trial Number': TrialNo,
     'Reward' :RewardStimuli,
     'Difficulty Level':DifficultyLevel,
               })
trial_info_df = pd.DataFrame(trial_info)
os.chdir('/mnt/cs/departmentN5/4Shivangi/Data')
trial_info_df.to_pickle("Trial_Info.pkl")
# we saved the data here because syncopy is not updated in oesyncopy environment
"""
# selecting trials and doing analysis on saved data ----------------------------------------------------------------------------------------------------------

# load eventdata and data instead of running it everytime so we don't have to switch environments - use /cs/opt/env/python/x86_64/conda/envs/esi-2023b
eventData = spy.load('/mnt/cs/departmentN5/4Shivangi/Datasets/Data/eventData.spy') #load eventdata
datalfp = spy.load('/mnt/cs/departmentN5/4Shivangi/Datasets/Data/datalfp.spy') #load data
os.chdir('/mnt/cs/departmentN5/4Shivangi/Datasets/Data')
trial_info_df = pd.read_pickle("Trial_Info.pkl")

cfg = spy.StructDict()
cfg.latency = [-0.2, 0.3]
data = spy.selectdata(cfg, datalfp)

# parameter wise timelock analysis----
RS = np.unique(trial_info_df['Reward'].to_numpy())
DL = np.unique(trial_info_df['Difficulty Level'].to_numpy())
TrialNo = trial_info_df['Trial Number'].to_numpy()

# varying reward param
chan = 10
plt.figure()
for reward_value in RS:
    
    reward_value
    selectM = trial_info_df[(trial_info_df['Reward'] == reward_value)]
    
    troi = selectM['Trial Number'].to_numpy() #trials of interest
    tr = troi.astype(int) 
    tr_ind = np.where(np.isin(TrialNo, tr))[0]  # Find indices where TrialNo matches tr
    troi_idx = tr_ind[:len(tr)]  # Get the indices of trials of interest
    
    cfg = spy.StructDict()
    cfg.trials = troi_idx
    datas = spy.selectdata(cfg, data) #selected data
    
    cfg = spy.StructDict()
    cfg.trials='all'
    cfg.keeptrials = False
    D = spy.timelockanalysis(datas, cfg) # timelock analysis
    
    sig = D.avg[:,chan]
    plt.plot(D.time[0],sig)
    plt.ylim((-60,60))

plt.legend(RS)
plt.title('varying reward level')
plt.show(block = False)

# varying difficulty level
chan = 10
plt.figure()
for diff_level in DL:
    
    diff_level
    selectM = trial_info_df[(trial_info_df['Difficulty Level'] == diff_level)]
    
    troi = selectM['Trial Number'].to_numpy() #trials of interest
    tr = troi.astype(int) 
    tr_ind = np.where(np.isin(TrialNo, tr))[0]  # Find indices where TrialNo matches tr
    troi_idx = tr_ind[:len(tr)]  # Get the indices of trials of interest
    
    cfg = spy.StructDict()
    cfg.trials = troi_idx
    datas = spy.selectdata(cfg, data) #selected data
    
    cfg = spy.StructDict()
    cfg.trials='all'
    cfg.keeptrials = False
    D = spy.timelockanalysis(datas, cfg) # timelock analysis
    
    sig = D.avg[:,chan]
    plt.plot(D.time[0],sig)
    

plt.legend(DL)
plt.title('varying difficulty level')
plt.show(block = False)

# varying difficulty level - easy vs hard
DL = np.unique(trial_info_df['Difficulty Level'].to_numpy())
a = np.array([3430.0, 3470.0])
b = np.array([3449.0, 3451.0])
DL = np.vstack((a,b))
TrialNo = trial_info_df['Trial Number'].to_numpy()

chan = 10
plt.figure()
for diff_level in DL:
    
    diff_level
    selectM = trial_info_df[(trial_info_df['Difficulty Level'] == (diff_level[0] or diff_level[1]))]
    
    troi = selectM['Trial Number'].to_numpy() #trials of interest
    tr = (troi).astype(int) 
    tr_ind = np.where(np.isin(TrialNo, tr))[0]  # Find indices where TrialNo matches tr
    troi_idx = tr_ind[:len(tr)]  # Get the indices of trials of interest
    
    cfg = spy.StructDict()
    cfg.trials = troi_idx
    datas = spy.selectdata(cfg, data) #selected data
    
    cfg = spy.StructDict()
    cfg.trials='all'
    cfg.keeptrials = False
    D = spy.timelockanalysis(datas, cfg) # timelock analysis
    
    sig = D.avg[:,chan]
    plt.plot(D.time[0],sig)

# plot for all channels - identify V1 and V4 arrays ----
Sig_CH = np.array_split(data.channel, 6)
Sig_CH = np.array(Sig_CH)

# for varying difficulty levels
for i in range(0, 6):
    nrows, ncols = 6, 6
    fig, axes = plt.subplots(nrows, ncols) 
    for ichan in range(0, 32):
        row, col = ichan // ncols, ichan % ncols
        ax = axes[row, col]
        
        for diff_level in DL:
            selectM = trial_info_df[trial_info_df['Difficulty Level'] == diff_level]
            troi = selectM['Trial Number'].to_numpy()  # trials of interest
            tr = troi.astype(int)
            tr_ind = np.where(np.isin(TrialNo, tr))[0]  # Find indices where TrialNo matches tr
            troi_idx = tr_ind[:len(tr)]  # Get the indices of trials of interest
            
            cfg = spy.StructDict()
            cfg.trials = troi_idx
            cfg.channel = Sig_CH[i]
            datas = spy.selectdata(cfg, data)  # selected data
            
            cfg = spy.StructDict()
            cfg.trials = 'all'
            cfg.keeptrials = False
            D = spy.timelockanalysis(datas, cfg)  # timelock analysis
            ax.plot(D.time[0], D.avg[:, ichan],label=f'Diff Level: {diff_level}') 
            ax.set_ylim((-60,60))
            Line, Label = ax.get_legend_handles_labels()
    fig.suptitle(f'Array: {i}')
    fig.legend(Line, Label, loc='lower right') 
    axes[5,2].set_visible(False)
    axes[5,3].set_visible(False)
    axes[5,4].set_visible(False)
    axes[5,5].set_visible(False)
    plt.show(block = False)

# for varying difficulty levels - easy vs hard
a = np.array([3430.0, 3470.0])
b = np.array([3449.0, 3451.0])
DL = np.vstack((a,b))
TrialNo = trial_info_df['Trial Number'].to_numpy()

for i in range(0, 6):
    nrows, ncols = 6, 6
    fig, axes = plt.subplots(nrows, ncols) 
    for ichan in range(0, 32):
        row, col = ichan // ncols, ichan % ncols
        ax = axes[row, col]
        
        for diff_level in DL:
            selectM = trial_info_df[trial_info_df['Difficulty Level'] == (diff_level[0] or diff_level[1])]
            troi = selectM['Trial Number'].to_numpy()  # trials of interest
            tr = troi.astype(int)
            tr_ind = np.where(np.isin(TrialNo, tr))[0]  # Find indices where TrialNo matches tr
            troi_idx = tr_ind[:len(tr)]  # Get the indices of trials of interest
            
            cfg = spy.StructDict()
            cfg.trials = troi_idx
            cfg.channel = Sig_CH[i]
            datas = spy.selectdata(cfg, data)  # selected data
            
            cfg = spy.StructDict()
            cfg.trials = 'all'
            cfg.keeptrials = False
            D = spy.timelockanalysis(datas, cfg)  # timelock analysis
            ax.plot(D.time[0], D.avg[:, ichan],label=f'Diff Level: {diff_level}') 
            ax.set_ylim((-60,60))
            Line, Label = ax.get_legend_handles_labels()
    fig.suptitle(f'Array: {i}')
    fig.legend(Line, Label, loc='lower right') 
    axes[5,2].set_visible(False)
    axes[5,3].set_visible(False)
    axes[5,4].set_visible(False)
    axes[5,5].set_visible(False)
    plt.show(block = False)

# for varying reward levels
for i in range(0, 6):
    nrows, ncols = 6, 6
    fig, axes = plt.subplots(nrows, ncols) 
    for ichan in range(0, 32):
        row, col = ichan // ncols, ichan % ncols
        ax = axes[row, col]
        
        for reward_value in RS:
            selectM = trial_info_df[trial_info_df['Reward'] == reward_value]
            troi = selectM['Trial Number'].to_numpy()  # trials of interest
            tr = troi.astype(int)
            tr_ind = np.where(np.isin(TrialNo, tr))[0]  # Find indices where TrialNo matches tr
            troi_idx = tr_ind[:len(tr)]  # Get the indices of trials of interest
            
            cfg = spy.StructDict()
            cfg.trials = troi_idx
            cfg.channel = Sig_CH[i]
            datas = spy.selectdata(cfg, data)  # selected data
            
            cfg = spy.StructDict()
            cfg.trials = 'all'
            cfg.keeptrials = False
            D = spy.timelockanalysis(datas, cfg)  # timelock analysis
            ax.plot(D.time[0], D.avg[:, ichan],label=f'Reward Value: {reward_value}') 
            ax.set_ylim((-60,60))
            Line, Label = ax.get_legend_handles_labels()
    fig.suptitle(f'Array: {i}')
    fig.legend(Line, Label, loc='lower right') 
    axes[5,2].set_visible(False)
    axes[5,3].set_visible(False)
    axes[5,4].set_visible(False)
    axes[5,5].set_visible(False)
    plt.show(block = False)


"""
#################################################################################################################################################################################
## averaging across trials of different parameters - centered at reward time - SIGNAL AROUND REWARD HAS LICKING ARTIFACTS SO WE WILL NOT PURSUE THIS FURTHER

# centering trials at stim spawn A with a pre and a post time to be able to average across trials ----
spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder) 
aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

trldef = snip.create_uneven_trldef(aligned_events, start=[3000],stop=[3090])
eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)
eventData.time[0]

# imp 0 - stick to this for now to check things out
# Reward parameter
RewardStimuli = [] #consecutive list of each - this is bascically eventcode or eventmarker
Samples = [] #can be used to obtain time - 
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list(range(5000,5999))) #5000 means no reward - happens when monkey didn't switch blocks. 5500 means 500ms pulse of reward
    RewardStimuli = np.append(RewardStimuli,trial[filterlist, 3]) #sometimes ther might be two rewards - use stimA or StimB reached eventmarkers along with exit button reached eventmarker to cross check
    sample = trial[filterlist,1]
    Samples = np.append(Samples, sample) 
    
eventData.definetrial(pre=0.2, trigger=RewardStimuli, post=0.3)
eventData.time[100]
eventData.time[0]

datalfp = spy.load(spy_filename, tag='lfp')
snip.make_trialdefinition(datalfp, eventData)
datalfp.save('/mnt/cs/departmentN5/4Shivangi/Datasets/Rew_try/datalfp.spy')

# creating trial info table----
# Trial number or trial ID
TrialNo = [] 
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list(range(1000,2999))) #trial number
    TrialNo = np.append(TrialNo,trial[filterlist, 3]) 

# Reward parameter
RewardStimuli = [] #consecutive list of each - this is bascically eventcode or eventmarker
Samples = [] #can be used to obtain time - 
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list(range(5000,5999))) #5000 means no reward - happens when monkey didn't switch blocks. 5500 means 500ms pulse of reward
    RewardStimuli = np.append(RewardStimuli,trial[filterlist, 3]) #sometimes ther might be two rewards - use stimA or StimB reached eventmarkers along with exit button reached eventmarker to cross check
    sample = trial[filterlist,1]
    Samples = np.append(Samples, sample)          

# Difficulty level
DifficultyLevel = [] #consecutive list of each - this is bascically eventcode or eventmarker
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list(range(3400,3600))) #3470 means 70% morph for one stimulus and 30% morph for seccond stimulus - check log files to confirm
    DifficultyLevel = np.append(DifficultyLevel,trial[filterlist, 3])

# Trial Info
trial_info = ({
     'Trial Number': TrialNo,
     'Reward' :RewardStimuli,
     #'Difficulty Level':DifficultyLevel,
               })
trial_info_df = pd.DataFrame(trial_info)
os.chdir('/mnt/cs/departmentN5/4Shivangi/Datasets/Rew_try')
trial_info_df.to_pickle("Trial_Info.pkl")

# we saved the data here because syncopy is not updated in oesyncopy environment

# selecting trials and doing analysis on saved data ----------------------------------------------------------------------------------------------------------

# load eventdata and data instead of running it everytime so we don't have to switch environments - use /cs/opt/env/python/x86_64/conda/envs/esi-2023b
datalfp = spy.load('/mnt/cs/departmentN5/4Shivangi/Datasets/Rew_try/datalfp.spy') #load data
os.chdir('/mnt/cs/departmentN5/4Shivangi/Datasets/Rew_try')
trial_info_df = pd.read_pickle("Trial_Info.pkl")

data = datalfp

# parameter wise timelock analysis----
RS = np.unique(trial_info_df['Reward'].to_numpy())
DL = np.unique(trial_info_df['Difficulty Level'].to_numpy())
TrialNo = trial_info_df['Trial Number'].to_numpy()

# varying reward param
chan = 100
plt.figure()
for reward_value in RS:
    
    reward_value
    selectM = trial_info_df[(trial_info_df['Reward'] == reward_value)]
    
    troi = selectM['Trial Number'].to_numpy() #trials of interest
    tr = troi.astype(int) 
    tr_ind = np.where(np.isin(TrialNo, tr))[0]  # Find indices where TrialNo matches tr
    troi_idx = tr_ind[:len(tr)]  # Get the indices of trials of interest
    
    cfg = spy.StructDict()
    cfg.trials = troi_idx
    datas = spy.selectdata(cfg, data) #selected data
    
    cfg = spy.StructDict()
    cfg.trials='all'
    cfg.keeptrials = False
    D = spy.timelockanalysis(datas, cfg) # timelock analysis
    
    sig = D.avg[:,chan]
    plt.plot(D.time[0],sig)

plt.legend(RS)
plt.show(block = False)

# varying difficulty level
chan = 10
plt.figure()
for diff_level in DL:
    
    diff_level
    selectM = trial_info_df[(trial_info_df['Difficulty Level'] == diff_level)]
    
    troi = selectM['Trial Number'].to_numpy() #trials of interest
    tr = troi.astype(int) 
    tr_ind = np.where(np.isin(TrialNo, tr))[0]  # Find indices where TrialNo matches tr
    troi_idx = tr_ind[:len(tr)]  # Get the indices of trials of interest
    
    cfg = spy.StructDict()
    cfg.trials = troi_idx
    datas = spy.selectdata(cfg, data) #selected data
    
    cfg = spy.StructDict()
    cfg.trials='all'
    cfg.keeptrials = False
    D = spy.timelockanalysis(datas, cfg) # timelock analysis
    
    sig = D.avg[:,chan]
    plt.plot(D.time[0],sig)

plt.legend(DL)
plt.show(block = False)

"""