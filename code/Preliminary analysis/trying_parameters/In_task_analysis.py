import os
from preprocessing import align_ephys as align
from preprocessing import snippet_ephys as snip
from rf import rf_mapping as rfm
import syncopy as spy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### averaging across trials ----------------------------------------------------------------------------------------------------------------------------------------------
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

# selecting trials and doing analysis on saved data 

# load eventdata and data instead of running it everytime so we don't have to switch environments - use /cs/opt/env/python/x86_64/conda/envs/esi-2023b
eventData = spy.load('/mnt/cs/departmentN5/4Shivangi/Datasets/Data/eventData.spy') #load eventdata
datalfp = spy.load('/mnt/cs/departmentN5/4Shivangi/Datasets/Data/datalfp.spy') #load data
os.chdir('/mnt/cs/departmentN5/4Shivangi/Datasets/Data')
trial_info_df = pd.read_pickle("Trial_Info.pkl")

cfg = spy.StructDict()
cfg.latency = [-0.1, 0.1]
data = spy.selectdata(cfg, datalfp)

# time average analysis on entire dataset ----
cfg = spy.StructDict()
cfg.trials='all'
cfg.keeptrials = False
D = spy.timelockanalysis(data, cfg) # timelock analysis
D.channel = data.channel

chan = 2
plt.figure()
sig = D.avg[:,chan]
plt.plot(D.time[0],sig)
plt.show(block = False)

# plot for all channels - identify V1 and V4 ones

for chan in range(0,data.channel.size):
    chan
    if chan == 0:
        sig_chan = D.avg[:,chan]
    else:
        sig_chan = np.vstack([sig_chan, D.avg[:,chan]])
    
Sig_CH = np.array_split(sig_chan, 6)
Sig_CH = np.array(Sig_CH)

for i in range(0,6): #all channels of an array together
    plt.figure()
    for ichan in range(0,32):
        plt.plot(D.time[0],Sig_CH[i,ichan,:])
    plt.show(block = False)

for i in range(0,6): #channles of an array separately
    nrows, ncols = 6, 6
    fig, axes = plt.subplots(nrows, ncols)
    for ichan in range(0,32):
      row, col = ichan // ncols, ichan % ncols
      ax = axes[row, col]
      ax.plot(D.time[0],Sig_CH[i,ichan,:])
    plt.show(block = False)
            
# avg of channles of an array
plt.figure()    
for i in range(0,6):
    avg_ch = np.mean(Sig_CH[i,:,:], axis = 0)
    plt.plot(D.time[0],avg_ch)
plt.show(block = False)

"""
# check channel convention in rf mapping - looks like the channels are already divided into arrays
max_subplots = 32
nchans = 192
nfigs = (nchans-1)//max_subplots

for ifig in range(nfigs+1):
    ifig
    if ifig == nfigs:
        fig_chans = (nchans - ifig*max_subplots)%max_subplots
        if fig_chans == 0:
            fig_chans = max_subplots
    else:
        fig_chans = max_subplots
    
    for iax, ichan in enumerate(range(ifig*max_subplots, ifig*max_subplots+fig_chans)):
        ichan
"""

### correct vs incorrect trials -----------------------------------------------------------------------------------------------------------------------------

# centering trials at stim spawn A with a pre and a post time to be able to average across trials ----
spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder) 
aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[3011],stop=[1,2])
eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)
eventData.time[1]
eventData.save('/mnt/cs/departmentN5/4Shivangi/Data_hm/eventData.spy')

datalfp = spy.load(spy_filename, tag='lfp')
snip.make_trialdefinition(datalfp, eventData)
datalfp.save('/mnt/cs/departmentN5/4Shivangi/Data_hm/datalfp.spy')

# creating trial info table----
Response = [] 
print(eventData.dimord)
for trial in eventData.trials:
    filterlist = np.isin(trial[:,3], list(range(1,3)))
    Response = np.append(Response,trial[filterlist, 3]) #response size and trialno size is differnt when trial stop is [5000,6000]??

TrialNo = [] 
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list(range(1000,2999))) #trial number
    TrialNo = np.append(TrialNo,trial[filterlist, 3]) # why is this different from when we used trial stop to be [5000,6000]? - cuz it seem sthat that also gives bith correct and incorrect trials

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
    
eventData = spy.load('/mnt/cs/departmentN5/4Shivangi/Data_hm/eventData.spy') #load eventdata
datalfp = spy.load('/mnt/cs/departmentN5/4Shivangi/Data_hm/datalfp.spy') #load data