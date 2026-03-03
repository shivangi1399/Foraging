# obtain lfp of trials and play around with it a bit

import os
from preprocessing import align_ephys as align
from preprocessing import snippet_ephys as snip
from rf import rf_mapping as rfm
import syncopy as spy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#------ CSD ---------------------------------------------------------------------------------------------------------------------------------------------------------
# obtaining alignments after preprocessing
spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/CSD/001/test_recordings-Cosmos-20230214-CSD-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder)

aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

# obtaining start, stop, trigger - all trials
MappingStimulusOn = 201 #http://esi-svgit001.esi.local/havenith-scholvinck-lab/preprocessing/-/blob/master/preprocessing/snippet_ephys.py#L123
trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[MappingStimulusOn], stop=[3090])
ed = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)

# defining trials - spikes
data = spy.load(spy_filename, tag='spikes')
snip.make_trialdefinition(data,ed) 

# plot psth - use oesyncv2 for this
psth = spy.spike_psth(data, binsize=0.01)
plt.plot(psth.avg)
plt.show()

# defining trials - lfps
datalfp = spy.load(spy_filename, tag='lfp')
snip.make_trialdefinition(datalfp,ed)

# understanding data
trial = datalfp.trials[0]
trial.shape
time = datalfp.time[0]
time.shape

# plotting
chan = 2
plt.plot(time,trial[:,chan])
plt.plot(time,trial)
plt.show()

# spectral analysis
cfg = spy.StructDict()
cfg.method = 'mtmfft'
cfg.foilim = [2,100]
freqpow = spy.freqanalysis(cfg, datalfp)

pstr = freqpow.trials[0]
pstr.shape
freq = freqpow.freq

plt.figure()
plt.plot(freq,pstr[0,0,:,chan])
plt.show(block = False)

#------ task ---------------------------------------------------------------------------------------------------------------------------------------------------------
# obtaining alignments after preprocessing
spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder) 

"""
# redoing steps in line 65 
import glob
evt_pattern = '*_events' + '[0-9]'*6 + '.npy'
matches = glob.glob(os.path.join(compute_folder, evt_pattern))

align_file = ['/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001_events114227.npy']
alignments = []
for file in matches:
    head, tail = os.path.split(file)
    alignments.append(tail[-10:-4])
"""

aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

# obtaining start, stop, trigger - all trials
trldef = snip.create_uneven_trldef(aligned_events, start=[3000], stop=[3090])
ed = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)

# defining trials - lfps
datalfp = spy.load(spy_filename, tag='lfp')
snip.make_trialdefinition(datalfp,ed) #apparently this cuts the data into trials - I can't really understand how from the code tho - ask Katharine

# understanding data
trial = datalfp.trials[1]
trial.shape
time = datalfp.time[1]
time.shape

# plotting the whole trial
plt.figure()
chan = 4
plt.plot(time,trial[:,chan])
plt.plot(time,trial)
plt.show(block = False)

# spectral analysis
cfg = spy.StructDict()
cfg.method = 'mtmfft'
cfg.foilim = [2,100]
freqpow = spy.freqanalysis(cfg, datalfp)

pstr = freqpow.trials[1]
pstr.shape
freq = freqpow.freq

plt.figure()
plt.plot(freq,pstr[0,0,:,chan])
plt.show(block = False)

#---------------------------------------------------------------------------------------------------------------------------------------
# implementing snippeting done in Kim's thesis

# ----- Around Stimulus
spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder) 
aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

TrialStart = 3001 # Placeholder spawns
TrialEnd = 3090 # TrialEnd
Trigger = 3011 #Stimulus A Spawned
pre_t = 0.1
post_t = 0.25

# create an eventdata that has samples + event, but not yet trials
# parameters(data=[sampleNo,Event], labels of cols, samplerate)
trldef = snip.create_uneven_trldef(aligned_events, start=[3000], stop=[3090])
eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)

eventData.definetrial(pre=pre_t, trigger=Trigger, post=post_t) #split consec data into trial sections

datalfp = spy.load(spy_filename, tag='lfp')
snip.make_trialdefinition(datalfp, eventData)

# understanding data
trial = datalfp.trials[0]
trial.shape
time = datalfp.time[0]
time.shape

trial = datalfp.trials[0]
trial.shape
time = datalfp.time[0]
time.shape

# plotting the trial
plt.figure()
chan = 4
plt.plot(time,trial[:,chan])
#plt.plot(time,trial)
plt.plot(time,datalfp.trials[1][:,chan])
plt.show(block = False)

# ---- Around reward
spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder) 
aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

# Define trials between start and stop to collect a list of RewardStimuli later used as triggers
TrialStart = 3000
TrialEnd = 3090
pre_t = 0.1
post_t = 0.1

eventData = spy.EventData(aligned_events[:,[1,3]].astype('int'), dimord=["sample","eventid"], samplerate=snip.evt_samplerate) 
eventData.definetrial(trigger=TrialStart, start = TrialStart, stop = TrialEnd) #use trldef[:,2]*=-1 for these inputs

RewardStimuli = [] #consecutive list of each - this is bascically eventcode or eventmarker
Samples = []
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,1], list(range(5000,5999))) #5000 means no reward - happens when monkey didn't switch blocks
    RewardStimuli = np.append(RewardStimuli,trial[filterlist, 1])
    sample = trial[filterlist,0]
    Samples = np.append(Samples, sample)
    
# Redefine Trials with a padding and the respective Trigger(RewardEvent) for each trial that contains a trigger
eventData.definetrial(pre=pre_t, trigger=RewardStimuli, post=post_t) #split consec data into trial sections

datalfp = spy.load(spy_filename, tag='lfp')
snip.make_trialdefinition(datalfp, eventData)

# understanding data
trial = datalfp.trials[0]
trial.shape
time = datalfp.time[0]
time.shape

# plotting the trial
plt.figure()
chan = 4
plt.plot(time,trial[:,chan])
#plt.plot(time,trial)
plt.plot(datalfp.time[1],datalfp.trials[1][:,chan]) #here time axes is of same length
plt.show(block = False)

#---------------------------------------------------------------------------------------------------------------------------------------
# Selecting Trials of a particular Response and obtaining useful info

# first implementation ----- don't use this
spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder) 
aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

TrialStart = 3000
TrialEnd = 3090
pre_t = 0
post_t = 0

eventData = spy.EventData(aligned_events[:,[1,3]].astype('int'), dimord=["sample","eventid"], samplerate=snip.evt_samplerate) 
eventData.definetrial(start = TrialStart, stop = TrialEnd) #using definetrials in this format might casue some problems so stick with implementation 2, definetrial still works with post and pre time tho
trldef = eventData.trialdefinition
trldef[:,2]*=0
eventData.trialdefinition = trldef
eventData.time[1]

Response = [] #consecutive list of each - this is bascically eventcode or eventmarker
print(eventData.dimord)
for trial in eventData.trials:
    filterlist =np.isin(trial[:,1], list(range(3400,3600))) # obtaining only correct response
    Response = np.append(Response,trial[filterlist, 1])

eventData.definetrial(start = TrialStart, stop = Response) #the difference in first and second implementation is coming from using Start Trial as trigger 

datalfp = spy.load(spy_filename, tag='lfp')
snip.make_trialdefinition(datalfp, eventData)

# understanding data
trial = datalfp.trials[0]
trial.shape
time = datalfp.time[0]
time.shape

# plotting the trial
plt.figure()
chan = 4
plt.plot(datalfp.time[0],datalfp.trials[0][:,chan]) 
plt.plot(datalfp.time[1],datalfp.trials[1][:,chan])
plt.show(block = False)

# second implementation ----- use this
spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder) 
aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

TrialStart = 3000
TrialEnd = 3090

trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[3011], stop=[5000,6000]) #correct trials only
eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)
eventData.time[0]
eventData.trials[0][:,0]
evt_tr = eventData.trials[0][:,-1]
np.unique(evt_tr) #for trial no

Response = [] 
Samples = []
print(eventData.dimord)
for trial in eventData.trials:
    filterlist = np.isin(trial[:,3], list(range(1,2))) #only correct response
    Response = np.append(Response,trial[filterlist, 3])

#eventData.definetrial(start = TrialStart, stop = Response) #don't use this - define a new trialdef

datalfp = spy.load(spy_filename, tag='lfp')
snip.make_trialdefinition(datalfp, eventData)

# understanding data
trial = datalfp.trials[0]
trial.shape
time = datalfp.time[0]
time.shape

# plotting the trial
plt.figure()
chan = 4
plt.plot(datalfp.time[0],datalfp.trials[0][:,chan]) #trial lengths are different and we need to understand the time axis
plt.plot(datalfp.time[100],datalfp.trials[100][:,chan])
plt.show(block = False)

# for this particular thing we could have used eventData.definetrial(start = TrialStart, stop = 1) but this general implementation might be useful elsewhere. 
# This canbe used to obtain range of eventcodes for each trial 

##########################################################################################################################################################################

#---------------------------------------------------------------------------------------------------------------------------------------------
# obtaining info for each trial from eventmarkers and obtaining timing of eventmarkers

spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder) 
aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

TrialStart = 3000
TrialEnd = 3090

trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[3011], stop=[5000,6000]) #correct trials only - this function helps cut trials correctly. Trigger is where 0 starts
eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)
eventData.time[0] #average trial time is 2.5 seconds

# Trial number or trial ID
evt_tr = eventData.trials[0][:,-1]
np.unique(evt_tr) #for trial no - any number between 1000 and 2999 will give trial number
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

                     #define a new trialdef instead of using definetrial function to obtain the trials with different reward etc

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

# Pre and post time
# in eventdata.trialdefinition the frst row is start, second stop and third trigger - using these samples we can find sample needed for a particular pre time etc
# check definetrial function from syncopy for code

# Aligning eventmarker to time ----
tr = 0
t = eventData.time[tr]
samt = eventData.trials[tr][:,1]
t.shape
samt.shape #samt corresponds to each sample corresponding to each eventmarker in the last column - and the corresponding t is the time at that eventmarker
ind = np.where(samt==Samples[tr]) #Samples is the sample at the required eventmarker has occured
t[ind[0][0]] # and this time gives the time at which the eventmarker has occured

# Time points at which stimulus B spawned
StimB = [] 
StimBSam = [] 
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list(range(3020,3022))) 
    StimB = np.append(StimB,trial[filterlist, 3]) 
    sample = trial[filterlist,1]
    StimBSam = np.append(StimBSam, sample)

StimBTime = []
trn = trial_info_df.shape
for tr in range(0,trn[0]):
    t = eventData.time[tr]
    samt = eventData.trials[tr][:,1]
    ind = np.where(samt==StimBSam[tr]) 
    StimBTime = np.append(StimBTime,t[ind[0][0]])

# Time point at which stim A or stim B is reached
StimReach = [] 
StimRSam = [] 
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list([3013,3023])) 
    StimReach = np.append(StimReach,trial[filterlist, 3]) 
    sample = trial[filterlist,1]
    StimRSam = np.append(StimRSam, sample)

StimReachTime = []
trn = trial_info_df.shape
trlist = [i for i in range(trn[0]) if i != 201] #there is some problem with finding this time - check if this eventmarker occurs after reward eventmarker
for tr in trlist:
    tr
    t = eventData.time[tr]
    samt = eventData.trials[tr][:,1]
    ind = np.where(samt==StimRSam[tr]) 
    StimReachTime = np.append(StimReachTime,t[ind[0][0]])

# Time point at which dividing is finished
DivideFinish = [] 
DFSam = [] 
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list([3043])) 
    DivideFinish = np.append(DivideFinish,trial[filterlist, 3]) 
    sample = trial[filterlist,1]
    DFSam = np.append(DFSam, sample)

DFTime = []
trn = trial_info_df.shape
for tr in range(0,trn[0]):
    t = eventData.time[tr]
    samt = eventData.trials[tr][:,1]
    ind = np.where(samt==DFSam[tr]) 
    DFTime = np.append(DFTime,t[ind[0][0]])

# reduce run time --still needs work ----
StimBTimet = []
sampt = []
for tr in range(0,5):#trn[0]):
    sampt = np.vstack((sampt,eventData.trials[tr][:,1]))

for t, samt, stim_sam in zip(eventData.time, eventData.trials[:, 1], StimBSam):
    ind = np.where(samt == stim_sam)[0]
    if len(ind) > 0:
        StimBTimet.append(t[ind[0]])

# data
datalfp = spy.load(spy_filename, tag='lfp')
snip.make_trialdefinition(datalfp, eventData)

# plotting the trial with eventmarkers
plt.figure()
chan = 4
plt.plot(datalfp.time[1],datalfp.trials[1][:,chan]) #trial lengths are different
plt.axvline(x = StimBTime[1], color = 'r', label = 'StimBSpawn', linestyle='dashed') 
plt.axvline(x = DFTime[1], color = 'b', label = 'dividing finish', linestyle='dashed')
plt.plot(datalfp.time[2],datalfp.trials[2][:,chan])
plt.axvline(x = StimBTime[2], color = 'r', label = 'StimBSpawn', linestyle='dashed') 
plt.axvline(x = DFTime[2], color = 'b', label = 'dividing finish', linestyle='dashed')
plt.show(block = False)

## separating trials based on parameters ----
RS = np.unique(RewardStimuli)
DL = np.unique(DifficultyLevel)
reward_value = RS[1]
diff_value = DL[0]
selectM = trial_info_df[(trial_info_df['Reward'] == reward_value) & (trial_info_df['Difficulty Level'] == diff_value)] #dataframe with selected parameters
selectM = trial_info_df[(trial_info_df['Reward'] == reward_value)]
selectM

troi = selectM['Trial Number'].to_numpy() #trials of interest
troi

plt.figure()
chan = 4
for tr in troi:
    tr = int(tr)
    tr_ind = np.where(tr == TrialNo)
    tr_ind = tr_ind[0][0]
    plt.plot(datalfp.time[tr_ind],datalfp.trials[tr_ind][:,chan])

plt.show(block = False)

###########################################################################################################################################################################
## averaging across trials of different parameters

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

""" 
# trying to use definetrial to select pre and post data - does not work; just use another function like selectdata to cut trials shorter instead of trying to change the trialdef
eventData.definetrial(pre=pre_t, trigger=3011, post=post_t) #split consec data into trial sections
trldef = eventData.trialdefinition
trldef[:,2]*= -1  #correction so that the trigger it at t = 0
eventData.trialdefinition = trldef  # maybe we need to write our own code for pre and post time and cutting data around certain time
"""

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

# selecting trials and doing analysis on saved data ----------------------------------------------------------------------------------------------------------
# load eventdata and data instead of running it everytime so we don't have to switch environments - use /cs/opt/env/python/x86_64/conda/envs/esi-2023b
eventData = spy.load('/mnt/cs/departmentN5/4Shivangi/Data/eventData.spy') #load eventdata
datalfp = spy.load('/mnt/cs/departmentN5/4Shivangi/Data/datalfp.spy') #load data
os.chdir('/mnt/cs/departmentN5/4Shivangi/Data')
trial_info_df = pd.read_pickle("Trial_Info.pkl")

cfg = spy.StructDict()
cfg.latency = [-0.1, 0.1]
data = spy.selectdata(cfg, datalfp)

RS = np.unique(trial_info_df['Reward'].to_numpy())
DL = np.unique(trial_info_df['Difficulty Level'].to_numpy())
reward_value = RS[1]
diff_value = DL[0]
#selectM = trial_info_df[(trial_info_df['Reward'] == reward_value) & (trial_info_df['Difficulty Level'] == diff_value)] #dataframe with selected parameters
selectM = trial_info_df[(trial_info_df['Reward'] == reward_value)]

TrialNo = trial_info_df['Trial Number'].to_numpy()
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
D = spy.timelockanalysis(datas, cfg)

# timelock analysis
chan = 4
plt.figure()
plt.plot(D.time[0],D.avg[:,chan])
plt.show(block = False)

       # rest of the syncopy analysis based on parameters is done in parameter_analysis.py scripy