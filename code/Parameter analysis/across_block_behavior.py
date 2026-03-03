# This will help us look at how immediate reward levels and general decreasing reward probability affects monkey’s performance
import os
from preprocessing import align_ephys as align
from preprocessing import snippet_ephys as snip
from rf import rf_mapping as rfm
import syncopy as spy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## PREPARING DATA
spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder) 
aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

# looking at blocks ----
trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[3011], stop=[3091])
eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)

TrialNo = [] 
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list(range(1000,2999))) #trial number
    TrialNo = np.append(TrialNo,trial[filterlist, 3]) 

TrialNo_block = TrialNo
os.chdir('/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/Trial_sess_history/Data_20230214')
np.save('TrialNo_block.npy', TrialNo_block)

# looking at trials ----
trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[3011], stop=[5000,6000])
eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)

# Trial Info
TrialNo = [] 
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list(range(1000,2999))) #trial number
    TrialNo = np.append(TrialNo,trial[filterlist, 3]) 

RewardStimuli = [] #consecutive list of each - this is bascically eventcode or eventmarker
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list(range(5000,5999))) #5000 means no reward - happens when monkey didn't switch blocks. 5500 means 500ms pulse of reward
    RewardStimuli = np.append(RewardStimuli,trial[filterlist, 3])  

DifficultyLevel = [] #consecutive list of each - this is bascically eventcode or eventmarker
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,3], list(range(3400,3600))) #3470 means 70% morph for one stimulus and 30% morph for seccond stimulus - check log files to confirm
    DifficultyLevel = np.append(DifficultyLevel,trial[filterlist, 3])

trial_info = ({
     'Trial Number': TrialNo,
     'Reward' :RewardStimuli,
     'Difficulty Level':DifficultyLevel,
               })
trial_info_df = pd.DataFrame(trial_info)
os.chdir('/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/Trial_sess_history/Data_20230214')
trial_info_df.to_pickle("Trial_Info.pkl")

datalfp = spy.load(spy_filename, tag='lfp')
snip.make_trialdefinition(datalfp, eventData)
datalfp.save('/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/Trial_sess_history/Data_20230214/datalfp.spy')
        
# Responses and the trial info for which response variable is saved
Response = []  #why is response variable not saved for all the trials??
TrialNo_res = []
diflev = []
RewStim = []
for trial in eventData.trials:
    filterlist = np.isin(trial[:, 3], list(range(1, 3)))
    Response = np.append(Response, trial[filterlist, 3])
    
    if np.any(np.isin(trial[filterlist, 3], list(range(1, 3)))):
        filterlistr = np.isin(trial[:, 3], list(range(1000, 2999)))
        TrialNo_res = np.append(TrialNo_res, trial[filterlistr, 3])
        
        filterlist =np.isin(trial[:,3], list(range(3400,3600)))
        diflev = np.append(diflev,trial[filterlist, 3])
        
        filterlist =np.isin(trial[:,3], list(range(5000,5999)))
        RewStim = np.append(RewStim,trial[filterlist, 3])
        
trial_info_res = ({
     'Trial Number': TrialNo_res,
     'Reward' :RewStim,
     'Difficulty Level':diflev,
     'Response':Response
               })
trial_info_res = pd.DataFrame(trial_info_res)
os.chdir('/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/Trial_sess_history/Data_20230214')
trial_info_res.to_pickle("Trial_Info_res.pkl")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Loading data 
datalfp = spy.load('/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/Trial_sess_history/Data_20230214/datalfp.spy')
os.chdir('/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/Trial_sess_history/Data_20230214')
#trial_info_df = pd.read_pickle("Trial_Info.pkl")
trial_info_df = pd.read_pickle("Trial_Info_res.pkl") #select the trial info depeding on the analysis - here we want to only include the resposne trials
TrialNo_block = np.load('TrialNo_block.npy') #trials at which monkey moved towards apple to change blocks

# cutting trials
cfg = spy.StructDict()
cfg.latency = [-0.2, 0.3]
data = spy.selectdata(cfg, datalfp)

# block info ------- 
# group trials by blocks and create block info df
TrialNo = trial_info_df['Trial Number'].to_numpy()
troi_comm = np.intersect1d(TrialNo+1, TrialNo_block) #block change trials
tr_indp = np.where(np.isin(TrialNo, troi_comm-1))[0]

df_n = trial_info_df.loc[tr_indp+1] #this gives block start trials 
blk_st = df_n['Trial Number'].to_numpy().astype(int)
trial_blk = [range(blk_st[i], blk_st[i+1]-1) for i in range(len(blk_st)-1)] #gives trials in a block

no_tr_blk = [] #number of trials in a block
diff_blk = []
reward_blk = []
a = np.array([3430.0, 3470.0])
b = np.array([3449.0, 3451.0])
for i in trial_blk:
    df_blk = trial_info_df.loc[trial_info_df['Trial Number'].isin(i)]
    diff = np.unique(df_blk['Difficulty Level'].to_numpy()) 
    no_tr_blk.append(len(df_blk))
    
    if np.all(diff == a):
        diff_blk.append(0)
    else:
        diff_blk.append(1)
    
block_info = ({
    'block number' : range(0,(np.array(diff_blk).shape[0])),
    'Trials': trial_blk,
    'No of trials' :no_tr_blk,
    'Difficulty Level':diff_blk,
               })
block_info_df = pd.DataFrame(block_info)

# reward and behavior info for each block
ncols = 8
fig, axes = plt.subplots(int((np.array(diff_blk).shape[0]-1)/ncols), ncols)
value_map = {5000.: 0, 5150.: 1, 5400.: 2}
for j in range(0,(np.array(diff_blk).shape[0])-1): #only looking at first 56 blocks
    i = trial_blk[j]
    
    df_blk = trial_info_df.loc[trial_info_df['Trial Number'].isin(i)]
    rew = df_blk['Reward'].to_numpy()
    res = df_blk['Response'].to_numpy()
    rew = np.vectorize(value_map.get)(rew)
    
    row, col = j // ncols, j % ncols
    ax = axes[row,col]
    diflev = block_info_df.iloc[j,3] 
    if diflev == 0:
        ax.plot(rew,color='b')
        ax.yaxis.set(ticks=np.arange(0, 3, 1))
    else:
        ax.plot(rew,color='r')
        ax.yaxis.set(ticks=np.arange(0, 3, 1))
        
    incorr_tr = [i for i, val in enumerate(res) if val == 2] # identify incorrect trials
    for xc in incorr_tr:
        ax.axvline(x=xc, color = 'black', ls=':')
    
plt.show(block = False)
    