# Looks into the effect of trial/block history on neural activity 

import os
from preprocessing import align_ephys as align
from preprocessing import snippet_ephys as snip
from rf import rf_mapping as rfm
import syncopy as spy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

####### Within block analysis

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

##############################################################################################################################################################################
## ANALYSING TRIAL/BLOCK HISTORY

# Loading data - use /cs/opt/env/python/x86_64/conda/envs/esi-2023b
datalfp = spy.load('/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/Trial_sess_history/Data_20230214/datalfp.spy')
os.chdir('/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/Trial_sess_history/Data_20230214')
trial_info_df = pd.read_pickle("Trial_Info.pkl")
TrialNo_block = np.load('TrialNo_block.npy') #trials at which monkey moved towards apple to change blocks

# cutting trials
cfg = spy.StructDict()
cfg.latency = [-0.2, 0.3]
data = spy.selectdata(cfg, datalfp)

# looking at trial history ----
# plotting one block
trn = 4
trn_b = int(TrialNo_block[trn+1])
trn_a = int(TrialNo_block[trn])

signal = []
chan = 4
for tn in range(trn_a,trn_b):
    tn
    a = data.trials[tn][:,chan]
    signal = np.hstack((a,signal))

plt.figure()
plt.plot(signal)
xcc = 501
xcoords = np.arange(501,501*16,501)
for xc in xcoords:
    plt.axvline(x=xc, color = 'r', ls=':')
plt.show(block = False)

# filter trials by previous trial reward ----
RS = np.unique(trial_info_df['Reward'].to_numpy())
TrialNo = trial_info_df['Trial Number'].to_numpy()

chan = 10
plt.figure()
for reward_value in RS: #reward of previous trial
    
    reward_value
    selectM = trial_info_df[(trial_info_df['Reward'] == reward_value)]
    
    troi = selectM['Trial Number'].to_numpy() #trials of interest
    tr = (troi+1).astype(int) 
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
plt.title('effect of varying reward level in previous trial')
plt.show(block = False)

# transition trial grouping by difficulty level ----
a = np.array([3430.0, 3470.0])
b = np.array([3449.0, 3451.0])
DL = np.vstack((a,b))
TrialNo = trial_info_df['Trial Number'].to_numpy()

chan = 10
plt.figure()
for i, diff_level in enumerate(DL): #only at block change
    selectM = trial_info_df[(trial_info_df['Difficulty Level'] == diff_level[0]) | (trial_info_df['Difficulty Level'] == diff_level[1])]
    troi = selectM['Trial Number'].to_numpy().astype(int)
    
    if i == 0:  # Easy to Hard transition
        selectMp = trial_info_df[(trial_info_df['Difficulty Level'] == DL[1][0]) | (trial_info_df['Difficulty Level'] == DL[1][1])]
    else:  # Hard to Easy transition
        selectMp = trial_info_df[(trial_info_df['Difficulty Level'] == DL[0][0]) | (trial_info_df['Difficulty Level'] == DL[0][1])]
        
    troip = (selectMp['Trial Number'].to_numpy() - 2).astype(int)
    
    troi_comm = np.intersect1d(troi, troip)
    tr_ind = np.where(np.isin(TrialNo, troi_comm))[0]
    
    cfg = spy.StructDict(trials=tr_ind+1) #looks at trial after the block change  #spy.StructDict(trials=tr_ind) - if we want to average the trial before the block
    datas = spy.selectdata(cfg, data)
    
    cfg = spy.StructDict(trials='all', keeptrials=False)
    D = spy.timelockanalysis(datas, cfg)
    
    sig = D.avg[:, chan]
    plt.plot(D.time[0], sig)

plt.title('Effect of varying difficulty level in previous trial')
plt.legend(['easy_hard', 'hard_easy'])
plt.show(block=False)

# trial grouping by difficulty level and reward level at block change ----
troi_comm = np.intersect1d(TrialNo+1, TrialNo_block) #block change trials
tr_indp = np.where(np.isin(TrialNo, troi_comm-1))[0]
df_p = trial_info_df.loc[tr_indp]
df_n = trial_info_df.loc[tr_indp+1] #it is +1 because the df doesn't have the block change trials 

# difficulty level
dfp_easy = df_p.loc[df_p['Difficulty Level'].isin(DL[0])] 
dfp_hard = df_p.loc[df_p['Difficulty Level'].isin(DL[1])]
trp_easy = dfp_easy['Trial Number'].to_numpy().astype(int)
trp_hard = dfp_hard['Trial Number'].to_numpy().astype(int)
dfn_easy = df_n.loc[df_n['Difficulty Level'].isin(DL[0])] 
dfn_hard = df_n.loc[df_n['Difficulty Level'].isin(DL[1])]
trn_easy = dfn_easy['Trial Number'].to_numpy().astype(int)
trn_hard = dfn_hard['Trial Number'].to_numpy().astype(int)

plt.figure()
for tr_setp in [trp_hard, trp_easy]:
    tr_setp
    for tr_setn in [trn_hard, trn_easy]:
        tr_setn
        
        troi_sets = np.intersect1d(tr_setp, tr_setn-2)
        tr_ind = np.where(np.isin(TrialNo, troi_sets))[0]
        tr_ind.shape
        
        cfg = spy.StructDict(trials=tr_ind+1) #looks at trial after the block change  #spy.StructDict(trials=tr_ind) - if we want to average the trial before the block
        datas = spy.selectdata(cfg, data)
        
        cfg = spy.StructDict(trials='all', keeptrials=False)
        D = spy.timelockanalysis(datas, cfg)
        
        sig = D.avg[:, chan]
        plt.plot(D.time[0], sig)

plt.title('Effect of varying difficulty level in previous trial')
plt.legend(['hard_hard','hard_easy','easy_hard', 'easy_easy'])
plt.show(block=False)
  
# reward level
RS = np.unique(trial_info_df['Reward'].to_numpy())
dfp_low = df_p.loc[df_p['Reward']==(RS[1])] 
dfp_high = df_p.loc[df_p['Reward']==(RS[2])]
trp_low = dfp_low['Trial Number'].to_numpy().astype(int)
trp_high = dfp_high['Trial Number'].to_numpy().astype(int)
dfn_low = df_n.loc[df_n['Reward']==(RS[1])] 
dfn_high = df_n.loc[df_n['Reward']==(RS[2])]
trn_low = dfn_low['Trial Number'].to_numpy().astype(int)
trn_high = dfn_high['Trial Number'].to_numpy().astype(int)

plt.figure()
for tr_setp in [trp_low]: #high to high had only one case and rest none
    tr_setp
    for tr_setn in [trn_high]:
        tr_setn
        
        troi_sets = np.intersect1d(tr_setp, tr_setn-2)
        tr_ind = np.where(np.isin(TrialNo, troi_sets))[0]
        tr_ind.shape
        
        cfg = spy.StructDict(trials=tr_ind+1) #looks at trial after the block change  #spy.StructDict(trials=tr_ind) - if we want to average the trial before the block
        datas = spy.selectdata(cfg, data)
        
        cfg = spy.StructDict(trials='all', keeptrials=False)
        D = spy.timelockanalysis(datas, cfg)
        
        sig = D.avg[:, chan]
        plt.plot(D.time[0], sig)

plt.title('Effect of varying reward level in previous trial - at block change')
plt.legend(['low_high'])
plt.show(block=False)
  
# neural activity within block ----
troi_comm = np.intersect1d(TrialNo+1, TrialNo_block) #block change trials
tr_indp = np.where(np.isin(TrialNo, troi_comm-1))[0]

# group trials by blocks and create block info df
df_n = trial_info_df.loc[tr_indp+1] #this gives block start trials 
blk_st = df_n['Trial Number'].to_numpy().astype(int)
trial_blk = [range(blk_st[i], blk_st[i+1]-1) for i in range(len(blk_st)-1)] #gives trials in a block

no_tr_blk = [] #number of trials in a block
diff_blk = []
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

# plot neural activity for early and late block trials
split_factor = 10

early_blk = np.empty((0,), dtype=int) 
late_blk = np.empty((0,), dtype=int)

for subarray in trial_blk: #obtaining early and late trials for all the blocks
    trl_blk = np.array_split(np.array(subarray), split_factor)
    early_indices = np.where(np.isin(TrialNo, trl_blk[0]))[0]
    early_blk = np.hstack((early_blk, early_indices))
    
    late_indices = np.where(np.isin(TrialNo, trl_blk[-1]))[0]
    late_blk = np.hstack((late_blk, late_indices))
    
cfg = spy.StructDict(trials=early_blk) #obtaining neural activitry for all early and late trials
datas = spy.selectdata(cfg, data)
cfg = spy.StructDict(trials='all', keeptrials=False)
D_early = spy.timelockanalysis(datas, cfg)
    
cfg = spy.StructDict(trials=late_blk)
datas = spy.selectdata(cfg, data)
cfg = spy.StructDict(trials='all', keeptrials=False)
D_late = spy.timelockanalysis(datas, cfg)
    
plt.figure()    
sig_early = D_early.avg[:, chan]
sig_late = D_late.avg[:, chan]
plt.plot(D_early.time[0], sig_early)
plt.plot(D_early.time[0], sig_late)
plt.title('neural activity for early and late block trials')
plt.legend(['early trials','late trials'])
plt.show(block=False)

# plot neural activity for early and late block trials - split by diff level

blk_easy = block_info_df[block_info_df['Difficulty Level'] == 0] # Filter block information based on difficulty level
blk_hard = block_info_df[block_info_df['Difficulty Level'] == 1]
easy_bln = blk_easy['block number'].to_numpy().astype(int)
hard_bln = blk_hard['block number'].to_numpy().astype(int)

easy_trblk = [trial_blk[i] for i in easy_bln] # Extract trial blocks corresponding to easy and hard blocks
hard_trblk = [trial_blk[i] for i in hard_bln]

split_factor = 5

def get_early_late_indices(trblk):
    early_indices = []
    late_indices = []
    for subarray in trblk:
        trl_blk = np.array_split(np.array(subarray), split_factor)
        early_indices.extend(np.where(np.isin(TrialNo, trl_blk[0]))[0])
        late_indices.extend(np.where(np.isin(TrialNo, trl_blk[-1]))[0])
    return np.array(early_indices), np.array(late_indices)

early_blk_easy, late_blk_easy = get_early_late_indices(easy_trblk) # Obtain early and late trials for easy and hard blocks
early_blk_hard, late_blk_hard = get_early_late_indices(hard_trblk)
    
cfg = spy.StructDict(trials=early_blk_easy) #obtaining neural activitry for all early and late trials
datas = spy.selectdata(cfg, data) #310 trials
cfg = spy.StructDict(trials='all', keeptrials=False)
D_early_easy = spy.timelockanalysis(datas, cfg)

cfg = spy.StructDict(trials=late_blk_easy) #290 trials
datas = spy.selectdata(cfg, data)
cfg = spy.StructDict(trials='all', keeptrials=False)
D_late_easy = spy.timelockanalysis(datas, cfg)

cfg = spy.StructDict(trials=early_blk_hard) #184 trials
datas = spy.selectdata(cfg, data)
cfg = spy.StructDict(trials='all', keeptrials=False)
D_early_hard = spy.timelockanalysis(datas, cfg)
    
cfg = spy.StructDict(trials=late_blk_hard) #159 trials
datas = spy.selectdata(cfg, data)
cfg = spy.StructDict(trials='all', keeptrials=False)
D_late_hard = spy.timelockanalysis(datas, cfg)
    
    
plt.figure()    
sig_early_easy = D_early_easy.avg[:, chan]
sig_late_easy = D_late_easy.avg[:, chan]
sig_early_hard = D_early_hard.avg[:, chan]
sig_late_hard = D_late_hard.avg[:, chan]
plt.plot(D_early.time[0], sig_early_easy)
plt.plot(D_early.time[0], sig_late_easy)
plt.plot(D_early.time[0], sig_early_hard)
plt.plot(D_early.time[0], sig_late_hard)
plt.title('neural activity for early and late block trials')
plt.legend(['early_easy','late_easy','early_hard','late_hard'])
plt.show(block=False)

# neural activity withing block right after stim onset - trial by trial basis -----
cfg = spy.StructDict()
cfg.latency = [0, 0.1]
data_st = spy.selectdata(cfg, data)

Amp = [] #define an amplitude in 100ms after stim onset - do it for one channel for now
chan = 10
for tr in data_st.trials:
    x = tr[:,chan]
    xm = np.mean(x)
    Amp.append((np.sum((x-xm)**2))**0.5)
    
# plot amp for all trials divided by blocks
TrialNo = trial_info_df['Trial Number'].to_numpy()
troi_comm = np.intersect1d(TrialNo+1, TrialNo_block) #block change trials
tr_indp = np.where(np.isin(TrialNo, troi_comm-1))[0]

plt.figure()
plt.plot(range(0,981),Amp)
for xc in tr_indp:
    plt.axvline(x=xc, color = 'r', ls=':')
plt.show(block = False)




















##------------------------------------------------------------------------------------------------------------------------------------------------------------  
# plot for all channels and arrays
Sig_CH = np.array_split(data.channel, 6)
Sig_CH = np.array(Sig_CH)

# for varying reward levels in  the prev trial
for i in range(0, 6):
    nrows, ncols = 6, 6
    fig, axes = plt.subplots(nrows, ncols) 
    for ichan in range(0, 32):
        row, col = ichan // ncols, ichan % ncols
        ax = axes[row, col]
        
        for reward_value in RS:
            selectM = trial_info_df[trial_info_df['Reward'] == reward_value]
            troi = selectM['Trial Number'].to_numpy()  # trials of interest
            tr = (troi+1).astype(int)
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

# for varying difficulty levels in the prev trial
for i in range(0, 6):
    nrows, ncols = 6, 6
    fig, axes = plt.subplots(nrows, ncols) 
    for ichan in range(0, 32):
        row, col = ichan // ncols, ichan % ncols
        ax = axes[row, col]
        
        for diff_level in DL:
            selectM = trial_info_df[trial_info_df['Difficulty Level'] == diff_level]
            troi = selectM['Trial Number'].to_numpy()  # trials of interest
            tr = (troi+1).astype(int)
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



