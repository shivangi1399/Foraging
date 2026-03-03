# this script just has some things we tried out to learn about the analysis pipeline from Katharine

#-------------------------------------------------------------------------------------------------------------------------------------------
# RF mapping

from rf import rf_mapping as rfm #run in oesyncopy environments

folder = '/as/projects/MWzeronoise/test_recordings/Cosmos/20230203/BarMapping/001' #saves figures in the corresponding cs folder
rfm.rf_mapping_bar(folder,do_preprocessing=False)

#------------------------------------------------------------------------------------------------------------------------------------------
# snippeting - for a csd session

import os
from preprocessing import align_ephys as align
from preprocessing import snippet_ephys as snip
import syncopy as spy
import matplotlib.pyplot as plt

spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/CSD/001/test_recordings-Cosmos-20230214-CSD-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder)

aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

# obtaining start, stop, trigger - all trials
MappingStimulusOn = 201 #http://esi-svgit001.esi.local/havenith-scholvinck-lab/preprocessing/-/blob/master/preprocessing/snippet_ephys.py#L123
trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[MappingStimulusOn], stop=[3090])
ed = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)
# unadjusted_sample - use this for reward info

# defining trials
data = spy.load(spy_filename, tag='spikes')
snip.make_trialdefinition(data,ed) #this can be done to obtain lfps if we use 'lfp' tag and so on - and then normal syncopy functions can be used for further analysis

# plot psth - use oesyncv2 for this
psth = spy.spike_psth(data, binsize=0.01)
plt.plot(psth.avg)
plt.show(block = False)

#-------------------------------------------------------------------------------------------------------------------------------------------
# obtaining eventmarkers

spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/CSD/001/test_recordings-Cosmos-20230214-CSD-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder)

aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])
# aligned_events - gives list of event markers - we can guess start and stop points from this

#-------------------------------------------------------------------------------------------------------------------------------------------
# obtaining reaction points (use beh_env for this part)

import os
import numpy as np

os.chdir('/mnt/cs/departmentN5/4Shivangi/functions')
from sdm import sess_data_maker 
import reaction_time as rct

basepath_as = '//as/projects/OWzeronoise/Reward_Training/001/'

file_log_mouse = basepath_as+'20230126/RWD/028/2023_01_26-11_19_28_001_RWD_028_RNT_GrassyLandscapeWithBackgroundDark_Cont.log'

sess_data = sess_data_maker(file_log_mouse, 'mouse')

r_time, _ = rct.reaction_time(sess_data, [100, 125, 200])

#---------------------------------------------------------------------------------------------------------------------------------------------
# obtaining info for each trial from eventmarkers

spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder) 
aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

TrialStart = 3000
TrialEnd = 3090

trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[3011], stop=[5000,6000]) #correct trials only - this function helps cut trials correctly. Trigger is where 0 starts
eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)
eventData.time[1] #average trial time is 2.5 seconds

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
Samples = [] #can be used to obtain time 
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

# Pre and post time
# in eventdata.trialdefinition the frst row is start, second stop and third trigger - using these samples we can find sample needed for a particular pre time etc
# check definetrial function from syncopy for code

# Aligning eventmarker to time
tr = 0
t = eventData.time[tr]
samt = eventData.trials[tr][:,1]
t.shape
samt.shape #samt corresponds to each sample corresponding to each eventmarker in the last column - and the corresponding t is the time at that eventmarker
ind = np.where(samt==Samples[0])
t[ind[0][0]]



