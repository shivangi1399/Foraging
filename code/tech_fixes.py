import os
from preprocessing import align_ephys as align
from preprocessing import snippet_ephys as snip
from rf import rf_mapping as rfm
import syncopy as spy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# whole trial
# using snippeting functions
spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder) 
aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger=[3011],stop=[3090])
eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)
eventData.time[0]

# using definetrial
spy_filename = '/cs/projects/MWzeronoise/test_recordings/Cosmos/20230214/LeafForaging/001/test_recordings-Cosmos-20230214-LeafForaging-001.spy'
compute_folder = os.path.dirname(spy_filename)
alignments = align.find_all_alignments(compute_folder) 
aligned_events, basename = align.load_aligned_eventmarkers(compute_folder, alignments[0])

eventdata = spy.EventData(aligned_events[:,[1,3]].astype('int'), dimord=["sample","eventid"], samplerate=snip.evt_samplerate) 
eventdata.definetrial(start = [3000], trigger=[3011], stop = [3090])
eventdata.time[0]

eventdata = spy.EventData(aligned_events[:,[1,3]].astype('int'), dimord=["sample","eventid"], samplerate=snip.evt_samplerate) 
eventdata.definetrial(pre = 0, start = [3000], trigger=[3011], stop = [3090], post = 0)
eventdata.time[0]

eventdata = spy.EventData(aligned_events[:,[1,3]].astype('int'), dimord=["sample","eventid"], samplerate=snip.evt_samplerate) 
eventdata.definetrial(start = [3000], trigger=[3011], stop = [3090])
eventdata.time[0]
trldef = eventData.trialdefinition
trldef[:,2]*=0
eventData.trialdefinition = trldef
eventData.time[0]

# part of data
TrialStart = 3000
TrialEnd = 3090 
Trigger = 3011
pre_t = 0.1
post_t = 0.1

trldef = snip.create_uneven_trldef(aligned_events, start=[3000], stop=[3090])
eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)

eventData.definetrial(pre=pre_t, trigger=Trigger, post=post_t) 
eventData.time[0]

###########################################################################################
# reward as trigger implementations


"""
# imp 1
trldef = eventData.trialdefinition
trldef[:,2] = Samples
eventData.trialdefinition = trldef

# imp 2 - just try definetrial function with new version of syncopy - does not work ughh
os.chdir('/mnt/cs/departmentN5/4Shivangi/Datasets')
np.save('alignev.npy', aligned_events)

aligned_events = np.load('alignev.npy')

TrialStart = 3000
TrialEnd = 3090
pre_t = 0.1
post_t = 0.1

eventData = spy.EventData(aligned_events[:,[1,3]].astype('int'), dimord=["sample","eventid"], samplerate=1000) 
eventData.definetrial(trigger=TrialStart, start = TrialStart, stop = TrialEnd) #use trldef[:,2]*=-1 for these inputs
trldef[:,2]*=-1
eventData.trialdefinition = trldef

RewardStimuli = [] #consecutive list of each - this is bascically eventcode or eventmarker
Samples = []
print(eventData.dimord)
for trial in eventData.trials:
    
    filterlist =np.isin(trial[:,1], list(range(5000,5999))) #5000 means no reward - happens when monkey didn't switch blocks
    RewardStimuli = np.append(RewardStimuli,trial[filterlist, 1])
    sample = trial[filterlist,0]
    Samples = np.append(Samples, sample)

eventData.definetrial(pre=0.2, trigger=RewardStimuli, post=0.3)
eventData.time[100]

# imp 3
import re
trldef = snip.create_uneven_trldef(aligned_events, start=[3000], trigger = [5400], stop=[3090])
eventData = spy.EventData(aligned_events, dimord=["unadjusted_sample","sample","log_times","eventid"], trialdefinition=trldef, samplerate=snip.evt_samplerate)
eventData.time[0]

# pattern = r"5[0-9]{3}"
# matches = re.findall(pattern, data)
"""