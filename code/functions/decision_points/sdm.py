# obtaining decision points

# use beh_env for this part

import os
import numpy as np

os.chdir('/mnt/cs/departmentN5/4Shivangi/functions')
from parse_logfile_newest import TextLog

def sess_data_maker(filename, species, evt_for_mouse = 3000):
    sess_x = []
    sess_y = []
    sess_t = []

    with TextLog(filename) as log:
        if species=='mouse':
            trinfo=log.get_info_per_trial(return_eventmarkers=True,return_loc=True, start=evt_for_mouse)
        else:
            trinfo=log.get_info_per_trial(return_eventmarkers=True,return_loc=True)

    locc = np.array(trinfo["Location"])
    locTs = np.array(trinfo["LocationTs"])

    for tr_id in range(len(locTs)):

        tr_x = locc[tr_id][:,0] - locc[tr_id][:,0][0]
        tr_y = locc[tr_id][:,1] - locc[tr_id][:,1][0]
        tr_t = locTs[tr_id] - locTs[tr_id][0]
        sess_x.append(tr_x)
        sess_y.append(tr_y)
        sess_t.append(tr_t)

    sess_data = np.vstack([sess_x,sess_y,sess_t]).T

    return sess_data



#---------------------------------------------------------------------------------------------------------------------------------------------
"""

# make sessions

basepath_as = '//as/projects/OWzeronoise/Reward_Training/001/'
filename = basepath_as+'20230126/RWD/028/2023_01_26-11_19_28_001_RWD_028_RNT_GrassyLandscapeWithBackgroundDark_Cont.log'
species =  'mouse'          
evt_for_mouse = 3000

sess_x = []
sess_y = []
sess_t = []

with TextLog(filename) as log:
    trinfo=log.get_info_per_trial(return_eventmarkers=True,return_loc=True, start=evt_for_mouse)
        
    
locc = np.array(trinfo["Location"])
locTs = np.array(trinfo["LocationTs"])

for tr_id in range(len(locTs)):
    print("Trial id:", tr_id)
    
    tr_x = locc[tr_id][:,0] - locc[tr_id][:,0][0]
    tr_y = locc[tr_id][:,1] - locc[tr_id][:,1][0]
    tr_t = locTs[tr_id] - locTs[tr_id][0]
    
    sess_x.append(tr_x) #all the trials here are unequal which is why vstacks doesn't compile it - check if they need to be equal length and how to do it
    sess_y.append(tr_y)
    sess_t.append(tr_t)
    
    tr_t.shape

np.array(sess_y)
sess_data = np.vstack([sess_x,sess_y,sess_t]).T

"""