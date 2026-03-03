import numpy as np
from parse_logfile import TextLog
from preprocessing.align_ephys import find_subsequence


def align_irec(logfile, irec_netfile):
    # load data
    irec_evt = np.genfromtxt(irec_netfile, delimiter=',', skip_header=1)

    with TextLog(logfile) as log:
        evt, ts, evt_desc, true_ts = log.parse_eventmarkers()

    # match log start
    rand_id = evt[evt > 30000]
    st_log = find_subsequence(irec_evt[:,1], rand_id)

    if st_log.size < 1:
        print(rand_id)
        raise ValueError('No matches found in %s for %s'%(irec_netfile, logfile))

    print(st_log)
    irec_evt = irec_evt[st_log[0]:,:]

    # align irec to the start of trial 1
    startTrial = 3000
    first_trl = true_ts[evt == startTrial][0]

    irec_first_trl = irec_evt[:,0][irec_evt[:,1] == startTrial][0]
    log_irec_offset = first_trl - irec_first_trl # add this value to irec times to make them Unreal times

    return log_irec_offset


def match_irec_times(log_times, irec_t, log_irec_offset):
    # assume irec has much higher sampling rate than log
    irec_idx =np.searchsorted(irec_t, log_times-log_irec_offset)

    return irec_idx


def subsample_irec(irec_idx, irec_x, irec_y):
    # assumes an usually equal gap between indicies
    n_samples = irec_idx.size
    # x = np.zeros(n_samples)
    # y = np.zeros(n_samples)
    gap = np.diff(irec_idx).min()
    left = gap // 2
    right = left + gap % 2

    new_idx = np.r_[[np.arange(x-left,x+right, dtype=int) for x in irec_idx]]

    x = np.median(irec_x[new_idx], axis=1)
    y = np.median(irec_y[new_idx], axis=1)

    return x,y


def irec2log(irec_x, irec_y, irec_t, log_times, log_irec_offset):
    irec_idx = match_irec_times(log_times, irec_t, log_irec_offset)
    x,y = subsample_irec(irec_idx, irec_x, irec_y)
    return x,y,irec_t[irec_idx]

