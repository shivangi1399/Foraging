"""
=============================================================================================
Process the neural (LFP) data into one continuous stream per channel.
=============================================================================================
Pipeline position: step 2 (see README.md). Consumes full_mask_reg.npz from Regressors.py (step 1)
and writes neural_data.npz -- the fit target that DesignMatrix.py (step 3) and FittingGLM.py
(step 4) use.

- Load the LFP data
- Set neural activity after reward onset to 0 (reward contamination)
- Stitch the trials into a continuous stream
- Drop invalid trials and zero-mean the result

Channels are parallelised across SLURM workers with acme; each worker is self-contained
(loads the LFP, parses the log for reward timing, processes its channel, saves neural_data.npz).

Run in the warping env (needs acme: `conda install -c conda-forge esi-acme`).
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import syncopy as spy

from acme import ParallelMap

sys.path.insert(1, '/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog

# -------------------------
# Config
# -------------------------
data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets'
lfp_data_dir = os.path.join(data_dir, 'neural_data/stimAalign_cut/clean_full_length')
log_file_dir = os.path.join(data_dir, 'raw_data')
results_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp/all_trials/full_length/GLM'

sessions = ['20230214']
channels = [2]

# SLURM / parallelisation config
SLURM_PARTITION = '8GBS'
MAX_WORKERS = 100
MEM_PER_WORKER = '8GB'


# -------------------------
# Helpers
# -------------------------

def find_logfile(log_dir, session_name):
    """Find the single unreal .log for a session inside log_dir/session_name."""
    session_dir = os.path.join(log_dir, session_name)
    log_files = sorted(glob.glob(os.path.join(session_dir, '*.log')))
    if len(log_files) == 0:
        raise FileNotFoundError(f"No .log file found in {session_dir}")
    if len(log_files) > 1:
        raise ValueError(f"Multiple .log files found in {session_dir}:\n" + "\n".join(log_files))
    return log_files[0]


def remove_nan_trials_channels(datas):
    """Drop all-NaN trials/channels left in place when the data was cleaned."""
    trial_mask = [not np.all(np.isnan(tr)) for tr in datas.trials]
    if not any(trial_mask):
        return None, [], trial_mask

    trial_stack = np.concatenate(datas.trials, axis=0)
    valid_ch_idx = np.where(~np.all(np.isnan(trial_stack), axis=0))[0]
    if len(valid_ch_idx) == 0:
        return None, [], trial_mask

    datas_clean = spy.selectdata(spy.StructDict(channel=valid_ch_idx), datas)
    valid_channels = [datas.channel[i] for i in valid_ch_idx]
    return datas_clean, valid_channels, trial_mask


def ensure_trialindex_in_trialdefinition(datalfp):
    """Ensure trialdefinition has a 4th column holding the trial index (0..nTrials-1)."""
    try:
        if datalfp.trialdefinition.shape[1] < 4:
            nTrials = datalfp.trialdefinition.shape[0]
            datalfp.trialdefinition = np.hstack(
                (datalfp.trialdefinition, np.arange(nTrials).reshape(-1, 1)))
    except Exception as e:
        print("Warning: could not ensure trialindex in trialdefinition:", e)


# -------------------------
# Per-channel worker
# -------------------------

def process_channel(channel, session_name, lfp_data_dir, log_file_dir, results_dir):
    """Build the continuous, reward-trimmed, zero-meaned neural stream for one channel."""
    filename = find_logfile(log_file_dir, session_name)

    # Load the LFP and select this channel
    datalfp = spy.load(os.path.join(lfp_data_dir, session_name, 'Cleaned_lfp_FT.spy'))
    ensure_trialindex_in_trialdefinition(datalfp)
    data = spy.selectdata(spy.StructDict(), datalfp)
    channel_data = data.selectdata(channel=f"channel_{channel:03d}")
    channel_clean, _, trial_mask = remove_nan_trials_channels(channel_data)

    # Trial outcome + reward timing from the log
    with TextLog(filename) as log:
        df = pd.DataFrame(log.get_info_per_trial(return_eventmarkers=True, return_loc=False))
        evt, ts, _, _ = log.parse_eventmarkers()

    outcome_mask = (df["Correct"] != 0) | (df["Wrong"] != 0)   # exclude no-response trials
    full_mask = trial_mask & outcome_mask

    trial_onset_ts = ts[np.where(evt == 3000)[0]][:-1]           # drop the last (non-existent) trial
    trial_end_ts = ts[np.where(evt == 3090)[0]]
    reward_onset_ts = ts[np.where((evt >= 5000) & (evt < 6000))[0]]

    # Seconds from trial onset to the first reward inside the trial (0 if none)
    dur_reward_sec = []
    for i in range(len(trial_onset_ts)):
        r = reward_onset_ts[(reward_onset_ts > trial_onset_ts[i]) & (reward_onset_ts < trial_end_ts[i])]
        dur_reward_sec.append(r[0] - trial_onset_ts[i] if len(r) > 0 else 0)
    dur_reward = np.floor(np.array(dur_reward_sec) * channel_clean.samplerate).astype(int)

    # Trim each trial at reward onset, then zero out invalid trials
    neural_data_trimmed_list = []
    for i, trial in enumerate(channel_clean.trials):
        temp = trial.copy()
        if dur_reward[i] != 0:
            temp[min(dur_reward[i], len(trial)):] = 0
        neural_data_trimmed_list.append(temp)
    for i, valid in enumerate(full_mask):
        if not valid:
            neural_data_trimmed_list[i][:] = 0

    neural_data_trimmed = np.concatenate(neural_data_trimmed_list, axis=0)

    # Drop invalid trials (via the regressor-length mask), zero-mean, save
    output_path = os.path.join(results_dir, session_name, f'channel{channel}_regressors')
    os.makedirs(output_path, exist_ok=True)
    full_mask_reg = np.load(os.path.join(output_path, 'full_mask_reg.npz'))['data']
    neural_data_trimmed = neural_data_trimmed[full_mask_reg]
    neural_data_trimmed -= np.mean(neural_data_trimmed)
    np.savez_compressed(os.path.join(output_path, 'neural_data.npz'), data=neural_data_trimmed)

    print(f"[ch{channel}] {int(np.sum(full_mask))} valid trials, "
          f"{neural_data_trimmed.shape[0]} samples -> {output_path}")
    return output_path


# -------------------------
# Driver: parallelise channels across the cluster
# -------------------------
if __name__ == '__main__':
    for session_name in sessions:
        n_workers = min(MAX_WORKERS, len(channels))
        print(f"[{session_name}] {len(channels)} channels -> {n_workers} workers on '{SLURM_PARTITION}'")

        with ParallelMap(process_channel, channels, session_name, lfp_data_dir, log_file_dir, results_dir,
                         n_inputs=len(channels),
                         partition=SLURM_PARTITION,
                         n_workers=n_workers,
                         mem_per_worker=MEM_PER_WORKER,
                         write_worker_results=False,   # workers save their own npz; just return the path
                         setup_interactive=False) as pmap:
            pmap.compute()
