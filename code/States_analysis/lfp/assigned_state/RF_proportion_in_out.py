"""
Compute the average proportion of timepoints where each RF (channel) is
within vs outside the stimulus, for two time windows:

1. Stim-onset aligned  – timepoints from stim onset to 0.6 s after onset
2. RT-aligned           – timepoints within [RT − 0.45 s, RT + 0.45 s]

Uses the pre-computed RF overlap data in RF_stim_collapse.h5 and the log
files for stimulus timing.  RT values come from emissions.npy.

Output
------
For each session a dict saved as .npz with keys:
    stim_prop_A, stim_prop_B   – (nChannels,) proportion inside A/B (stim window)
    rt_prop_A,   rt_prop_B     – (nChannels,) proportion inside A/B (RT window)
    channel_ids                 – (nChannels,) channel/point labels
Also prints a cross-session summary table.
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py

sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog  # noqa: E402

# ---- paths / config --------------------------------------------------------
eye_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data'
rf_stim_dir  = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/RFarea_stim'
processed_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis/processed'
output_dir   = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp/RF_proportions'
os.makedirs(output_dir, exist_ok=True)

# sessions that have RF_stim_collapse.h5
sessions = ['20230203', '20230208', '20230209', '20230214']

session_logfiles = {
    '20230203': '2023_02_03-11_35_57_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230208': '2023_02_08-10_58_17_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230209': '2023_02_09-11_19_51_Cosmos_LeafForaging_001_KAS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230214': '2023_02_14-11_42_27_Cosmos_LeafForaging_001_PAF_GrassyLandscapeWithBackgroundDark_Cont.log',
}

session_folders = {
    '20230203': 'Cosmos_20230203_LeafForaging_001',
    '20230208': 'Cosmos_20230208_LeafForaging_001',
    '20230209': 'Cosmos_20230209_LeafForaging_001',
    '20230214': 'Cosmos_20230214_LeafForaging_001',
}

n_stimuli = 5
stim_window_end = 0.6  # seconds after stim onset for stim-aligned window
pre_rt  = 0.45         # seconds before RT
post_rt = 0.45         # seconds after RT

# ---- helpers ----------------------------------------------------------------

def get_aligned_stim_times(logfile_path):
    """Parse log file and return list of aligned stimulus times per trial."""
    stim_ts = []
    stim_name = 'ImageStimulus'

    with TextLog(logfile_path) as log:
        log.parse_all_state_times(state='StartTrial', times='StateStarted')
        log.parse_all_state_times(state='EndTrial', times='StateStarted')
        evt, ts, evt_desc, true_ts = log.parse_eventmarkers()
        indx = [ii for ii, name in enumerate(log.all_ids['name'])
                if name.startswith(stim_name)]

        for ii, istim in enumerate(indx):
            if ii + n_stimuli == len(indx):
                break
            this_id = log.all_ids[istim]
            next_id = log.all_ids[indx[ii + n_stimuli]]
            _, pos_ts = log.parse_spherical(
                obj_id=this_id['id'],
                st=this_id['start'],
                end=next_id['start'])
            stim_ts.append(pos_ts)

    # number of trials from the parsed stimuli
    n_trials = len(stim_ts) // n_stimuli

    aligned_list = []
    for itrl in range(n_trials):
        stim_times = stim_ts[itrl * n_stimuli + 3].T  # target stimulus (index 3)
        aligned = stim_times - stim_times[0]           # time 0 = stim onset
        aligned_list.append(aligned)

    return aligned_list


def compute_proportions_for_session(session):
    """
    For a single session, compute per-channel proportion of timepoints
    inside stimulus A/B for the two windows (full stim, RT-centred).
    """
    print(f"\n=== Session {session} ===")

    # 1. Aligned stimulus times
    logfile = os.path.join(eye_data_dir, session, session_logfiles[session])
    aligned_stim_times = get_aligned_stim_times(logfile)
    n_trials_log = len(aligned_stim_times)
    print(f"  Log trials: {n_trials_log}")

    # 2. RT values
    emissions_path = os.path.join(processed_dir, session_folders[session], 'emissions.npy')
    rt_values = np.load(emissions_path).flatten()
    print(f"  RT values loaded: {len(rt_values)}")

    # 3. Open RF overlap HDF5
    h5_path = os.path.join(rf_stim_dir, session, 'RF_stim_collapse.h5')

    # accumulators: channel_id → [total_tp, inside_A, inside_B] for each window
    stim_counts = {}   # stim-onset aligned window
    rt_counts   = {}   # RT-aligned window

    with h5py.File(h5_path, 'r') as f:
        trial_names = sorted(f.keys(), key=lambda x: int(x.split('_')[-1]))
        n_trials_h5 = len(trial_names)

        for trial_idx, trial_name in enumerate(trial_names):
            if trial_idx >= n_trials_log:
                break

            # progress every 100 trials
            if trial_idx % 100 == 0:
                print(f"  Session {session} | trial {trial_idx}/{n_trials_h5}")

            trial_group = f[trial_name]
            trial_times = aligned_stim_times[trial_idx]
            rt = rt_values[trial_idx] if trial_idx < len(rt_values) else np.nan

            # pre-compute which timepoint indices fall in each window
            # so we skip irrelevant timepoints without touching HDF5
            tp_names = sorted(trial_group.keys(),
                              key=lambda x: int(x.split('_')[-1]))
            n_tp = min(len(tp_names), len(trial_times))

            stim_tp_set = set()
            rt_tp_set = set()
            for tp_idx in range(n_tp):
                t = trial_times[tp_idx]
                if np.isnan(t):
                    continue
                if 0 <= t <= stim_window_end:
                    stim_tp_set.add(tp_idx)
                if not np.isnan(rt) and (rt - pre_rt) <= t <= (rt + post_rt):
                    rt_tp_set.add(tp_idx)

            relevant_tps = stim_tp_set | rt_tp_set
            if not relevant_tps:
                continue

            # only read HDF5 groups for relevant timepoints
            for tp_idx in relevant_tps:
                tp_group = trial_group[tp_names[tp_idx]]
                in_stim = tp_idx in stim_tp_set
                in_rt   = tp_idx in rt_tp_set

                for point_name in tp_group.keys():
                    point_group = tp_group[point_name]
                    inside_A = bool(point_group['inside_transformed_outline_A'][()])
                    inside_B = bool(point_group['inside_transformed_outline_B'][()])

                    # Map Point index to neural data channel name.
                    # point_name is "Point_{idx+1}_xx_yy" where idx+1 is the
                    # 1-indexed recording channel number (center_coords has all
                    # 192 channels; NaN channels are skipped in the HDF5).
                    ch_num = int(point_name.split('_')[1])
                    ch_id = f"channel_{ch_num:03d}"

                    if in_stim:
                        if ch_id not in stim_counts:
                            stim_counts[ch_id] = [0, 0, 0]  # total, insA, insB
                        stim_counts[ch_id][0] += 1
                        stim_counts[ch_id][1] += int(inside_A)
                        stim_counts[ch_id][2] += int(inside_B)

                    if in_rt:
                        if ch_id not in rt_counts:
                            rt_counts[ch_id] = [0, 0, 0]
                        rt_counts[ch_id][0] += 1
                        rt_counts[ch_id][1] += int(inside_A)
                        rt_counts[ch_id][2] += int(inside_B)

        print(f"  Session {session} | done ({n_trials_h5} trials)")

    # compute proportions
    channel_ids = sorted(stim_counts.keys(), key=lambda x: int(x.split('_')[-1]))
    n_ch = len(channel_ids)
    stim_prop_A = np.zeros(n_ch)
    stim_prop_B = np.zeros(n_ch)
    rt_prop_A   = np.full(n_ch, np.nan)
    rt_prop_B   = np.full(n_ch, np.nan)

    for i, ch in enumerate(channel_ids):
        total, insA, insB = stim_counts[ch]
        stim_prop_A[i] = insA / total if total > 0 else 0.0
        stim_prop_B[i] = insB / total if total > 0 else 0.0

        if ch in rt_counts:
            total_rt, insA_rt, insB_rt = rt_counts[ch]
            if total_rt > 0:
                rt_prop_A[i] = insA_rt / total_rt
                rt_prop_B[i] = insB_rt / total_rt

    print(f"  Channels: {n_ch}")
    print(f"  Stim-onset: mean prop inside A = {np.nanmean(stim_prop_A):.3f}, "
          f"B = {np.nanmean(stim_prop_B):.3f}")
    print(f"  RT-aligned: mean prop inside A = {np.nanmean(rt_prop_A):.3f}, "
          f"B = {np.nanmean(rt_prop_B):.3f}")

    return {
        'channel_ids': np.array(channel_ids),
        'stim_prop_A': stim_prop_A,
        'stim_prop_B': stim_prop_B,
        'rt_prop_A': rt_prop_A,
        'rt_prop_B': rt_prop_B,
    }


# ---- main ------------------------------------------------------------------
all_results = {}
for sess in sessions:
    result = compute_proportions_for_session(sess)
    all_results[sess] = result

    # save per-session
    np.savez(os.path.join(output_dir, f'RF_proportion_{sess}.npz'), **result)

# ---- cross-session summary -------------------------------------------------
print("\n" + "=" * 70)
print("Cross-session summary: average proportion of timepoints inside RF")
print("=" * 70)
rows = []
for sess in sessions:
    r = all_results[sess]
    rows.append({
        'session': sess,
        'stim_mean_A': np.nanmean(r['stim_prop_A']),
        'stim_mean_B': np.nanmean(r['stim_prop_B']),
        'rt_mean_A':   np.nanmean(r['rt_prop_A']),
        'rt_mean_B':   np.nanmean(r['rt_prop_B']),
        'n_channels':  len(r['channel_ids']),
    })
summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))

# overall average across sessions
print(f"\nOverall stim-onset:  A = {summary_df['stim_mean_A'].mean():.3f}, "
      f"B = {summary_df['stim_mean_B'].mean():.3f}")
print(f"Overall RT-aligned:  A = {summary_df['rt_mean_A'].mean():.3f}, "
      f"B = {summary_df['rt_mean_B'].mean():.3f}")
