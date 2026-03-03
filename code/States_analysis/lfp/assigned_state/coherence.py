"""
Summary
-------
This script computes trial-wise LFP coherence between electrode arrays
for different cognitive states across multiple recording sessions.

For each session:
- LFP data are segmented into trials and cleaned of NaNs.
- Channels are grouped into 6 anatomical arrays.
- For every array pair, signals are averaged across channels within each array.
- Trial-wise coherence spectra are computed using SciPy's coherence function.

Across sessions:
- Trial-wise coherence is pooled by cognitive state and array pair.
- Nonparametric permutation tests (shuffling trials) are used to compare
  coherence between pairs of cognitive states at each frequency.
- Max/min-based thresholds are applied to control for multiple comparisons
  across frequencies.

Outputs:
- Frequency-resolved coherence differences and significance masks are saved.
- Summary plots show mean coherence per state and significant frequency bands.
"""

# -----------------------------
# Imports
# -----------------------------
import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import coherence
import syncopy as spy
import json

# custom path for parse_logfile
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog  # noqa: E402

# -----------------------------
# User Config
# -----------------------------
lfp_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/clean_full_length'
trial_info_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/full_length'
states_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis'
output_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_lfp/all_trials/200_600/coherence_scipy_trialwise'
results_dir = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp/coherence_scipy_trialwise'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

sessions = ['20230203', '20230208', '20230209', '20230213', '20230214']
N_STATES_TO_USE = 3
fs = 1000  # sampling rate
fmin, fmax = 2, 100
n_perms = 1000
alpha = 0.05
rng = np.random.default_rng(42)

# load states info for all sessions
state_probs = np.load(f'{states_data_dir}/foraging_shivangi_no_sess1_clipped_state_assignments.npy')
with open(f'{states_data_dir}/foraging_shivangi_no_sess1_clipped_session_index.json') as f:
    session_index = json.load(f)

session_to_probs = {}
for sess in session_index:
    session_id = sess['session_id']
    session_date = session_id.split('_')[1]
    session_to_probs[session_date] = state_probs[
        sess['start_idx']: sess['end_idx'] + 1
    ]

# -----------------------------
# Helper functions
# -----------------------------
def remove_nan_trials_channels(datas):
    trial_mask = [not np.all(np.isnan(tr)) for tr in datas.trials]
    if not any(trial_mask):
        return None, []
    cfg = spy.StructDict(trials=np.where(trial_mask)[0])
    datas_clean = spy.selectdata(cfg, datas)
    trial_stack = np.stack(datas_clean.trials, axis=0)
    valid_ch_idx = np.where(~np.all(np.isnan(trial_stack), axis=(0, 1)))[0]
    if len(valid_ch_idx) == 0:
        return None, []
    cfg = spy.StructDict(channel=valid_ch_idx)
    datas_clean = spy.selectdata(cfg, datas_clean)
    valid_channels = [datas.channel[i] for i in valid_ch_idx]
    return datas_clean, valid_channels

def ensure_trialindex_in_trialdefinition(datalfp):
    if datalfp.trialdefinition.shape[1] < 4:
        nTrials = datalfp.trialdefinition.shape[0]
        datalfp.trialdefinition = np.hstack(
            (datalfp.trialdefinition, np.arange(nTrials).reshape(-1, 1))
        )

def compute_trialwise_array_coherence(trials_array1, trials_array2, fs=1000, nperseg=None):
    """
    Compute coherence per trial, then average across trials.
    Returns frequency axis and coherence per trial (n_trials x n_freqs).
    """
    n_trials = trials_array1.shape[0]
    coh_all = []
    f_axis = None
    for t in range(n_trials):
        sig1 = np.nanmean(trials_array1[t, :, :], axis=1)  # mean across channels in array 1
        sig2 = np.nanmean(trials_array2[t, :, :], axis=1)  # mean across channels in array 2
        f, Cxy = coherence(sig1, sig2, fs=fs, nperseg=nperseg)
        freq_mask = (f >= fmin) & (f <= fmax)
        if f_axis is None:
            f_axis = f[freq_mask]
        coh_all.append(Cxy[freq_mask])
    coh_all = np.stack(coh_all, axis=0)  # (n_trials x n_freqs)
    return f_axis, coh_all

def permutation_test(data1, data2, n_perms=1000, alpha=0.05, rng=None):
    """Permutation test for trial-wise coherence (averaged across trials)."""
    n1, n2 = data1.shape[0], data2.shape[0]
    pooled = np.vstack([data1, data2])
    labels = np.array([0]*n1 + [1]*n2)
    real_diff = np.nanmean(data1, axis=0) - np.nanmean(data2, axis=0)

    max_dist = np.zeros(n_perms)
    min_dist = np.zeros(n_perms)
    for i in range(n_perms):
        rng.shuffle(labels)
        perm1 = pooled[labels == 0]
        perm2 = pooled[labels == 1]
        diff = np.nanmean(perm1, axis=0) - np.nanmean(perm2, axis=0)
        max_dist[i] = np.nanmax(diff)
        min_dist[i] = np.nanmin(diff)

    upper_thr = np.percentile(max_dist, 100 * (1 - alpha/2))
    lower_thr = np.percentile(min_dist, 100 * (alpha/2))
    sig_mask = (real_diff > upper_thr) | (real_diff < lower_thr)
    return real_diff, sig_mask, (lower_thr, upper_thr)

# -----------------------------
# Main Analysis
# -----------------------------
state_data_coh = {}

for session_name in sessions:
    print(f"\n=== Processing session {session_name} ===")
    lfp_path = os.path.join(lfp_data_dir, session_name, 'Cleaned_lfp_FT.spy')
    trial_info_path = os.path.join(trial_info_dir, session_name, 'Trial_Info.pkl')
    if not os.path.exists(lfp_path) or not os.path.exists(trial_info_path):
        continue

    # Load state info
    predicted_states = session_to_probs[session_name]
    trial_info_df = pd.read_pickle(trial_info_path)
    trial_info_df.iloc[:, 0] = (trial_info_df.iloc[:, 0] - 1000).astype('Int64')
    stim_df = pd.DataFrame({'TrialIndex': np.arange(len(predicted_states)), 'States': predicted_states})
    combined_df = pd.merge(trial_info_df, stim_df, left_on='Trial_Number', right_on='TrialIndex', how='inner')

    # Load LFP
    datalfp = spy.load(lfp_path)
    ensure_trialindex_in_trialdefinition(datalfp)
    cfg = spy.StructDict(latency=[-0.2, 0.6])
    data = spy.selectdata(cfg, datalfp)
    selected_trials = data.trialdefinition[:, 3].astype(int)
    states_trial_info_filt = combined_df[combined_df['TrialIndex'].isin(selected_trials)]
    unique_states = np.sort(np.unique(states_trial_info_filt['States'].to_numpy()))[:N_STATES_TO_USE]

    for state_value in unique_states:
        cfg_sel = spy.StructDict(trials=np.where(states_trial_info_filt['States'] == state_value)[0])
        datas_state = spy.selectdata(cfg_sel, data)
        datas_state_clean, valid_channels = remove_nan_trials_channels(datas_state)
        if datas_state_clean is None:
            continue

        # Split channels into arrays
        Sig_CH = np.array_split(valid_channels, 6)

        # Store coherence for this state/session
        if state_value not in state_data_coh:
            state_data_coh[state_value] = []

        for i in range(len(Sig_CH)):
            for j in range(i+1, len(Sig_CH)):
                ch_idx1 = [valid_channels.index(c) for c in Sig_CH[i] if c in valid_channels]
                ch_idx2 = [valid_channels.index(c) for c in Sig_CH[j] if c in valid_channels]
                if not ch_idx1 or not ch_idx2:
                    continue

                trials_array1 = np.stack([t[:, ch_idx1] for t in datas_state_clean.trials], axis=0)
                trials_array2 = np.stack([t[:, ch_idx2] for t in datas_state_clean.trials], axis=0)

                f, coh_trials = compute_trialwise_array_coherence(trials_array1, trials_array2, fs=fs)
                state_data_coh[state_value].append({
                    'arrays': (i+1, j+1),
                    'coh_trials': coh_trials,  # (n_trials x n_freqs)
                    'freqs': f
                })

# -----------------------------
# Permutation Test
# -----------------------------
print("\n=== Running permutation tests across states (array-level) ===")
pairs = list(itertools.combinations(sorted(state_data_coh.keys()), 2))

for (s1, s2) in pairs:
    print(f"--> Testing pair ({s1} vs {s2})")
    arrays_unique = list({entry['arrays'] for entry in state_data_coh[s1]})

    for arr_pair in arrays_unique:
        vals1, vals2 = [], []
        f_axis = None
        for entry in state_data_coh[s1]:
            if entry['arrays'] == arr_pair:
                vals1.append(entry['coh_trials'])
                f_axis = entry['freqs']
        for entry in state_data_coh[s2]:
            if entry['arrays'] == arr_pair:
                vals2.append(entry['coh_trials'])
        if not vals1 or not vals2:
            continue
        data1 = np.concatenate(vals1, axis=0)
        data2 = np.concatenate(vals2, axis=0)

        diff, sig, thr = permutation_test(data1, data2, n_perms=n_perms, alpha=alpha, rng=rng)

        # save results
        npz_name = f"permdata_trialwise_coh_pair{s1}_{s2}_array{arr_pair[0]}-{arr_pair[1]}.npz"
        npz_path = os.path.join(results_dir, npz_name)
        np.savez_compressed(npz_path,
                            diff=diff,
                            sig=sig,
                            thr=thr,
                            mean1=np.nanmean(data1, axis=0),
                            mean2=np.nanmean(data2, axis=0),
                            x_axis=f_axis,
                            s1=s1,
                            s2=s2,
                            arrays=arr_pair)

        # Plot
        plt.figure(figsize=(6, 4))
        plt.plot(f_axis, np.nanmean(data1, axis=0), label=f"State {s1}")
        plt.plot(f_axis, np.nanmean(data2, axis=0), label=f"State {s2}")
        plt.fill_between(f_axis, np.nanmean(data1, axis=0)-np.nanstd(data1, axis=0),
                         np.nanmean(data1, axis=0)+np.nanstd(data1, axis=0), alpha=0.2)
        plt.fill_between(f_axis, np.nanmean(data2, axis=0)-np.nanstd(data2, axis=0),
                         np.nanmean(data2, axis=0)+np.nanstd(data2, axis=0), alpha=0.2)
        plt.fill_between(f_axis, np.nanmean(data1, axis=0) - np.nanmean(data2, axis=0),
                         where=sig, color='orange', alpha=0.3)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Coherence")
        plt.title(f"Array {arr_pair[0]}-{arr_pair[1]}: State {s1} vs {s2}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"trialwise_coh_pair{s1}_{s2}_array{arr_pair[0]}-{arr_pair[1]}.pdf"))
        plt.close()

print(f"Trial-wise coherence results saved under {output_dir}")
