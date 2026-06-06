"""
Summary
-------
Reward-onset-centered version of erp_spectra_stats.py.

This script performs the same trial-wise, nonparametric statistical comparisons
of time-locked ERPs, power spectra, and aperiodic-adjusted (FOOOF) residual
power between cognitive states - but with LFP segments centered on each trial's
reward-delivery time, looking back in time before the reward.

Reward time per trial is parsed from the Unreal log file:
  * stim onset  = event 3011
  * trial end   = event 3090
  * reward      = first event in [5000..5999] between stim onset and trial end
The reward time stored per trial is `reward_ts - stim_onset_ts` (seconds from
stim onset), matching the convention used in RF_inout_channel_raster.py.

For each session:
- LFP data are loaded (stimulus-aligned, full length).
- Reward times are parsed from the log file.
- For each trial, a fixed window around reward onset is extracted:
    [reward - pre_rew, reward + post_rew]  (time 0 = reward onset).
- Trials and channels containing only NaNs are removed.
- Trials are grouped by cognitive state.
- Three signal representations are computed:
    * Time-domain ERPs (reward-locked LFP trials),
    * Power spectra (2 to 100 Hz, using periodogram, keeping trials),
    * FOOOF-derived residual spectra computed from mean power per channel.

Across sessions:
- Trial-wise data are pooled by cognitive state.
- Pairwise permutation tests (shuffling trials) are performed between states:
    * at the single-channel level,
    * at the array level (channels grouped into 6 arrays),
    * and at a combined-array level (arrays 1 to 3 merged, others separate).
- Max/min-based permutation thresholds are used to control for multiple
  comparisons across time or frequency.
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
from scipy.signal import periodogram
from fooof import FOOOF
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
processed_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis/processed'
eye_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data'

# ---- Reward-walkthrough cluster switch ----
# The 5xxx reward marker lands either ~0 ms or ~334 ms after the stimulus
# walkthrough (3013/3023). This switch restricts the analysis to one cluster so
# you can check whether a pre-reward oscillation moves with the marker offset.
#   'all'       -> no filtering (original behavior)
#   'near0'     -> only trials with |reward - walkthrough| <= cluster_tol
#   'around334' -> only trials with reward - walkthrough within cluster_tol of 0.334 s
reward_walk_cluster = 'around334'
cluster_tol = 0.05          # seconds, half-width of each cluster window
cluster_center_334 = 0.334  # seconds, center of the late cluster
cluster_tag = 'all' if reward_walk_cluster == 'all' else f'cluster_{reward_walk_cluster}'

# Roots for plots and results. The cluster folder is inserted directly under
# reward_aligned/ so that all outputs for one cluster (all_trials +
# correct_trials) live together
results_dir = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp'
reward_aligned_plots_base = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_lfp/reward_aligned'
reward_aligned_plots_root = os.path.join(reward_aligned_plots_base, cluster_tag)
reward_aligned_results_root = os.path.join(results_dir, 'reward_aligned', cluster_tag)

output_dir = os.path.join(reward_aligned_plots_root, 'all_trials', 'erp_spectra')
results_data_dir = os.path.join(reward_aligned_results_root, 'all_trials', 'erp_spectra')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_data_dir, exist_ok=True)
colors = [(0.55, 0.0, 0.55), (0.0, 0.39, 0.39), (0.8, 0.33, 0.0)]

sessions = ['20230203', '20230208', '20230209', '20230213', '20230214']
session_folders = {
    '20230203': 'Cosmos_20230203_LeafForaging_001',
    '20230208': 'Cosmos_20230208_LeafForaging_001',
    '20230209': 'Cosmos_20230209_LeafForaging_001',
    '20230213': 'Cosmos_20230213_LeafForaging_002',
    '20230214': 'Cosmos_20230214_LeafForaging_001',
}
session_logfiles = {
    '20230203': '2023_02_03-11_35_57_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230208': '2023_02_08-10_58_17_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230209': '2023_02_09-11_19_51_Cosmos_LeafForaging_001_KAS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230213': '2023_02_13-11_13_43_Cosmos_LeafForaging_002_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230214': '2023_02_14-11_42_27_Cosmos_LeafForaging_001_PAF_GrassyLandscapeWithBackgroundDark_Cont.log',
}

N_STATES_TO_USE = 4
n_perms = 1000
alpha = 0.05
rng = np.random.default_rng(42)

# ---- Diagnostic toggle ----
# When True: run the reward - walkthrough (3013/3023) histogram analysis AND
# the per-group eventmarker filtering.
run_diagnostic = False

# When True: only include correct trials (ResponseCorrect, no block-exit).
correct_only = False

# Groups of trials to compare by their reward - walkthrough diff (seconds).
# Each entry: 'group_label': (low, high) inclusive-low, exclusive-high.
# Use this to see which eventmarkers are associated with each diff bucket.
diagnostic_diff_groups = {
    'near_zero':      (-0.05, 0.05),
    'around_0.334s':  (0.30, 0.40),
}
# Event codes to exclude from per-trial eventmarker listings, summary table,
# and bar charts.
diagnostic_exclude_codes = {996}

# Reward-centering parameters
# Window is centered on reward onset; pre_rew is the look-back duration
# (going back in time from reward), post_rew is the look-forward duration.
pre_rew = 1.5 # seconds before reward onset
post_rew = 0.0  #seconds after reward onset

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
def ensure_trialindex_in_trialdefinition(datalfp):
    if datalfp.trialdefinition.shape[1] < 4:
        nTrials = datalfp.trialdefinition.shape[0]
        datalfp.trialdefinition = np.hstack(
            (datalfp.trialdefinition, np.arange(nTrials).reshape(-1, 1))
        )


def get_reward_times_from_log(session_name):
    """
    Parse the Unreal log for a session and return reward time per log trial
    (seconds from stim onset). Returns NaN for trials with no reward event
    in [stim_onset, trial_end].
    """
    log_path = os.path.join(eye_data_dir, session_name, session_logfiles[session_name])
    with TextLog(log_path) as log:
        log.make_id_struct()
        evt, ts, _, _ = log.parse_eventmarkers()

    target_onset = ts[np.where(evt == 3011)[0]]
    trial_end_ts = ts[np.where(evt == 3090)[0]]
    reward_mask = (evt >= 5000) & (evt <= 5999)
    reward_ts_all = ts[reward_mask]

    n_log_trials = len(target_onset)
    rew_t_per_trial = np.full(n_log_trials, np.nan)
    for trl in range(n_log_trials):
        t0 = target_onset[trl]
        t1 = trial_end_ts[trl] if trl < len(trial_end_ts) else t0 + 5.0
        in_window = (reward_ts_all >= t0) & (reward_ts_all <= t1)
        if in_window.any():
            first_idx = np.where(in_window)[0][0]
            rew_t_per_trial[trl] = reward_ts_all[first_idx] - t0
    return rew_t_per_trial


def get_reward_walkthrough_diffs_from_log(session_name):
    """
    Parse the Unreal log and, for each trial, compute the time difference
    between the reward event (5000-5999) and the stimulus walkthrough event
    (3013 = stim A, 3023 = stim B), restricted to [stim_onset, trial_end].

    Returns
    -------
    diffs : np.ndarray, shape (n_log_trials,)
        reward_ts - walkthrough_ts (seconds). NaN if either event missing.
    which_stim : np.ndarray of object, shape (n_log_trials,)
        'A' (3013), 'B' (3023), or '' if no walkthrough event found.
    trial_indices : np.ndarray, shape (n_log_trials,)
        Log-trial index per entry (0..n_log_trials-1).
    """
    log_path = os.path.join(eye_data_dir, session_name, session_logfiles[session_name])
    with TextLog(log_path) as log:
        log.make_id_struct()
        evt, ts, _, _ = log.parse_eventmarkers()

    target_onset = ts[np.where(evt == 3011)[0]]
    trial_end_ts = ts[np.where(evt == 3090)[0]]

    walk_mask = (evt == 3013) | (evt == 3023)
    walk_ts_all = ts[walk_mask]
    walk_evt_all = evt[walk_mask]

    reward_mask = (evt >= 5000) & (evt <= 5999)
    reward_ts_all = ts[reward_mask]

    n_log_trials = len(target_onset)
    diffs = np.full(n_log_trials, np.nan)
    which_stim = np.array([''] * n_log_trials, dtype=object)
    trial_indices = np.arange(n_log_trials)

    for trl in range(n_log_trials):
        t0 = target_onset[trl]
        t1 = trial_end_ts[trl] if trl < len(trial_end_ts) else t0 + 5.0

        w_in = (walk_ts_all >= t0) & (walk_ts_all <= t1)
        r_in = (reward_ts_all >= t0) & (reward_ts_all <= t1)
        if not w_in.any() or not r_in.any():
            continue

        w_first = np.where(w_in)[0][0]
        r_first = np.where(r_in)[0][0]
        diffs[trl] = reward_ts_all[r_first] - walk_ts_all[w_first]
        which_stim[trl] = 'A' if walk_evt_all[w_first] == 3013 else 'B'

    return diffs, which_stim, trial_indices


def get_reward_walk_cluster_mask(session_name, cluster):
    """
    Per-log-trial boolean selecting trials whose reward(5xxx) - walkthrough
    (3013/3023) offset falls in the requested cluster.

    cluster:
      'all'       -> all trials (no filtering; returns all True)
      'near0'     -> |diff| <= cluster_tol
      'around334' -> |diff - cluster_center_334| <= cluster_tol

    Trials with no walkthrough or no reward event (NaN diff) are excluded from
    'near0'/'around334' (they cannot be assigned to a cluster).
    """
    diffs, _, _ = get_reward_walkthrough_diffs_from_log(session_name)
    if cluster == 'all':
        return np.ones(len(diffs), dtype=bool)
    valid = ~np.isnan(diffs)
    if cluster == 'near0':
        return valid & (np.abs(diffs) <= cluster_tol)
    elif cluster == 'around334':
        return valid & (np.abs(diffs - cluster_center_334) <= cluster_tol)
    else:
        raise ValueError(f"Unknown reward_walk_cluster: {cluster!r}")


def get_correct_trial_mask_from_log(session_name):
    """
    Per-log-trial boolean: True if event 1 (ResponseCorrect) occurs in the
    trial's [stim_onset, trial_end] window AND the trial is not a block-exit
    trial (event 3091). Matches the correct_mask convention used in
    RF_inout_channel_raster.py.
    """
    log_path = os.path.join(eye_data_dir, session_name, session_logfiles[session_name])
    with TextLog(log_path) as log:
        log.make_id_struct()
        evt, ts, _, _ = log.parse_eventmarkers()
    target_onset = ts[np.where(evt == 3011)[0]]
    trial_end_ts = ts[np.where(evt == 3090)[0]]
    response_correct_ts = ts[np.where(evt == 1)[0]]
    block_exit_ts = ts[np.where(evt == 3091)[0]]
    n_log_trials = len(target_onset)

    # Block-exit per trial: each 3091 maps to the trial whose stim onset most
    # recently preceded it (same logic as RF_inout_channel_raster.py).
    has_block_exit = np.zeros(n_log_trials, dtype=bool)
    for t_exit in block_exit_ts:
        idx = np.searchsorted(target_onset, t_exit, side='right') - 1
        if 0 <= idx < n_log_trials:
            has_block_exit[idx] = True

    mask = np.zeros(n_log_trials, dtype=bool)
    for trl in range(n_log_trials):
        if has_block_exit[trl]:
            continue
        t0 = target_onset[trl]
        t1 = trial_end_ts[trl] if trl < len(trial_end_ts) else t0 + 5.0
        if np.any((response_correct_ts >= t0) & (response_correct_ts <= t1)):
            mask[trl] = True
    return mask


def _periodogram_hann(signal, fs, freq_range):
    """Single-taper (Hann) periodogram for a 1-D signal."""
    freqs, pxx = periodogram(signal, fs=fs, window='hann')
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    return freqs[mask], pxx[mask]


def _periodogram_multitaper(signal, fs, freq_range, tapsmofrq=4):
    """
    Multi-taper periodogram using DPSS tapers, matching syncopy's
    mtmfft with tapsmofrq (spectral smoothing bandwidth in Hz).
    """
    from scipy.signal.windows import dpss
    nsamples = len(signal)
    T = nsamples / fs
    NW = tapsmofrq * T / 2
    K = max(int(2 * NW - 1), 1)
    tapers = dpss(nsamples, NW, Kmax=K)

    freqs = np.fft.rfftfreq(nsamples, d=1.0 / fs)
    mask = (freqs > freq_range[0]) & (freqs <= freq_range[1])

    pxx_sum = np.zeros(mask.sum())
    for taper in tapers:
        windowed = signal * taper
        fft_vals = np.fft.rfft(windowed)
        pxx = (np.abs(fft_vals) ** 2) / (fs * np.sum(taper ** 2))
        pxx_sum += pxx[mask]
    pxx_avg = pxx_sum / K

    return freqs[mask], pxx_avg


def compute_spectrum_trials(trials_3d, fs):
    """
    Compute power spectrum for each trial and channel:
      - 2-30 Hz:  single Hann taper
      - 30-100 Hz: DPSS multitaper with 4 Hz smoothing
    Concatenated along the frequency axis.
    """
    ntrials, nsamples, nchan = trials_3d.shape

    dummy = np.zeros(nsamples)
    freqs_low, _ = _periodogram_hann(dummy, fs, (2, 30))
    freqs_high, _ = _periodogram_multitaper(dummy, fs, (30, 100), tapsmofrq=4)
    freqs_combined = np.concatenate((freqs_low, freqs_high))

    power = np.full((ntrials, len(freqs_combined), nchan), np.nan)
    for t in range(ntrials):
        for c in range(nchan):
            if np.all(np.isnan(trials_3d[t, :, c])):
                continue
            sig = trials_3d[t, :, c]
            _, pxx_low = _periodogram_hann(sig, fs, (2, 30))
            _, pxx_high = _periodogram_multitaper(sig, fs, (30, 100), tapsmofrq=4)
            power[t, :, c] = np.concatenate((pxx_low, pxx_high))
    return power, freqs_combined


def permutation_test(data1, data2, n_perms=1000, alpha=0.05, rng=None):
    """
    data1, data2: arrays of shape (nTrials, nTime/Freq)
    Returns: real_diff, sig_mask (bool array same shape as real_diff)
    """
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
# DIAGNOSTIC BLOCK
# (histogram of reward - walkthrough diffs + per-diff-group eventmarker listing)
# Toggle with `run_diagnostic`. The main analysis below runs regardless.
# -----------------------------
if run_diagnostic:
    print("\n=== Building reward - walkthrough (3013/3023) time-difference histogram ===")

    hist_output_dir = os.path.join(reward_aligned_plots_base, 'reward_walkthrough_diff_hist')
    os.makedirs(hist_output_dir, exist_ok=True)

    per_session_diffs = {}     # session_name -> dict with diffs, which_stim, trial_indices
    all_diffs_combined = []
    all_stim_combined = []
    all_session_labels = []

    for session_name in sessions:
        log_path = os.path.join(eye_data_dir, session_name, session_logfiles[session_name])
        if not os.path.exists(log_path):
            print(f"  Log file not found for {session_name}, skipping")
            continue

        diffs, which_stim, trial_indices = get_reward_walkthrough_diffs_from_log(session_name)
        per_session_diffs[session_name] = {
            'diffs': diffs,
            'which_stim': which_stim,
            'trial_indices': trial_indices,
        }
        valid = ~np.isnan(diffs)
        n_valid = int(valid.sum())
        n_total = len(diffs)
        if n_valid > 0:
            print(f"  {session_name}: {n_valid}/{n_total} trials with both events, "
                  f"min={np.nanmin(diffs):.3f}s, max={np.nanmax(diffs):.3f}s, "
                  f"median={np.nanmedian(diffs):.3f}s")
        else:
            print(f"  {session_name}: no trials with both walkthrough and reward events")

        all_diffs_combined.append(diffs[valid])
        all_stim_combined.append(which_stim[valid])
        all_session_labels.append(np.array([session_name] * n_valid))

        # Save per-session CSV so the user can inspect / decide which trials to skip
        df_out = pd.DataFrame({
            'TrialIndex': trial_indices,
            'WhichStim': which_stim,
            'RewardMinusWalkthrough_s': diffs,
        })
        df_out.to_csv(
            os.path.join(hist_output_dir, f"reward_walkthrough_diffs_{session_name}.csv"),
            index=False,
        )

    if all_diffs_combined:
        all_diffs_combined = np.concatenate(all_diffs_combined)
        all_stim_combined = np.concatenate(all_stim_combined)
        all_session_labels = np.concatenate(all_session_labels)
    else:
        all_diffs_combined = np.array([])
        all_stim_combined = np.array([], dtype=object)
        all_session_labels = np.array([], dtype=object)

    # Per-session histograms (one subplot per session)
    n_sess_with_data = len(per_session_diffs)
    if n_sess_with_data > 0:
        ncols = min(3, n_sess_with_data)
        nrows = int(np.ceil(n_sess_with_data / ncols))
        fig_ps, axes_ps = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                                       sharex=True)
        axes_ps = np.atleast_1d(axes_ps).flatten()

        if all_diffs_combined.size > 0:
            bin_edges = np.linspace(np.nanmin(all_diffs_combined),
                                    np.nanmax(all_diffs_combined), 40)
        else:
            bin_edges = np.linspace(0, 1, 40)

        for i, (sess_name, d) in enumerate(per_session_diffs.items()):
            ax = axes_ps[i]
            diffs = d['diffs']
            which_stim = d['which_stim']
            diffs_A = diffs[(which_stim == 'A') & ~np.isnan(diffs)]
            diffs_B = diffs[(which_stim == 'B') & ~np.isnan(diffs)]
            ax.hist(diffs_A, bins=bin_edges, alpha=0.6, label=f'A (n={len(diffs_A)})',
                    color=colors[0])
            ax.hist(diffs_B, bins=bin_edges, alpha=0.6, label=f'B (n={len(diffs_B)})',
                    color=colors[1])
            if np.any(~np.isnan(diffs)):
                ax.axvline(np.nanmedian(diffs), color='k', ls='--', lw=1,
                           label=f'median={np.nanmedian(diffs):.3f}s')
            ax.set_title(sess_name)
            ax.set_xlabel('reward - walkthrough (s)')
            ax.set_ylabel('# trials')
            ax.legend(fontsize=7)

        for j in range(len(per_session_diffs), len(axes_ps)):
            axes_ps[j].set_visible(False)

        fig_ps.suptitle('Reward - walkthrough (3013/3023) time difference per session',
                        fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig_ps.savefig(os.path.join(hist_output_dir,
                                    'reward_walkthrough_diff_hist_per_session.pdf'))
        plt.close(fig_ps)

    # Combined histogram across all sessions
    if all_diffs_combined.size > 0:
        fig_c, ax_c = plt.subplots(figsize=(7, 4.5))
        bin_edges = np.linspace(np.nanmin(all_diffs_combined),
                                np.nanmax(all_diffs_combined), 60)
        diffs_A = all_diffs_combined[all_stim_combined == 'A']
        diffs_B = all_diffs_combined[all_stim_combined == 'B']
        ax_c.hist(diffs_A, bins=bin_edges, alpha=0.6,
                  label=f'A (n={len(diffs_A)})', color=colors[0])
        ax_c.hist(diffs_B, bins=bin_edges, alpha=0.6,
                  label=f'B (n={len(diffs_B)})', color=colors[1])
        ax_c.axvline(np.nanmedian(all_diffs_combined), color='k', ls='--', lw=1,
                     label=f'median={np.nanmedian(all_diffs_combined):.3f}s')
        ax_c.set_xlabel('reward - walkthrough (s)')
        ax_c.set_ylabel('# trials')
        ax_c.set_title('Reward - walkthrough (3013/3023) time difference - all sessions')
        ax_c.legend()
        plt.tight_layout()
        fig_c.savefig(os.path.join(hist_output_dir,
                                   'reward_walkthrough_diff_hist_combined.pdf'))
        plt.close(fig_c)

        # Combined CSV for quick outlier inspection
        pd.DataFrame({
            'Session': all_session_labels,
            'WhichStim': all_stim_combined,
            'RewardMinusWalkthrough_s': all_diffs_combined,
        }).to_csv(os.path.join(hist_output_dir,
                               'reward_walkthrough_diffs_all_sessions.csv'),
                  index=False)

    print(f"Histograms and CSVs saved under {hist_output_dir}")

    # ----- Per-diff-group eventmarker filtering -----
    # For each (label, (low, high)) in diagnostic_diff_groups, list every
    # eventmarker (code + description) for the trials that fall in that diff
    # range. Then summarize which event codes appear in each group so you can
    # compare (e.g. 0 s trials vs 0.334 s trials).
    # Trial window = [this trial's 3011, next trial's 3011) so we don't
    # truncate at 3090.
    from collections import Counter

    print("\n=== Per-diff-group eventmarker listing ===")
    group_rows = []                # for the long CSV (one row per trial per group)
    group_code_counts = {}         # group_label -> Counter of (evt_code, desc)

    for group_label, (lo, hi) in diagnostic_diff_groups.items():
        print(f"\n  --- Group '{group_label}': {lo} <= diff < {hi} ---")
        group_code_counts[group_label] = Counter()

        for session_name, d in per_session_diffs.items():
            diffs = d['diffs']
            which_stim = d['which_stim']
            trial_indices = d['trial_indices']

            log_path = os.path.join(eye_data_dir, session_name, session_logfiles[session_name])
            with TextLog(log_path) as log:
                log.make_id_struct()
                evt, ts, evt_desc, _ = log.parse_eventmarkers()
            evt_desc = np.array(evt_desc, dtype=object)
            target_onset = ts[np.where(evt == 3011)[0]]

            mask = ~np.isnan(diffs) & (diffs >= lo) & (diffs < hi)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            print(f"    {session_name}: {len(idx)} trials in group")
            for trl in idx:
                t0 = target_onset[trl]
                t1 = target_onset[trl + 1] if trl + 1 < len(target_onset) else ts[-1] + 1e-6
                in_window = (ts >= t0) & (ts < t1)
                evts_in_trial = evt[in_window]
                ts_in_trial = ts[in_window] - t0
                desc_in_trial = evt_desc[in_window]

                # Drop excluded codes (e.g. 996 PhotodiodeUpdate)
                keep = ~np.isin(evts_in_trial, list(diagnostic_exclude_codes))
                evts_in_trial = evts_in_trial[keep]
                ts_in_trial = ts_in_trial[keep]
                desc_in_trial = desc_in_trial[keep]

                print(f"      trial {int(trial_indices[trl])} "
                      f"(stim {which_stim[trl]}): diff={diffs[trl]:.3f}s, "
                      f"n_events={len(evts_in_trial)}")
                for e, t_rel, dsc in zip(evts_in_trial, ts_in_trial, desc_in_trial):
                    print(f"          t={t_rel:7.3f}s  evt={int(e):5d}  desc={dsc}")
                    group_code_counts[group_label][(int(e), dsc)] += 1

                events_str = ';'.join(
                    f"{int(e)}({dsc})@{t_rel:.3f}s"
                    for e, t_rel, dsc in zip(evts_in_trial, ts_in_trial, desc_in_trial)
                )
                group_rows.append({
                    'Group': group_label,
                    'Session': session_name,
                    'TrialIndex': int(trial_indices[trl]),
                    'WhichStim': which_stim[trl],
                    'RewardMinusWalkthrough_s': float(diffs[trl]),
                    'N_Events': int(len(evts_in_trial)),
                    'Events': events_str,
                })

    # ----- Group comparison summary -----
    print("\n=== Eventmarker frequency per group ===")
    all_codes = set()
    for c in group_code_counts.values():
        all_codes.update(c.keys())
    if all_codes:
        # Build a comparison table: rows = (code, desc), cols = group counts
        summary_rows = []
        for code, dsc in sorted(all_codes):
            row = {'EventCode': code, 'Description': dsc}
            for g in diagnostic_diff_groups:
                row[g] = group_code_counts[g].get((code, dsc), 0)
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)

        # Print to console
        with pd.option_context('display.max_rows', None,
                               'display.max_colwidth', 60):
            print(summary_df.to_string(index=False))

        summary_df.to_csv(
            os.path.join(hist_output_dir,
                         'diagnostic_group_event_summary.csv'),
            index=False,
        )

        # Events that appear in ONE group but not another
        group_names = list(diagnostic_diff_groups.keys())
        if len(group_names) >= 2:
            print("\n  Events unique to each group (vs other groups):")
            for g in group_names:
                this_codes = set(group_code_counts[g].keys())
                others = set().union(
                    *[set(group_code_counts[g2].keys())
                      for g2 in group_names if g2 != g]
                )
                unique = sorted(this_codes - others)
                print(f"    '{g}' only: {unique}")

    if group_rows:
        pd.DataFrame(group_rows).to_csv(
            os.path.join(hist_output_dir,
                         'diagnostic_group_trials_eventmarkers.csv'),
            index=False,
        )
        print(f"\n  Per-trial eventmarker CSV saved to "
              f"{os.path.join(hist_output_dir, 'diagnostic_group_trials_eventmarkers.csv')}")
    else:
        print("  No trials matched any diagnostic group.")

    # ----- Histogram of event-code counts per diagnostic group -----
    if all_codes:
        group_names = list(diagnostic_diff_groups.keys())
        n_groups = len(group_names)
        # Sort by total count across groups (descending) and keep the top N
        # for readability.
        MAX_CODES_TO_PLOT = 40
        codes_sorted = sorted(
            all_codes,
            key=lambda cd: -sum(group_code_counts[g].get(cd, 0) for g in group_names),
        )[:MAX_CODES_TO_PLOT]
        if not codes_sorted:
            print("  No event codes to plot - skipping bar plots.")
        else:
            labels = [f"{c}\n{(d or '')[:18]}" for c, d in codes_sorted]
            x = np.arange(len(codes_sorted))
            bar_w = 0.8 / max(n_groups, 1)

            # (1) Raw counts: grouped bar chart, codes on x-axis
            fig_g, ax_g = plt.subplots(figsize=(max(8, 0.45 * len(codes_sorted)), 5))
            for i, g in enumerate(group_names):
                counts = [group_code_counts[g].get(cd, 0) for cd in codes_sorted]
                n_trials_g = sum(1 for r in group_rows if r['Group'] == g)
                ax_g.bar(x + (i - (n_groups - 1) / 2) * bar_w, counts, bar_w,
                         label=f"{g} (n_trials={n_trials_g})",
                         color=colors[i % len(colors)])
            ax_g.set_xticks(x)
            ax_g.set_xticklabels(labels, rotation=75, ha='right', fontsize=7)
            ax_g.set_ylabel('Event count (summed across trials in group)')
            ax_g.set_title('Eventmarker frequency per diagnostic group')
            ax_g.legend()
            plt.tight_layout()
            fig_g.savefig(os.path.join(hist_output_dir,
                                       'diagnostic_group_event_counts.pdf'))
            plt.close(fig_g)

            # (2) Per-trial rate (count / n_trials_in_group): fair comparison when
            # groups have different numbers of trials.
            n_trials_per_group = {g: max(sum(1 for r in group_rows if r['Group'] == g), 1)
                                  for g in group_names}
            fig_r, ax_r = plt.subplots(figsize=(max(8, 0.45 * len(codes_sorted)), 5))
            for i, g in enumerate(group_names):
                rates = [group_code_counts[g].get(cd, 0) / n_trials_per_group[g]
                         for cd in codes_sorted]
                ax_r.bar(x + (i - (n_groups - 1) / 2) * bar_w, rates, bar_w,
                         label=f"{g} (n_trials={n_trials_per_group[g]})",
                         color=colors[i % len(colors)])
            ax_r.set_xticks(x)
            ax_r.set_xticklabels(labels, rotation=75, ha='right', fontsize=7)
            ax_r.set_ylabel('Events per trial')
            ax_r.set_title('Eventmarker rate per trial - diagnostic groups')
            ax_r.legend()
            plt.tight_layout()
            fig_r.savefig(os.path.join(hist_output_dir,
                                       'diagnostic_group_event_rate.pdf'))
            plt.close(fig_r)

            print(f"  Diagnostic histograms saved to "
                  f"{os.path.join(hist_output_dir, 'diagnostic_group_event_counts.pdf')} "
                  f"and diagnostic_group_event_rate.pdf")

# -----------------------------
# Main data collection
# -----------------------------
state_data_timelock = {}
state_data_spectra = {}
state_data_residuals = {}

total_trials_kept = 0
total_trials_discarded = 0

for session_name in sessions:
    print(f"\n=== Processing session {session_name} ===")
    lfp_path = os.path.join(lfp_data_dir, session_name, 'Cleaned_lfp_FT.spy')
    trial_info_path = os.path.join(trial_info_dir, session_name, 'Trial_Info.pkl')
    log_path = os.path.join(eye_data_dir, session_name, session_logfiles[session_name])

    if not os.path.exists(lfp_path) or not os.path.exists(trial_info_path):
        print(f"  LFP or trial info missing for {session_name}, skipping")
        continue
    if not os.path.exists(log_path):
        print(f"  Log file not found for {session_name}, skipping")
        continue

    # parse reward times from log (seconds from stim onset, one per log trial)
    reward_times = get_reward_times_from_log(session_name)
    print(f"  Log reward events parsed: {(~np.isnan(reward_times)).sum()}/{len(reward_times)}")

    # state info
    predicted_states = session_to_probs[session_name]
    trial_info_df = pd.read_pickle(trial_info_path)
    trial_info_df.iloc[:, 0] = (trial_info_df.iloc[:, 0] - 1000).astype('Int64')
    n_states_avail = len(predicted_states)
    n_rew_avail = min(n_states_avail, len(reward_times))
    stim_df = pd.DataFrame({
        'TrialIndex': np.arange(n_states_avail),
        'States': predicted_states,
        'RewardTime': np.concatenate([
            reward_times[:n_rew_avail],
            np.full(n_states_avail - n_rew_avail, np.nan)
        ]) if n_rew_avail < n_states_avail else reward_times[:n_states_avail]
    })
    combined_df = pd.merge(trial_info_df, stim_df,
                           left_on='Trial_Number', right_on='TrialIndex', how='inner')

    # Restrict to one reward-walkthrough offset cluster if requested
    if reward_walk_cluster != 'all':
        cluster_indices = np.where(
            get_reward_walk_cluster_mask(session_name, reward_walk_cluster))[0]
        n_before_cluster = len(combined_df)
        combined_df = combined_df[combined_df['TrialIndex'].isin(cluster_indices)]
        print(f"  Reward-walk cluster '{reward_walk_cluster}': "
              f"{len(combined_df)}/{n_before_cluster} trials retained")

    # load LFP data (full length, stimulus-aligned)
    datalfp = spy.load(lfp_path)
    ensure_trialindex_in_trialdefinition(datalfp)
    fs = datalfp.samplerate
    all_channels = list(datalfp.channel)

    lfp_trial_indices = datalfp.trialdefinition[:, 3].astype(int)
    states_trial_info_filt = combined_df[combined_df['TrialIndex'].isin(lfp_trial_indices)]
    unique_states = np.sort(np.unique(states_trial_info_filt['States'].to_numpy()))[:N_STATES_TO_USE]

    for state_value in unique_states:
        state_trials = states_trial_info_filt[states_trial_info_filt['States'] == state_value]

        rew_centered_trials = []
        for _, row in state_trials.iterrows():
            trial_idx = row['TrialIndex']
            rew = row['RewardTime']

            # skip trials with no reward event
            if rew is None or (isinstance(rew, float) and np.isnan(rew)):
                continue

            lfp_trial_pos = np.where(lfp_trial_indices == trial_idx)[0]
            if len(lfp_trial_pos) == 0:
                continue
            lfp_trial_pos = lfp_trial_pos[0]

            trial_data = datalfp.trials[lfp_trial_pos]   # (nSamples, nChannels)
            trial_time = datalfp.time[lfp_trial_pos]      # (nSamples,)

            if np.all(np.isnan(trial_data)):
                continue

            # Extract samples within [reward - pre_rew, reward + post_rew]
            t_start = rew - pre_rew
            t_end = rew + post_rew
            print(f"    trial {trial_idx}: rew={rew:.3f}, "
                  f"trial_time range=[{trial_time[0]:.3f}, {trial_time[-1]:.3f}], "
                  f"post-rew available={trial_time[-1] - rew:.3f}s, "
                  f"pre-rew available={rew - trial_time[0]:.3f}s")
            time_mask = (trial_time >= t_start) & (trial_time <= t_end)

            if np.sum(time_mask) < 10:
                continue

            segment = trial_data[time_mask, :]  # (nSamples_seg, nChannels)
            rew_centered_trials.append(segment)

        if not rew_centered_trials:
            continue

        # Only keep trials with the full window
        expected_len = int(np.round((pre_rew + post_rew) * fs))
        n_before = len(rew_centered_trials)
        rew_centered_trials = [seg[:expected_len, :] for seg in rew_centered_trials
                               if seg.shape[0] >= expected_len]
        n_after = len(rew_centered_trials)
        n_discarded = n_before - n_after
        total_trials_kept += n_after
        total_trials_discarded += n_discarded
        print(f"  State {state_value}: kept {n_after}/{n_before} trials "
              f"(discarded {n_discarded} with <{expected_len} samples for "
              f"{pre_rew}+{post_rew}s window)")

        if not rew_centered_trials:
            continue

        trials_array = np.stack(rew_centered_trials, axis=0)

        # Reward-centered time vector (time 0 = reward onset)
        time_vec = np.linspace(-pre_rew, post_rew, expected_len)

        valid_ch_mask = ~np.all(np.isnan(trials_array), axis=(0, 1))
        valid_ch_idx = np.where(valid_ch_mask)[0]
        if len(valid_ch_idx) == 0:
            continue
        trials_array = trials_array[:, :, valid_ch_idx]
        valid_channels = [all_channels[i] for i in valid_ch_idx]

        trial_mask = ~np.all(np.isnan(trials_array), axis=(1, 2))
        trials_array = trials_array[trial_mask]
        if trials_array.shape[0] == 0:
            continue

        print(f"  State {state_value}: {trials_array.shape[0]} trials, "
              f"{len(valid_channels)} channels, {expected_len} samples")

        # Spectra
        power_trials, freqs_combined = compute_spectrum_trials(trials_array, fs)

        # FOOOF on mean spectrum per channel
        mean_spec = np.nanmean(power_trials, axis=0)
        resid_session = np.full_like(mean_spec, np.nan)
        freq_res = np.median(np.diff(freqs_combined))
        for ch_i, ch_name in enumerate(valid_channels):
            try:
                lower_pw = max(2 * freq_res, 1.0)
                upper_pw = 12
                if lower_pw >= upper_pw:
                    print(f"  FOOOF skipped {session_name}, ch {ch_name}: "
                          f"freq resolution too coarse ({freq_res:.2f} Hz)")
                    continue
                fm = FOOOF(peak_width_limits=[lower_pw, upper_pw],
                           max_n_peaks=6,
                           min_peak_height=0.05,
                           peak_threshold=1.5,
                           aperiodic_mode='fixed')
                fm.fit(freqs_combined, mean_spec[:, ch_i])
                resid_session[:, ch_i] = fm._spectrum_flat
            except Exception as e:
                print(f"  FOOOF failed {session_name}, ch {ch_name}: {e}")

        # store
        for dct, data_in, xaxis in [(state_data_timelock, trials_array, time_vec),
                                    (state_data_spectra, power_trials, freqs_combined)]:
            if state_value not in dct:
                dct[state_value] = []
            dct[state_value].append({'trials': data_in, 'time': xaxis, 'channels': valid_channels})

        if state_value not in state_data_residuals:
            state_data_residuals[state_value] = []
        state_data_residuals[state_value].append({'resid': resid_session,
                                                  'freqs': freqs_combined,
                                                  'channels': valid_channels})


print(f"\n=== Trial selection summary ===")
print(f"  Total trials kept:      {total_trials_kept}")
print(f"  Total trials discarded: {total_trials_discarded}")
denom = total_trials_kept + total_trials_discarded
if denom > 0:
    print(f"  Fraction kept:          {total_trials_kept / denom:.1%}")

# -----------------------------
# Permutation tests (pairwise)
# -----------------------------
print("\n=== Running permutation tests across states ===")
perm_results_cache = {}
pairs = list(itertools.combinations(sorted(state_data_timelock.keys()), 2))

for plot_type, store in [('timelock', state_data_timelock),
                         ('spectra', state_data_spectra),
                         ('residual', state_data_residuals)]:
    if not store:
        continue

    first_channels = store[next(iter(store))][0]['channels']
    Sig_CH = np.array_split(first_channels, 6)

    for (s1, s2) in pairs:
        print(f"--> Testing pair ({s1} vs {s2}) for {plot_type}")
        for i_arr, ch_names in enumerate(Sig_CH):
            # Per-channel plots
            fig, axes = plt.subplots(6, 6, figsize=(15, 12))
            axes = axes.flatten()
            for ichan, ch_name in enumerate(ch_names):
                ax = axes[ichan]
                vals1, vals2 = [], []
                x_axis = None
                for sess in store[s1]:
                    if ch_name not in sess['channels']:
                        continue
                    ch_idx = sess['channels'].index(ch_name)
                    if plot_type in ['timelock', 'spectra']:
                        x_axis = sess['time']
                    elif plot_type == 'residual':
                        x_axis = sess['freqs']
                    if plot_type == 'residual':
                        vals1.append(sess['resid'][:, ch_idx])
                    else:
                        vals1.append(sess['trials'][:, :, ch_idx])
                for sess in store[s2]:
                    if ch_name not in sess['channels']:
                        continue
                    ch_idx = sess['channels'].index(ch_name)
                    if plot_type in ['timelock', 'spectra']:
                        x_axis = sess['time']
                    elif plot_type == 'residual':
                        x_axis = sess['freqs']
                    if plot_type == 'residual':
                        vals2.append(sess['resid'][:, ch_idx])
                    else:
                        vals2.append(sess['trials'][:, :, ch_idx])

                if not vals1 or not vals2:
                    continue
                if plot_type == 'residual':
                    data1 = np.stack(vals1, axis=0)
                    data2 = np.stack(vals2, axis=0)
                else:
                    min_samples = min(v.shape[1] for v in vals1 + vals2)
                    vals1 = [v[:, :min_samples] for v in vals1]
                    vals2 = [v[:, :min_samples] for v in vals2]
                    x_axis = x_axis[:min_samples]
                    data1 = np.concatenate(vals1, axis=0)
                    data2 = np.concatenate(vals2, axis=0)

                diff, sig, thr = permutation_test(data1, data2, n_perms=n_perms, alpha=alpha, rng=rng)

                npz_name = f"permdata_{plot_type}_pair{s1}_{s2}_array{i_arr+1}_{ch_name}.npz"
                npz_path = os.path.join(results_data_dir, npz_name)
                if not os.path.exists(npz_path):
                    mean1 = np.nanmean(data1, axis=0)
                    mean2 = np.nanmean(data2, axis=0)
                    np.savez_compressed(
                        npz_path,
                        diff=diff,
                        sig=sig,
                        thr=thr,
                        mean1=mean1,
                        mean2=mean2,
                        x_axis=x_axis,
                        s1=s1,
                        s2=s2,
                        plot_type=plot_type,
                        ch_name=ch_name,
                        array_index=i_arr + 1
                    )

                ax.plot(x_axis, diff, color='k')
                ax.fill_between(x_axis, diff, where=sig, color='red', alpha=0.4)
                ax.axhline(0, color='gray', lw=0.5)
                ax.set_title(ch_name, fontsize=7)
                if plot_type == 'timelock':
                    ax.set_ylim(-15, 15)
            for j in range(len(ch_names), 36):
                axes[j].set_visible(False)
            fig.suptitle(f"{plot_type} {s1} vs {s2} - Array {i_arr+1} (reward-centered)")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fname = os.path.join(output_dir, f"perm_{plot_type}_pair{s1}_{s2}_array{i_arr+1}.pdf")
            fig.savefig(fname)
            plt.close(fig)

            # Array-level combined analysis
            print(f"  --> Running array-level stats for Array {i_arr+1} ({plot_type})")
            vals1_array, vals2_array = [], []
            x_axis = None
            for sess in store[s1]:
                ch_valid = [c for c in ch_names if c in sess['channels']]
                if not ch_valid:
                    continue
                ch_idx = [sess['channels'].index(c) for c in ch_valid]
                if plot_type in ['timelock', 'spectra']:
                    x_axis = sess['time']
                elif plot_type == 'residual':
                    x_axis = sess['freqs']
                if plot_type == 'residual':
                    vals1_array.append(np.mean(sess['resid'][:, ch_idx], axis=1))
                else:
                    vals1_array.append(np.mean(np.mean(sess['trials'][:, :, ch_idx], axis=0), axis=1))
            for sess in store[s2]:
                ch_valid = [c for c in ch_names if c in sess['channels']]
                if not ch_valid:
                    continue
                ch_idx = [sess['channels'].index(c) for c in ch_valid]
                if plot_type in ['timelock', 'spectra']:
                    x_axis = sess['time']
                elif plot_type == 'residual':
                    x_axis = sess['freqs']
                if plot_type == 'residual':
                    vals2_array.append(np.mean(sess['resid'][:, ch_idx], axis=1))
                else:
                    vals2_array.append(np.mean(np.mean(sess['trials'][:, :, ch_idx], axis=0), axis=1))
            if vals1_array and vals2_array:
                min_samples = min(v.shape[0] for v in vals1_array + vals2_array)
                vals1_array = [v[:min_samples] for v in vals1_array]
                vals2_array = [v[:min_samples] for v in vals2_array]
                x_axis = x_axis[:min_samples]
                data1_array = np.stack(vals1_array, axis=0)
                data2_array = np.stack(vals2_array, axis=0)
                diff_array, sig_array, thr_array = permutation_test(
                    data1_array, data2_array, n_perms=n_perms, alpha=alpha, rng=rng)
                fig_arr, ax_arr = plt.subplots(figsize=(6, 4))
                ax_arr.plot(x_axis, diff_array, color='k', lw=1.5)
                ax_arr.fill_between(x_axis, diff_array, where=sig_array, color='red', alpha=0.4)
                ax_arr.axhline(0, color='gray', lw=0.8)
                ax_arr.set_title(f"Array {i_arr+1} ({plot_type}) {s1} vs {s2} (reward-centered)")
                ax_arr.set_xlabel('Time rel. reward (s)' if plot_type == 'timelock' else 'Frequency (Hz)')
                ax_arr.set_ylabel('ΔAmplitude' if plot_type == 'timelock' else 'ΔResidual Power')
                if plot_type == 'timelock':
                    ax_arr.set_ylim(-15, 15)
                plt.tight_layout()
                fname_arr = os.path.join(
                    output_dir, f"perm_{plot_type}_pair{s1}_{s2}_ARRAYCOMBINED_array{i_arr+1}.pdf")
                fig_arr.savefig(fname_arr)
                plt.close(fig_arr)

                npz_array_name = f"permdata_{plot_type}_pair{s1}_{s2}_ARRAY_array{i_arr+1}.npz"
                npz_array_path = os.path.join(results_data_dir, npz_array_name)
                if not os.path.exists(npz_array_path):
                    mean1_array = np.nanmean(data1_array, axis=0)
                    mean2_array = np.nanmean(data2_array, axis=0)
                    np.savez_compressed(
                        npz_array_path,
                        diff=diff_array,
                        sig=sig_array,
                        thr=thr_array,
                        mean1=mean1_array,
                        mean2=mean2_array,
                        x_axis=x_axis,
                        s1=s1,
                        s2=s2,
                        plot_type=plot_type,
                        array_index=i_arr + 1
                    )

# -----------------------------
# array-level, grouped (arrays 1-3 merged)
# -----------------------------
print("\n=== Running permutation tests across states (array-level grouping) ===")
perm_results_cache = {}
pairs = list(itertools.combinations(sorted(state_data_timelock.keys()), 2))

for plot_type, store in [('timelock', state_data_timelock),
                         ('spectra', state_data_spectra),
                         ('residual', state_data_residuals)]:
    if not store:
        continue

    first_channels = store[next(iter(store))][0]['channels']
    Sig_CH = np.array_split(first_channels, 6)

    for (s1, s2) in pairs:
        print(f"--> Testing pair ({s1} vs {s2}) for {plot_type}")
        for i_arr, ch_names in enumerate(Sig_CH):
            if i_arr < 3:
                if i_arr == 0:
                    combined_ch_names = np.concatenate(Sig_CH[:3])
                else:
                    continue
            else:
                combined_ch_names = ch_names

            vals1_array, vals2_array = [], []
            x_axis = None

            for sess in store[s1]:
                ch_valid = [c for c in combined_ch_names if c in sess['channels']]
                if not ch_valid:
                    continue
                ch_idx = [sess['channels'].index(c) for c in ch_valid]
                if plot_type in ['timelock', 'spectra']:
                    x_axis = sess['time']
                elif plot_type == 'residual':
                    x_axis = sess['freqs']
                if plot_type == 'residual':
                    vals1_array.append(np.mean(sess['resid'][:, ch_idx], axis=1))
                else:
                    vals1_array.append(np.mean(np.mean(sess['trials'][:, :, ch_idx], axis=0), axis=1))

            for sess in store[s2]:
                ch_valid = [c for c in combined_ch_names if c in sess['channels']]
                if not ch_valid:
                    continue
                ch_idx = [sess['channels'].index(c) for c in ch_valid]
                if plot_type in ['timelock', 'spectra']:
                    x_axis = sess['time']
                elif plot_type == 'residual':
                    x_axis = sess['freqs']
                if plot_type == 'residual':
                    vals2_array.append(np.mean(sess['resid'][:, ch_idx], axis=1))
                else:
                    vals2_array.append(np.mean(np.mean(sess['trials'][:, :, ch_idx], axis=0), axis=1))

            if vals1_array and vals2_array:
                min_samples = min(v.shape[0] for v in vals1_array + vals2_array)
                vals1_array = [v[:min_samples] for v in vals1_array]
                vals2_array = [v[:min_samples] for v in vals2_array]
                x_axis = x_axis[:min_samples]
                data1_array = np.stack(vals1_array, axis=0)
                data2_array = np.stack(vals2_array, axis=0)
                diff_array, sig_array, thr_array = permutation_test(
                    data1_array, data2_array, n_perms=n_perms, alpha=alpha, rng=rng)

                fig_arr, ax_arr = plt.subplots(figsize=(6, 4))
                ax_arr.plot(x_axis, diff_array, color='k', lw=1.5)
                ax_arr.fill_between(x_axis, diff_array, where=sig_array, color='red', alpha=0.4)
                ax_arr.axhline(0, color='gray', lw=0.8)
                array_label = f"Array {i_arr+1}" if i_arr >= 3 else "Array 1-3"
                ax_arr.set_title(f"{array_label} ({plot_type}) {s1} vs {s2} (reward-centered)")
                ax_arr.set_xlabel('Time rel. reward (s)' if plot_type == 'timelock' else 'Frequency (Hz)')
                ax_arr.set_ylabel('ΔAmplitude' if plot_type == 'timelock' else 'ΔResidual Power')
                if plot_type == 'timelock':
                    ax_arr.set_ylim(-15, 15)
                plt.tight_layout()
                fname_arr = os.path.join(
                    output_dir, f"cb_perm_{plot_type}_pair{s1}_{s2}_{array_label.replace('-', '')}.pdf")
                fig_arr.savefig(fname_arr)
                plt.close(fig_arr)

                npz_array_name = f"cb_permdata_{plot_type}_pair{s1}_{s2}_{array_label.replace('-', '')}.npz"
                npz_array_path = os.path.join(results_data_dir, npz_array_name)
                if not os.path.exists(npz_array_path):
                    mean1_array = np.nanmean(data1_array, axis=0)
                    mean2_array = np.nanmean(data2_array, axis=0)
                    np.savez_compressed(
                        npz_array_path,
                        diff=diff_array,
                        sig=sig_array,
                        thr=thr_array,
                        mean1=mean1_array,
                        mean2=mean2_array,
                        x_axis=x_axis,
                        s1=s1,
                        s2=s2,
                        plot_type=plot_type,
                        array_index=i_arr + 1
                    )

print(f"Permutation-test results saved under {output_dir}")

# =================
# Summary Figures
# =================
from matplotlib.colors import LinearSegmentedColormap

summary_output_dir = os.path.join(output_dir, 'summary_plots')
os.makedirs(summary_output_dir, exist_ok=True)

plot_types_fig = ['timelock', 'residual']
states = [s for s in range(N_STATES_TO_USE) if s != 1]
arrays_list = [1, 2, 3, 4, 5, 6]
state_colors = {
    0: (0.55, 0.0, 0.55),
    1: (0.0, 0.39, 0.39),
    2: (0.8, 0.33, 0.0),
    3: (0.25, 0.35, 0.55)
}
sig_color = '#8dd3c7'
teal_cmap = LinearSegmentedColormap.from_list('teal_map', ['white', '#1f9e89'])


def load_permdata(plot_type, s1, s2, array_index):
    fname = f"permdata_{plot_type}_pair{s1}_{s2}_ARRAY_array{array_index}.npz"
    fpath = os.path.join(results_data_dir, fname)
    if not os.path.exists(fpath):
        return None
    return np.load(fpath)


def load_permdata_merged(plot_type, s1, s2, array_label):
    fname = f"cb_permdata_{plot_type}_pair{s1}_{s2}_{array_label.replace('-', '')}.npz"
    fpath = os.path.join(results_data_dir, fname)
    if not os.path.exists(fpath):
        return None
    return np.load(fpath)


# Figure 1: Real data + significance masks (per array)
pairs_fig = list(itertools.combinations(states, 2))

for plot_type in plot_types_fig:
    ylabel = 'Amplitude' if plot_type == 'timelock' else 'Residual Power'
    n_rows, n_cols = len(arrays_list), len(pairs_fig)

    fig_real, axes_real = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3*n_rows),
                                       sharex='col', sharey='row')
    if n_rows == 1: axes_real = np.expand_dims(axes_real, 0)
    if n_cols == 1: axes_real = np.expand_dims(axes_real, 1)

    summary_mask = []

    for i_array, array_index in enumerate(arrays_list):
        summary_mask_array = []
        for j_pair, (s1, s2) in enumerate(pairs_fig):
            ax = axes_real[i_array, j_pair]
            perm_array = load_permdata(plot_type, s1, s2, array_index)

            if perm_array is None:
                ax.axis('off')
                summary_mask_array.append(None)
                continue

            x_axis = perm_array['x_axis']
            data1, data2 = perm_array['mean1'], perm_array['mean2']
            sig_mask = perm_array['sig']

            ax.plot(x_axis, data1, color=state_colors[s1], label=f"State {s1}")
            ax.plot(x_axis, data2, color=state_colors[s2], label=f"State {s2}")
            if plot_type == 'timelock':
                ax.set_ylim(-15, 15)
            ax.fill_between(x_axis, ax.get_ylim()[0], ax.get_ylim()[1], where=sig_mask,
                            color=sig_color, alpha=0.4)

            if i_array == 0: ax.set_title(f"{s1} vs {s2}", fontsize=10)
            if j_pair == 0: ax.set_ylabel(f"Array {array_index}\n{ylabel}")
            if i_array == n_rows - 1:
                xlabel = 'Time rel. reward (s)' if plot_type == 'timelock' else 'Frequency (Hz)'
                ax.set_xlabel(xlabel)
            ax.legend(fontsize=6)

            summary_mask_array.append(sig_mask)
        summary_mask.append(summary_mask_array)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.25, hspace=0.25)
    plt.suptitle(f"{plot_type} - Mean across all sessions (reward-centered)", fontsize=14)
    plt.savefig(os.path.join(summary_output_dir, f"{plot_type}_all_arrays_pairs.pdf"))
    plt.close()

    # Figure 2: Summary heatmap
    if x_axis is not None:
        n_arrays_fig = len(arrays_list)
        n_pairs_fig = len(pairs_fig)
        n_time = len(x_axis)
        summary_array = np.zeros((n_arrays_fig, n_pairs_fig, n_time), dtype=int)

        for i_array in range(n_arrays_fig):
            for j_pair in range(n_pairs_fig):
                mask = summary_mask[i_array][j_pair]
                if mask is not None and mask.shape[0] == n_time:
                    summary_array[i_array, j_pair, :] = mask.astype(int)

        fig_sum, axes_sum = plt.subplots(1, n_pairs_fig, figsize=(6*n_pairs_fig, 4), sharey=True)
        if n_pairs_fig == 1: axes_sum = [axes_sum]

        for j_pair, (s1, s2) in enumerate(pairs_fig):
            ax = axes_sum[j_pair]
            im = ax.imshow(summary_array[:, j_pair, :], cmap=teal_cmap, aspect='auto',
                           interpolation='none',
                           extent=[x_axis[0], x_axis[-1], 0.5, n_arrays_fig + 0.5])
            ax.set_title(f"{s1} vs {s2}")
            ax.set_xlabel('Time rel. reward (s)' if plot_type == 'timelock' else 'Frequency (Hz)')
            if j_pair == 0:
                ax.set_ylabel('Arrays')
                ax.set_yticks(range(1, n_arrays_fig + 1))
                ax.set_yticklabels([str(a) for a in arrays_list])

        plt.subplots_adjust(left=0.05, right=0.88, top=0.88, bottom=0.12, wspace=0.3)
        cbar_ax = fig_sum.add_axes([0.90, 0.12, 0.02, 0.76])
        cbar = fig_sum.colorbar(im, cax=cbar_ax)
        cbar.set_label('Significant (1=pairwise)')

        plt.suptitle(f"{plot_type} - Pairwise significance across arrays (reward-centered)", fontsize=14)
        plt.savefig(os.path.join(summary_output_dir, f"{plot_type}_pairwise_summary_all_arrays.pdf"))
        plt.close()

# Figure 3: Merged Array plots (arrays 1-3 combined)
array_labels = ['Array 1-3', 'Array 4', 'Array 5', 'Array 6']

for plot_type in plot_types_fig:
    ylabel = 'Amplitude' if plot_type == 'timelock' else 'Residual Power'
    n_rows, n_cols = len(pairs_fig), len(array_labels)

    fig_real, axes_real = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3*n_rows),
                                       sharex='col', sharey='row')
    if n_rows == 1: axes_real = np.expand_dims(axes_real, 0)
    if n_cols == 1: axes_real = np.expand_dims(axes_real, 1)

    for i_pair, (s1, s2) in enumerate(pairs_fig):
        for j_array, array_label in enumerate(array_labels):
            ax = axes_real[i_pair, j_array]
            perm_array = load_permdata_merged(plot_type, s1, s2, array_label)

            if perm_array is None:
                ax.axis('off')
                continue

            x_axis = perm_array['x_axis']
            data1, data2 = perm_array['mean1'], perm_array['mean2']
            sig_mask = perm_array['sig']

            ax.plot(x_axis, data1, color=state_colors[s1], label=f"State {s1}")
            ax.plot(x_axis, data2, color=state_colors[s2], label=f"State {s2}")
            if plot_type == 'timelock':
                ax.set_ylim(-15, 15)
            ax.fill_between(x_axis, ax.get_ylim()[0], ax.get_ylim()[1], where=sig_mask,
                            color=sig_color, alpha=0.4)

            if i_pair == 0: ax.set_title(f"{array_label}", fontsize=10)
            if j_array == 0: ax.set_ylabel(f"{s1} vs {s2}\n{ylabel}")
            if i_pair == n_rows - 1:
                xlabel = 'Time rel. reward (s)' if plot_type == 'timelock' else 'Frequency (Hz)'
                ax.set_xlabel(xlabel)
            ax.legend(fontsize=6)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.25, hspace=0.25)
    plt.suptitle(f"{plot_type} - Mean across all sessions, merged arrays (reward-centered)", fontsize=14)
    plt.savefig(os.path.join(summary_output_dir, f"{plot_type}_merged_arrays_pairs.pdf"))
    plt.close()

print(f"Summary plots saved under {summary_output_dir}")

# =============================================================================
# Correct-trials-only array-level analysis
# =============================================================================
# Re-runs data collection filtered to trials with ResponseCorrect (event 1)
# in their [stim_onset, trial_end] window, then runs only the array-level
# permutation comparisons (per-array, plus merged Array 1-3 + Arrays 4/5/6).
# Saves plots under .../reward_aligned/correct_trials/. The all-trials
# analysis above is unaffected.

def run_correct_trials_analysis():
    print("\n###### Building correct-trials-only stores for array-level analysis ######")

    correct_output_dir = os.path.join(reward_aligned_plots_root, 'correct_trials')
    correct_results_dir = os.path.join(reward_aligned_results_root, 'correct_trials')
    os.makedirs(correct_output_dir, exist_ok=True)
    os.makedirs(correct_results_dir, exist_ok=True)
    
    state_data_timelock_corr = {}
    state_data_spectra_corr = {}
    state_data_residuals_corr = {}
    
    for session_name in sessions:
        print(f"\n=== [correct] Processing session {session_name} ===")
        lfp_path = os.path.join(lfp_data_dir, session_name, 'Cleaned_lfp_FT.spy')
        trial_info_path = os.path.join(trial_info_dir, session_name, 'Trial_Info.pkl')
        log_path = os.path.join(eye_data_dir, session_name, session_logfiles[session_name])
    
        if not (os.path.exists(lfp_path) and os.path.exists(trial_info_path)
                and os.path.exists(log_path)):
            print(f"  Missing files for {session_name}, skipping")
            continue
    
        reward_times = get_reward_times_from_log(session_name)
        correct_mask_log = get_correct_trial_mask_from_log(session_name)
        print(f"  Log: {correct_mask_log.sum()} correct trials "
              f"(of {len(correct_mask_log)} log trials)")
    
        predicted_states = session_to_probs[session_name]
        trial_info_df = pd.read_pickle(trial_info_path)
        trial_info_df.iloc[:, 0] = (trial_info_df.iloc[:, 0] - 1000).astype('Int64')
        n_states_avail = len(predicted_states)
        n_rew_avail = min(n_states_avail, len(reward_times))
        stim_df = pd.DataFrame({
            'TrialIndex': np.arange(n_states_avail),
            'States': predicted_states,
            'RewardTime': np.concatenate([
                reward_times[:n_rew_avail],
                np.full(n_states_avail - n_rew_avail, np.nan)
            ]) if n_rew_avail < n_states_avail else reward_times[:n_states_avail]
        })
        combined_df = pd.merge(trial_info_df, stim_df,
                               left_on='Trial_Number', right_on='TrialIndex', how='inner')
    
        # Keep only correct trials
        correct_log_indices = np.where(correct_mask_log)[0]
        combined_df = combined_df[combined_df['TrialIndex'].isin(correct_log_indices)]
        print(f"  Correct trials after state/info merge: {len(combined_df)}")

        # Restrict to one reward-walkthrough offset cluster if requested
        if reward_walk_cluster != 'all':
            cluster_indices = np.where(
                get_reward_walk_cluster_mask(session_name, reward_walk_cluster))[0]
            n_before_cluster = len(combined_df)
            combined_df = combined_df[combined_df['TrialIndex'].isin(cluster_indices)]
            print(f"  [correct] Reward-walk cluster '{reward_walk_cluster}': "
                  f"{len(combined_df)}/{n_before_cluster} trials retained")
    
        datalfp = spy.load(lfp_path)
        ensure_trialindex_in_trialdefinition(datalfp)
        fs = datalfp.samplerate
        all_channels = list(datalfp.channel)
    
        lfp_trial_indices = datalfp.trialdefinition[:, 3].astype(int)
        states_trial_info_filt = combined_df[combined_df['TrialIndex'].isin(lfp_trial_indices)]
        unique_states = np.sort(np.unique(
            states_trial_info_filt['States'].to_numpy()))[:N_STATES_TO_USE]
    
        for state_value in unique_states:
            state_trials = states_trial_info_filt[
                states_trial_info_filt['States'] == state_value]
    
            rew_centered_trials = []
            for _, row in state_trials.iterrows():
                trial_idx = row['TrialIndex']
                rew = row['RewardTime']
                if rew is None or (isinstance(rew, float) and np.isnan(rew)):
                    continue
                lfp_trial_pos = np.where(lfp_trial_indices == trial_idx)[0]
                if len(lfp_trial_pos) == 0:
                    continue
                lfp_trial_pos = lfp_trial_pos[0]
                trial_data = datalfp.trials[lfp_trial_pos]
                trial_time = datalfp.time[lfp_trial_pos]
                if np.all(np.isnan(trial_data)):
                    continue
                t_start = rew - pre_rew
                t_end = rew + post_rew
                time_mask = (trial_time >= t_start) & (trial_time <= t_end)
                if np.sum(time_mask) < 10:
                    continue
                rew_centered_trials.append(trial_data[time_mask, :])
    
            if not rew_centered_trials:
                continue
            expected_len = int(np.round((pre_rew + post_rew) * fs))
            rew_centered_trials = [seg[:expected_len, :] for seg in rew_centered_trials
                                   if seg.shape[0] >= expected_len]
            if not rew_centered_trials:
                continue
            trials_array = np.stack(rew_centered_trials, axis=0)
            time_vec = np.linspace(-pre_rew, post_rew, expected_len)
    
            valid_ch_mask = ~np.all(np.isnan(trials_array), axis=(0, 1))
            valid_ch_idx = np.where(valid_ch_mask)[0]
            if len(valid_ch_idx) == 0:
                continue
            trials_array = trials_array[:, :, valid_ch_idx]
            valid_channels = [all_channels[i] for i in valid_ch_idx]
    
            trial_mask = ~np.all(np.isnan(trials_array), axis=(1, 2))
            trials_array = trials_array[trial_mask]
            if trials_array.shape[0] == 0:
                continue
    
            print(f"  [correct] State {state_value}: {trials_array.shape[0]} trials, "
                  f"{len(valid_channels)} channels")
    
            power_trials, freqs_combined = compute_spectrum_trials(trials_array, fs)
    
            mean_spec = np.nanmean(power_trials, axis=0)
            resid_session = np.full_like(mean_spec, np.nan)
            freq_res = np.median(np.diff(freqs_combined))
            for ch_i, ch_name in enumerate(valid_channels):
                try:
                    lower_pw = max(2 * freq_res, 1.0)
                    upper_pw = 12
                    if lower_pw >= upper_pw:
                        continue
                    fm = FOOOF(peak_width_limits=[lower_pw, upper_pw],
                               max_n_peaks=6, min_peak_height=0.05,
                               peak_threshold=1.5, aperiodic_mode='fixed')
                    fm.fit(freqs_combined, mean_spec[:, ch_i])
                    resid_session[:, ch_i] = fm._spectrum_flat
                except Exception as e:
                    print(f"  [correct] FOOOF failed {session_name}, ch {ch_name}: {e}")
    
            for dct, data_in, xaxis in [
                (state_data_timelock_corr, trials_array, time_vec),
                (state_data_spectra_corr, power_trials, freqs_combined),
            ]:
                if state_value not in dct:
                    dct[state_value] = []
                dct[state_value].append(
                    {'trials': data_in, 'time': xaxis, 'channels': valid_channels})
            if state_value not in state_data_residuals_corr:
                state_data_residuals_corr[state_value] = []
            state_data_residuals_corr[state_value].append(
                {'resid': resid_session, 'freqs': freqs_combined,
                 'channels': valid_channels})
    
    
    print("\n=== [correct] Array-level permutation tests ===")
    pairs_corr = list(itertools.combinations(sorted(state_data_timelock_corr.keys()), 2))
    
    
    def _array_perm_save_npz(store, plot_type, ch_names_used, s1, s2,
                             array_index, npz_path):
        """Run array-level perm test on the given channel set and save the npz
        cache only (no per-array PDF). Mirrors the array-level computation in
        the all-trials block above."""
        vals1, vals2 = [], []
        x_axis = None
        for sess in store[s1]:
            ch_valid = [c for c in ch_names_used if c in sess['channels']]
            if not ch_valid:
                continue
            ch_idx = [sess['channels'].index(c) for c in ch_valid]
            x_axis = sess['time'] if plot_type != 'residual' else sess['freqs']
            if plot_type == 'residual':
                vals1.append(np.mean(sess['resid'][:, ch_idx], axis=1))
            else:
                vals1.append(np.mean(np.mean(sess['trials'][:, :, ch_idx], axis=0), axis=1))
        for sess in store[s2]:
            ch_valid = [c for c in ch_names_used if c in sess['channels']]
            if not ch_valid:
                continue
            ch_idx = [sess['channels'].index(c) for c in ch_valid]
            x_axis = sess['time'] if plot_type != 'residual' else sess['freqs']
            if plot_type == 'residual':
                vals2.append(np.mean(sess['resid'][:, ch_idx], axis=1))
            else:
                vals2.append(np.mean(np.mean(sess['trials'][:, :, ch_idx], axis=0), axis=1))
        if not (vals1 and vals2):
            return
        min_samples = min(v.shape[0] for v in vals1 + vals2)
        vals1 = [v[:min_samples] for v in vals1]
        vals2 = [v[:min_samples] for v in vals2]
        x_axis = x_axis[:min_samples]
        data1 = np.stack(vals1, axis=0)
        data2 = np.stack(vals2, axis=0)
        diff, sig, thr = permutation_test(
            data1, data2, n_perms=n_perms, alpha=alpha, rng=rng)
    
        if not os.path.exists(npz_path):
            np.savez_compressed(
                npz_path,
                diff=diff, sig=sig, thr=thr,
                mean1=np.nanmean(data1, axis=0),
                mean2=np.nanmean(data2, axis=0),
                x_axis=x_axis, s1=s1, s2=s2, plot_type=plot_type,
                array_index=array_index)
    
    
    for plot_type, store in [('timelock', state_data_timelock_corr),
                             ('spectra', state_data_spectra_corr),
                             ('residual', state_data_residuals_corr)]:
        if not store:
            continue
        first_channels = store[next(iter(store))][0]['channels']
        Sig_CH = np.array_split(first_channels, 6)
    
        for (s1, s2) in pairs_corr:
            print(f"  --> {plot_type}: state {s1} vs {s2}")
    
            # Per-array (1..6) -> permdata_{plot}_pair{s1}_{s2}_ARRAY_array{i+1}.npz
            for i_arr, ch_names in enumerate(Sig_CH):
                npz_name = (f"permdata_{plot_type}_pair{s1}_{s2}"
                            f"_ARRAY_array{i_arr+1}.npz")
                _array_perm_save_npz(
                    store, plot_type, ch_names, s1, s2,
                    array_index=i_arr + 1,
                    npz_path=os.path.join(correct_results_dir, npz_name))
    
            # Merged: Array 1-3 + individual arrays 4/5/6
            # -> cb_permdata_{plot}_pair{s1}_{s2}_{label_no_dash}.npz
            for i_arr, ch_names in enumerate(Sig_CH):
                if i_arr < 3:
                    if i_arr == 0:
                        combined_ch_names = np.concatenate(Sig_CH[:3])
                        array_label = "Array 1-3"
                    else:
                        continue
                else:
                    combined_ch_names = ch_names
                    array_label = f"Array {i_arr+1}"
                npz_name = (f"cb_permdata_{plot_type}_pair{s1}_{s2}"
                            f"_{array_label.replace('-', '')}.npz")
                _array_perm_save_npz(
                    store, plot_type, combined_ch_names, s1, s2,
                    array_index=i_arr + 1,
                    npz_path=os.path.join(correct_results_dir, npz_name))
    
    
    # -----------------------------
    # Summary figures (correct trials, array-level only)
    # -----------------------------
    correct_summary_dir = os.path.join(correct_output_dir, 'summary_plots')
    os.makedirs(correct_summary_dir, exist_ok=True)
    
    
    def load_permdata_correct(plot_type, s1, s2, array_index):
        fname = f"permdata_{plot_type}_pair{s1}_{s2}_ARRAY_array{array_index}.npz"
        fpath = os.path.join(correct_results_dir, fname)
        if not os.path.exists(fpath):
            return None
        return np.load(fpath)
    
    
    def load_permdata_merged_correct(plot_type, s1, s2, array_label):
        fname = f"cb_permdata_{plot_type}_pair{s1}_{s2}_{array_label.replace('-', '')}.npz"
        fpath = os.path.join(correct_results_dir, fname)
        if not os.path.exists(fpath):
            return None
        return np.load(fpath)
    
    
    pairs_fig_corr = list(itertools.combinations(
        [s for s in sorted(state_data_timelock_corr.keys()) if s != 1], 2))
    
    # Figure 1: per-array grid (rows = arrays, cols = state pairs)
    for plot_type in plot_types_fig:
        ylabel = 'Amplitude' if plot_type == 'timelock' else 'Residual Power'
        n_rows, n_cols = len(arrays_list), len(pairs_fig_corr)
        if n_cols == 0:
            continue
    
        fig_real, axes_real = plt.subplots(n_rows, n_cols,
                                           figsize=(6*n_cols, 3*n_rows),
                                           sharex='col', sharey='row')
        if n_rows == 1: axes_real = np.expand_dims(axes_real, 0)
        if n_cols == 1: axes_real = np.expand_dims(axes_real, 1)
    
        summary_mask = []
        x_axis = None
        for i_array, array_index in enumerate(arrays_list):
            summary_mask_array = []
            for j_pair, (s1, s2) in enumerate(pairs_fig_corr):
                ax = axes_real[i_array, j_pair]
                perm_array = load_permdata_correct(plot_type, s1, s2, array_index)
                if perm_array is None:
                    ax.axis('off')
                    summary_mask_array.append(None)
                    continue
    
                x_axis = perm_array['x_axis']
                data1, data2 = perm_array['mean1'], perm_array['mean2']
                sig_mask = perm_array['sig']
    
                ax.plot(x_axis, data1, color=state_colors[s1], label=f"State {s1}")
                ax.plot(x_axis, data2, color=state_colors[s2], label=f"State {s2}")
                if plot_type == 'timelock':
                    ax.set_ylim(-15, 15)
                ax.fill_between(x_axis, ax.get_ylim()[0], ax.get_ylim()[1],
                                where=sig_mask, color=sig_color, alpha=0.4)
    
                if i_array == 0: ax.set_title(f"{s1} vs {s2}", fontsize=10)
                if j_pair == 0: ax.set_ylabel(f"Array {array_index}\n{ylabel}")
                if i_array == n_rows - 1:
                    xlabel = ('Time rel. reward (s)' if plot_type == 'timelock'
                              else 'Frequency (Hz)')
                    ax.set_xlabel(xlabel)
                ax.legend(fontsize=6)
    
                summary_mask_array.append(sig_mask)
            summary_mask.append(summary_mask_array)
    
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08,
                            wspace=0.25, hspace=0.25)
        plt.suptitle(
            f"{plot_type} - Mean across all sessions (reward-centered, correct only)",
            fontsize=14)
        plt.savefig(os.path.join(correct_summary_dir,
                                 f"{plot_type}_all_arrays_pairs.pdf"))
        plt.close()
    
        # Figure 2: pairwise significance heatmap
        if x_axis is not None:
            n_arrays_fig = len(arrays_list)
            n_pairs_fig = len(pairs_fig_corr)
            n_time = len(x_axis)
            summary_array = np.zeros((n_arrays_fig, n_pairs_fig, n_time), dtype=int)
            for i_array in range(n_arrays_fig):
                for j_pair in range(n_pairs_fig):
                    mask = summary_mask[i_array][j_pair]
                    if mask is not None and mask.shape[0] == n_time:
                        summary_array[i_array, j_pair, :] = mask.astype(int)
    
            fig_sum, axes_sum = plt.subplots(1, n_pairs_fig,
                                             figsize=(6*n_pairs_fig, 4), sharey=True)
            if n_pairs_fig == 1: axes_sum = [axes_sum]
            for j_pair, (s1, s2) in enumerate(pairs_fig_corr):
                ax = axes_sum[j_pair]
                im = ax.imshow(summary_array[:, j_pair, :], cmap=teal_cmap,
                               aspect='auto', interpolation='none',
                               extent=[x_axis[0], x_axis[-1], 0.5, n_arrays_fig + 0.5])
                ax.set_title(f"{s1} vs {s2}")
                ax.set_xlabel('Time rel. reward (s)' if plot_type == 'timelock'
                              else 'Frequency (Hz)')
                if j_pair == 0:
                    ax.set_ylabel('Arrays')
                    ax.set_yticks(range(1, n_arrays_fig + 1))
                    ax.set_yticklabels([str(a) for a in arrays_list])
            plt.subplots_adjust(left=0.05, right=0.88, top=0.88, bottom=0.12, wspace=0.3)
            cbar_ax = fig_sum.add_axes([0.90, 0.12, 0.02, 0.76])
            cbar = fig_sum.colorbar(im, cax=cbar_ax)
            cbar.set_label('Significant (1=pairwise)')
            plt.suptitle(
                f"{plot_type} - Pairwise significance across arrays "
                "(reward-centered, correct only)",
                fontsize=14)
            plt.savefig(os.path.join(correct_summary_dir,
                                     f"{plot_type}_pairwise_summary_all_arrays.pdf"))
            plt.close()
    
    # Figure 3: merged-array grid (rows = state pairs, cols = Array 1-3, 4, 5, 6)
    array_labels_merged = ['Array 1-3', 'Array 4', 'Array 5', 'Array 6']
    for plot_type in plot_types_fig:
        ylabel = 'Amplitude' if plot_type == 'timelock' else 'Residual Power'
        n_rows, n_cols = len(pairs_fig_corr), len(array_labels_merged)
        if n_rows == 0:
            continue
    
        fig_real, axes_real = plt.subplots(n_rows, n_cols,
                                           figsize=(6*n_cols, 3*n_rows),
                                           sharex='col', sharey='row')
        if n_rows == 1: axes_real = np.expand_dims(axes_real, 0)
        if n_cols == 1: axes_real = np.expand_dims(axes_real, 1)
    
        for i_pair, (s1, s2) in enumerate(pairs_fig_corr):
            for j_array, array_label in enumerate(array_labels_merged):
                ax = axes_real[i_pair, j_array]
                perm_array = load_permdata_merged_correct(plot_type, s1, s2, array_label)
                if perm_array is None:
                    ax.axis('off')
                    continue
    
                x_axis = perm_array['x_axis']
                data1, data2 = perm_array['mean1'], perm_array['mean2']
                sig_mask = perm_array['sig']
    
                ax.plot(x_axis, data1, color=state_colors[s1], label=f"State {s1}")
                ax.plot(x_axis, data2, color=state_colors[s2], label=f"State {s2}")
                if plot_type == 'timelock':
                    ax.set_ylim(-15, 15)
                ax.fill_between(x_axis, ax.get_ylim()[0], ax.get_ylim()[1],
                                where=sig_mask, color=sig_color, alpha=0.4)
    
                if i_pair == 0: ax.set_title(f"{array_label}", fontsize=10)
                if j_array == 0: ax.set_ylabel(f"{s1} vs {s2}\n{ylabel}")
                if i_pair == n_rows - 1:
                    xlabel = ('Time rel. reward (s)' if plot_type == 'timelock'
                              else 'Frequency (Hz)')
                    ax.set_xlabel(xlabel)
                ax.legend(fontsize=6)
    
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08,
                            wspace=0.25, hspace=0.25)
        plt.suptitle(
            f"{plot_type} - Mean across all sessions, merged arrays "
            "(reward-centered, correct only)",
            fontsize=14)
        plt.savefig(os.path.join(correct_summary_dir,
                                 f"{plot_type}_merged_arrays_pairs.pdf"))
        plt.close()
    
    print(f"\nCorrect-trials summary figures saved under {correct_summary_dir}")


if correct_only:
    run_correct_trials_analysis()
else:
    print("\nSkipping correct-trials-only analysis (correct_only=False)")
