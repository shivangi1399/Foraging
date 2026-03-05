"""
Summary
-------
This script runs all pairwise permutation tests within state pairs,
crossing state identity and difficulty level.

Easy trials: Difficulty in [3430, 3470]
Hard trials: Difficulty in [3449, 3451]

Two state pairs are defined:
  1. States 0 & 2
  2. States 1 & 3

Within each pair, data is split into 4 groups (stateA_easy, stateA_hard,
stateB_easy, stateB_hard) and all 6 pairwise comparisons are tested:
  e.g. s0_easy vs s0_hard, s0_easy vs s2_easy, s0_easy vs s2_hard,
       s0_hard vs s2_easy, s0_hard vs s2_hard, s2_easy vs s2_hard

For each comparison the same pipeline is used:
  • ERPs, power spectra, FOOOF residuals
  • Per-channel permutation tests
  • Array-level permutation tests (6 arrays + combined 1-3)
"""

# -----------------------------
# Imports
# -----------------------------
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fooof import FOOOF
import syncopy as spy
import json
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog  # noqa: E402

# -----------------------------
# User Config
# -----------------------------
lfp_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/clean_full_length'
trial_info_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/full_length'
states_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis'
output_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_lfp/easy_hard_by_states/200_600/erp_spectra'
results_dir = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp'
results_data_dir = os.path.join(results_dir, "easy_hard_by_states", "200_600", "erp_spectra")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_data_dir, exist_ok=True)

sessions = ['20230203', '20230208', '20230209', '20230213', '20230214']
N_STATES_TO_USE = 4
n_perms = 1000
alpha = 0.05
rng = np.random.default_rng(42)

EASY_DIFFICULTIES = [3430, 3470]
STATE_PAIRS = {
    'states_0_2': [0, 2],
    'states_1_3': [1, 3],
}

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


def extract_power_trials(freq_analysis):
    try:
        freqs = getattr(freq_analysis, 'freq', None)
        channels = getattr(freq_analysis, 'channel', None)
        candidate = np.squeeze(freq_analysis.trials)
        if candidate.ndim == 3:
            return candidate, freqs, channels
        if candidate.ndim == 2 and freqs is not None and channels is not None:
            if candidate.shape[1] == len(freqs) * len(channels):
                reshaped = candidate.reshape((candidate.shape[0], len(freqs), len(channels)))
                return reshaped, freqs, channels
        return None, None, None
    except Exception as e:
        print("extract_power_trials error:", e)
        return None, None, None


def ensure_trialindex_in_trialdefinition(datalfp):
    if datalfp.trialdefinition.shape[1] < 4:
        nTrials = datalfp.trialdefinition.shape[0]
        datalfp.trialdefinition = np.hstack(
            (datalfp.trialdefinition, np.arange(nTrials).reshape(-1, 1))
        )


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
# Process each state pair separately
# -----------------------------
for pair_label, state_values in STATE_PAIRS.items():
    print(f"\n{'='*60}")
    print(f"Processing {pair_label}: states {state_values}, easy vs hard")
    print(f"{'='*60}")

    pair_output_dir = os.path.join(output_dir, pair_label)
    pair_results_dir = os.path.join(results_data_dir, pair_label)
    os.makedirs(pair_output_dir, exist_ok=True)
    os.makedirs(pair_results_dir, exist_ok=True)

    # Data stores: keyed by (state, difficulty) tuples
    # e.g. (0, 'easy'), (0, 'hard'), (2, 'easy'), (2, 'hard')
    diff_data_timelock = {(s, d): [] for s in state_values for d in ['easy', 'hard']}
    diff_data_spectra = {(s, d): [] for s in state_values for d in ['easy', 'hard']}
    diff_data_residuals = {(s, d): [] for s in state_values for d in ['easy', 'hard']}

    for session_name in sessions:
        print(f"\n--- Session {session_name} ---")
        lfp_path = os.path.join(lfp_data_dir, session_name, 'Cleaned_lfp_FT.spy')
        trial_info_path = os.path.join(trial_info_dir, session_name, 'Trial_Info.pkl')
        if not os.path.exists(lfp_path) or not os.path.exists(trial_info_path):
            continue

        # state info
        predicted_states = session_to_probs[session_name]
        trial_info_df = pd.read_pickle(trial_info_path)
        trial_info_df.iloc[:, 0] = (trial_info_df.iloc[:, 0] - 1000).astype('Int64')
        stim_df = pd.DataFrame({'TrialIndex': np.arange(len(predicted_states)), 'States': predicted_states})
        combined_df = pd.merge(trial_info_df, stim_df, left_on='Trial_Number', right_on='TrialIndex', how='inner')

        # load LFP data
        datalfp = spy.load(lfp_path)
        ensure_trialindex_in_trialdefinition(datalfp)
        cfg = spy.StructDict(latency=[-0.2, 0.6])
        data = spy.selectdata(cfg, datalfp)
        selected_trials = data.trialdefinition[:, 3].astype(int)
        states_trial_info_filt = combined_df[combined_df['TrialIndex'].isin(selected_trials)]

        for state_val in state_values:
            state_df = states_trial_info_filt[states_trial_info_filt['States'] == state_val]
            if len(state_df) == 0:
                print(f"  No trials for state {state_val} in session {session_name}")
                continue

            # Split into easy and hard
            easy_mask = state_df['Difficulty'].isin(EASY_DIFFICULTIES)
            easy_df = state_df[easy_mask]
            hard_df = state_df[~easy_mask]

            print(f"  State {state_val}: {len(easy_df)} easy trials, {len(hard_df)} hard trials")

            for diff_label, diff_df in [('easy', easy_df), ('hard', hard_df)]:
                if len(diff_df) == 0:
                    print(f"  No {diff_label} trials for state {state_val}, skipping")
                    continue

                # Get trial indices relative to the selected data
                trial_indices_in_data = []
                for trial_idx in diff_df['TrialIndex'].values:
                    where = np.where(selected_trials == trial_idx)[0]
                    if len(where) > 0:
                        trial_indices_in_data.append(where[0])
                trial_indices_in_data = np.array(trial_indices_in_data)

                if len(trial_indices_in_data) == 0:
                    continue

                cfg_sel = spy.StructDict(trials=trial_indices_in_data)
                datas_sel = spy.selectdata(cfg_sel, data)
                datas_clean, valid_channels = remove_nan_trials_channels(datas_sel)
                if datas_clean is None:
                    continue

                # --- Timelock trials ---
                trials_array = np.stack(datas_clean.trials, axis=0)
                time_vec = datas_clean.time[0]

                # --- Spectra ---
                cfg_low = spy.StructDict(method='mtmfft', foilim=[2, 30], out='pow', keeptrials=True, taper='hann')
                freq_low = spy.freqanalysis(datas_clean, cfg_low)
                low_power, freqs_low, channels_low = extract_power_trials(freq_low)

                cfg_high = spy.StructDict(method='mtmfft', foilim=[30, 100], out='pow', keeptrials=True, tapsmofrq=4)
                freq_high = spy.freqanalysis(datas_clean, cfg_high)
                high_power, freqs_high, channels_high = extract_power_trials(freq_high)

                if low_power is None or high_power is None:
                    continue

                power_trials = np.concatenate((low_power, high_power), axis=1)
                freqs_combined = np.concatenate((freqs_low, freqs_high))

                # --- FOOOF on mean spectrum per channel ---
                mean_spec = np.nanmean(power_trials, axis=0)
                resid_session = np.full_like(mean_spec, np.nan)
                freq_res = np.median(np.diff(freqs_combined))
                for ch_i, ch_name in enumerate(valid_channels):
                    try:
                        fm = FOOOF(peak_width_limits=[max(2 * freq_res, 1.0), 12],
                                   max_n_peaks=6,
                                   min_peak_height=0.05,
                                   peak_threshold=1.5,
                                   aperiodic_mode='fixed')
                        fm.fit(freqs_combined, mean_spec[:, ch_i])
                        resid_session[:, ch_i] = fm._spectrum_flat
                    except Exception as e:
                        print(f"FOOOF failed {session_name}, state {state_val}, ch {ch_name}: {e}")

                # Store keyed by (state, difficulty)
                key = (state_val, diff_label)
                diff_data_timelock[key].append({
                    'trials': trials_array, 'time': time_vec, 'channels': valid_channels
                })
                diff_data_spectra[key].append({
                    'trials': power_trials, 'time': freqs_combined, 'channels': valid_channels
                })
                diff_data_residuals[key].append({
                    'resid': resid_session, 'freqs': freqs_combined, 'channels': valid_channels
                })

    # -----------------------------
    # Generate all 6 pairwise comparisons
    # -----------------------------
    groups = [(s, d) for s in state_values for d in ['easy', 'hard']]
    all_comparisons = list(combinations(groups, 2))

    print(f"\n=== Running {len(all_comparisons)} pairwise permutation tests for {pair_label} ===")
    for (s1_state, s1_diff), (s2_state, s2_diff) in all_comparisons:
        s1_label = f"s{s1_state}_{s1_diff}"
        s2_label = f"s{s2_state}_{s2_diff}"
        comp_label = f"{s1_label}_vs_{s2_label}"
        comp_output_dir = os.path.join(pair_output_dir, comp_label)
        comp_results_dir = os.path.join(pair_results_dir, comp_label)
        os.makedirs(comp_output_dir, exist_ok=True)
        os.makedirs(comp_results_dir, exist_ok=True)

        key1 = (s1_state, s1_diff)
        key2 = (s2_state, s2_diff)

        print(f"\n--> Comparison: {comp_label}")

        for plot_type, store in [('timelock', diff_data_timelock),
                                 ('spectra', diff_data_spectra),
                                 ('residual', diff_data_residuals)]:
            if not store[key1] or not store[key2]:
                print(f"  Skipping {plot_type}: missing data for {s1_label} or {s2_label}")
                continue

            first_channels = store[key1][0]['channels']
            Sig_CH = np.array_split(first_channels, 6)

            print(f"  --> {plot_type}")

            for i_arr, ch_names in enumerate(Sig_CH):
                # --- Per-channel plots ---
                fig, axes = plt.subplots(6, 6, figsize=(15, 12))
                axes = axes.flatten()
                for ichan, ch_name in enumerate(ch_names):
                    ax = axes[ichan]
                    vals1, vals2 = [], []
                    x_axis = None
                    for sess in store[key1]:
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
                    for sess in store[key2]:
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
                        data1 = np.concatenate(vals1, axis=0)
                        data2 = np.concatenate(vals2, axis=0)

                    diff, sig, thr = permutation_test(data1, data2, n_perms=n_perms, alpha=alpha, rng=rng)

                    # Save channel level results
                    npz_name = f"permdata_{plot_type}_{comp_label}_array{i_arr+1}_{ch_name}.npz"
                    npz_path = os.path.join(comp_results_dir, npz_name)
                    if not os.path.exists(npz_path):
                        mean1 = np.nanmean(data1, axis=0)
                        mean2 = np.nanmean(data2, axis=0)
                        np.savez_compressed(
                            npz_path,
                            diff=diff, sig=sig, thr=thr,
                            mean1=mean1, mean2=mean2,
                            x_axis=x_axis,
                            s1=s1_label, s2=s2_label,
                            plot_type=plot_type,
                            ch_name=ch_name,
                            array_index=i_arr + 1,
                            state_pair=pair_label
                        )

                    # Plotting
                    ax.plot(x_axis, diff, color='k')
                    ax.fill_between(x_axis, diff, where=sig, color='red', alpha=0.4)
                    ax.axhline(0, color='gray', lw=0.5)
                    ax.set_title(ch_name, fontsize=7)
                for j in range(len(ch_names), 36):
                    axes[j].set_visible(False)
                fig.suptitle(f"{pair_label} | {plot_type} {comp_label} - Array {i_arr+1}")
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                fname = os.path.join(comp_output_dir, f"perm_{plot_type}_{comp_label}_array{i_arr+1}.pdf")
                fig.savefig(fname)
                plt.close(fig)

                # --- Array-level combined analysis ---
                print(f"    Array-level stats for Array {i_arr+1}")
                vals1_array, vals2_array = [], []
                x_axis = None
                for sess in store[key1]:
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
                for sess in store[key2]:
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
                    data1_array = np.stack(vals1_array, axis=0)
                    data2_array = np.stack(vals2_array, axis=0)
                    diff_array, sig_array, thr_array = permutation_test(
                        data1_array, data2_array, n_perms=n_perms, alpha=alpha, rng=rng)
                    fig_arr, ax_arr = plt.subplots(figsize=(6, 4))
                    ax_arr.plot(x_axis, diff_array, color='k', lw=1.5)
                    ax_arr.fill_between(x_axis, diff_array, where=sig_array, color='red', alpha=0.4)
                    ax_arr.axhline(0, color='gray', lw=0.8)
                    ax_arr.set_title(f"{pair_label} | Array {i_arr+1} ({plot_type}) {comp_label}")
                    ax_arr.set_xlabel('Time (s)' if plot_type == 'timelock' else 'Frequency (Hz)')
                    ax_arr.set_ylabel('ΔAmplitude' if plot_type == 'timelock' else 'ΔResidual Power')
                    plt.tight_layout()
                    fname_arr = os.path.join(
                        comp_output_dir, f"perm_{plot_type}_{comp_label}_ARRAYCOMBINED_array{i_arr+1}.pdf")
                    fig_arr.savefig(fname_arr)
                    plt.close(fig_arr)

                    npz_array_name = f"permdata_{plot_type}_{comp_label}_ARRAY_array{i_arr+1}.npz"
                    npz_array_path = os.path.join(comp_results_dir, npz_array_name)
                    if not os.path.exists(npz_array_path):
                        mean1_array = np.nanmean(data1_array, axis=0)
                        mean2_array = np.nanmean(data2_array, axis=0)
                        np.savez_compressed(
                            npz_array_path,
                            diff=diff_array, sig=sig_array, thr=thr_array,
                            mean1=mean1_array, mean2=mean2_array,
                            x_axis=x_axis,
                            s1=s1_label, s2=s2_label,
                            plot_type=plot_type,
                            array_index=i_arr + 1,
                            state_pair=pair_label
                        )

    # -----------------------------
    # Array-level grouped (arrays 1-3 combined)
    # -----------------------------
    print(f"\n=== Array-level grouped tests for {pair_label} ===")

    for (s1_state, s1_diff), (s2_state, s2_diff) in all_comparisons:
        s1_label = f"s{s1_state}_{s1_diff}"
        s2_label = f"s{s2_state}_{s2_diff}"
        comp_label = f"{s1_label}_vs_{s2_label}"
        comp_output_dir = os.path.join(pair_output_dir, comp_label)
        comp_results_dir = os.path.join(pair_results_dir, comp_label)

        key1 = (s1_state, s1_diff)
        key2 = (s2_state, s2_diff)

        for plot_type, store in [('timelock', diff_data_timelock),
                                 ('spectra', diff_data_spectra),
                                 ('residual', diff_data_residuals)]:
            if not store[key1] or not store[key2]:
                continue

            first_channels = store[key1][0]['channels']
            Sig_CH = np.array_split(first_channels, 6)

            print(f"--> {comp_label} {plot_type} (grouped arrays)")
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

                for sess in store[key1]:
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

                for sess in store[key2]:
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
                    data1_array = np.stack(vals1_array, axis=0)
                    data2_array = np.stack(vals2_array, axis=0)
                    diff_array, sig_array, thr_array = permutation_test(
                        data1_array, data2_array, n_perms=n_perms, alpha=alpha, rng=rng)

                    array_label = f"Array {i_arr+1}" if i_arr >= 3 else "Array 1-3"
                    fig_arr, ax_arr = plt.subplots(figsize=(6, 4))
                    ax_arr.plot(x_axis, diff_array, color='k', lw=1.5)
                    ax_arr.fill_between(x_axis, diff_array, where=sig_array, color='red', alpha=0.4)
                    ax_arr.axhline(0, color='gray', lw=0.8)
                    ax_arr.set_title(f"{pair_label} | {array_label} ({plot_type}) {comp_label}")
                    ax_arr.set_xlabel('Time (s)' if plot_type == 'timelock' else 'Frequency (Hz)')
                    ax_arr.set_ylabel('ΔAmplitude' if plot_type == 'timelock' else 'ΔResidual Power')
                    plt.tight_layout()
                    fname_arr = os.path.join(
                        comp_output_dir, f"cb_perm_{plot_type}_{comp_label}_{array_label.replace('-', '')}.pdf")
                    fig_arr.savefig(fname_arr)
                    plt.close(fig_arr)

                    npz_array_name = f"cb_permdata_{plot_type}_{comp_label}_{array_label.replace('-', '')}.npz"
                    npz_array_path = os.path.join(comp_results_dir, npz_array_name)
                    if not os.path.exists(npz_array_path):
                        mean1_array = np.nanmean(data1_array, axis=0)
                        mean2_array = np.nanmean(data2_array, axis=0)
                        np.savez_compressed(
                            npz_array_path,
                            diff=diff_array, sig=sig_array, thr=thr_array,
                            mean1=mean1_array, mean2=mean2_array,
                            x_axis=x_axis,
                            s1=s1_label, s2=s2_label,
                            plot_type=plot_type,
                            array_index=i_arr + 1,
                            state_pair=pair_label
                        )

    # =============================================================
    # Summary figures for state pair
    # =============================================================
    print(f"\n=== Generating summary figures for {pair_label} ===")

    summary_output_dir = os.path.join(pair_output_dir, 'summary')
    os.makedirs(summary_output_dir, exist_ok=True)

    # Colors for each (state, difficulty) group
    group_colors = {}
    base_state_colors = {
        0: (0.55, 0.0, 0.55),   # purple
        1: (0.0, 0.39, 0.39),   # teal
        2: (0.8, 0.33, 0.0),    # orange
        3: (0.25, 0.35, 0.55)   # slate blue
    }
    for s in state_values:
        group_colors[(s, 'easy')] = base_state_colors[s]
        # Lighter version for hard
        r, g, b = base_state_colors[s]
        group_colors[(s, 'hard')] = (min(r + 0.3, 1.0), min(g + 0.3, 1.0), min(b + 0.3, 1.0))

    sig_color = '#8dd3c7'
    teal_cmap = LinearSegmentedColormap.from_list('teal_map', ['white', '#1f9e89'])
    arrays_list = [1, 2, 3, 4, 5, 6]

    def load_array_permdata(comp_label, plot_type, array_index):
        """Load array-level permutation results for a comparison."""
        fname = f"permdata_{plot_type}_{comp_label}_ARRAY_array{array_index}.npz"
        fpath = os.path.join(pair_results_dir, comp_label, fname)
        if not os.path.exists(fpath):
            return None
        return np.load(fpath)

    def load_merged_permdata(comp_label, plot_type, array_label):
        """Load merged-array permutation results for a comparison."""
        fname = f"cb_permdata_{plot_type}_{comp_label}_{array_label.replace('-', '')}.npz"
        fpath = os.path.join(pair_results_dir, comp_label, fname)
        if not os.path.exists(fpath):
            return None
        return np.load(fpath)

    # Re-generate comparison list for summary
    groups = [(s, d) for s in state_values for d in ['easy', 'hard']]
    all_comparisons_summary = list(combinations(groups, 2))
    comp_labels = []
    for (s1s, s1d), (s2s, s2d) in all_comparisons_summary:
        comp_labels.append(f"s{s1s}_{s1d}_vs_s{s2s}_{s2d}")

    for plot_type in ['timelock', 'residual']:
        ylabel = 'Amplitude' if plot_type == 'timelock' else 'Residual Power'
        n_rows, n_cols = len(arrays_list), len(comp_labels)

        # -----------------------------
        # Figure 1: Real data + significance masks (arrays × comparisons)
        # -----------------------------
        fig_real, axes_real = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows),
                                           sharex='col', sharey='row')
        if n_rows == 1:
            axes_real = np.expand_dims(axes_real, 0)
        if n_cols == 1:
            axes_real = np.expand_dims(axes_real, 1)

        summary_mask = []

        for i_array, array_index in enumerate(arrays_list):
            summary_mask_array = []
            for j_comp, cl in enumerate(comp_labels):
                ax = axes_real[i_array, j_comp]
                perm_data = load_array_permdata(cl, plot_type, array_index)

                if perm_data is None:
                    ax.axis('off')
                    summary_mask_array.append(None)
                    continue

                x_axis = perm_data['x_axis']
                mean1 = perm_data['mean1']
                mean2 = perm_data['mean2']
                sig_mask = perm_data['sig']
                s1_lbl = str(perm_data['s1'])
                s2_lbl = str(perm_data['s2'])

                # Determine group keys for colors
                (s1s, s1d), (s2s, s2d) = all_comparisons_summary[j_comp]
                c1 = group_colors.get((s1s, s1d), 'blue')
                c2 = group_colors.get((s2s, s2d), 'red')

                ax.plot(x_axis, mean1, color=c1, label=s1_lbl)
                ax.plot(x_axis, mean2, color=c2, label=s2_lbl)
                ax.fill_between(x_axis, ax.get_ylim()[0], ax.get_ylim()[1],
                                where=sig_mask, color=sig_color, alpha=0.4)

                if i_array == 0:
                    ax.set_title(cl.replace('_', ' '), fontsize=8)
                if j_comp == 0:
                    ax.set_ylabel(f"Array {array_index}\n{ylabel}")
                if i_array == n_rows - 1:
                    xlabel = 'Time (s)' if plot_type == 'timelock' else 'Frequency (Hz)'
                    ax.set_xlabel(xlabel)
                ax.legend(fontsize=5)

                summary_mask_array.append(sig_mask)
            summary_mask.append(summary_mask_array)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.25, hspace=0.25)
        plt.suptitle(f"{pair_label} | {plot_type} - All comparisons across arrays", fontsize=14)
        plt.savefig(os.path.join(summary_output_dir, f"{plot_type}_all_arrays_comparisons.pdf"))
        plt.close()

        # -----------------------------
        # Figure 2: Summary heatmap (significance across arrays × comparisons)
        # -----------------------------
        # Get x_axis from first available result
        ref_x = None
        for cl in comp_labels:
            for ai in arrays_list:
                pd_ref = load_array_permdata(cl, plot_type, ai)
                if pd_ref is not None:
                    ref_x = pd_ref['x_axis']
                    break
            if ref_x is not None:
                break

        if ref_x is not None:
            n_time = len(ref_x)
            summary_array = np.zeros((len(arrays_list), len(comp_labels), n_time), dtype=int)

            for i_array in range(len(arrays_list)):
                for j_comp in range(len(comp_labels)):
                    mask = summary_mask[i_array][j_comp]
                    if mask is not None and mask.shape[0] == n_time:
                        summary_array[i_array, j_comp, :] = mask.astype(int)

            fig_sum, axes_sum = plt.subplots(1, len(comp_labels),
                                              figsize=(5 * len(comp_labels), 4), sharey=True)
            if len(comp_labels) == 1:
                axes_sum = [axes_sum]

            for j_comp, cl in enumerate(comp_labels):
                ax = axes_sum[j_comp]
                im = ax.imshow(summary_array[:, j_comp, :], cmap=teal_cmap, aspect='auto',
                               interpolation='none',
                               extent=[ref_x[0], ref_x[-1], 0.5, len(arrays_list) + 0.5])
                ax.set_title(cl.replace('_', ' '), fontsize=8)
                ax.set_xlabel('Time (s)' if plot_type == 'timelock' else 'Frequency (Hz)')
                if j_comp == 0:
                    ax.set_ylabel('Arrays')
                    ax.set_yticks(range(1, len(arrays_list) + 1))
                    ax.set_yticklabels([str(a) for a in arrays_list])

            plt.subplots_adjust(left=0.05, right=0.88, top=0.88, bottom=0.12, wspace=0.3)
            cbar_ax = fig_sum.add_axes([0.90, 0.12, 0.02, 0.76])
            cbar = fig_sum.colorbar(im, cax=cbar_ax)
            cbar.set_label('Significant')
            plt.suptitle(f"{pair_label} | {plot_type} - Significance summary", fontsize=14)
            plt.savefig(os.path.join(summary_output_dir,
                                      f"{plot_type}_pairwise_summary_heatmap.pdf"))
            plt.close()

        # -----------------------------
        # Figure 3: Merged arrays (1-3 combined, 4, 5, 6)
        # -----------------------------
        merged_labels = ['Array 1-3', 'Array 4', 'Array 5', 'Array 6']
        n_mrows, n_mcols = len(comp_labels), len(merged_labels)

        fig_merged, axes_merged = plt.subplots(n_mrows, n_mcols,
                                                figsize=(6 * n_mcols, 3 * n_mrows),
                                                sharex='col', sharey='row')
        if n_mrows == 1:
            axes_merged = np.expand_dims(axes_merged, 0)
        if n_mcols == 1:
            axes_merged = np.expand_dims(axes_merged, 1)

        for i_comp, cl in enumerate(comp_labels):
            for j_ml, ml in enumerate(merged_labels):
                ax = axes_merged[i_comp, j_ml]
                perm_data = load_merged_permdata(cl, plot_type, ml)

                if perm_data is None:
                    ax.axis('off')
                    continue

                x_axis = perm_data['x_axis']
                mean1 = perm_data['mean1']
                mean2 = perm_data['mean2']
                sig_mask = perm_data['sig']
                s1_lbl = str(perm_data['s1'])
                s2_lbl = str(perm_data['s2'])

                (s1s, s1d), (s2s, s2d) = all_comparisons_summary[i_comp]
                c1 = group_colors.get((s1s, s1d), 'blue')
                c2 = group_colors.get((s2s, s2d), 'red')

                ax.plot(x_axis, mean1, color=c1, label=s1_lbl)
                ax.plot(x_axis, mean2, color=c2, label=s2_lbl)
                ax.fill_between(x_axis, ax.get_ylim()[0], ax.get_ylim()[1],
                                where=sig_mask, color=sig_color, alpha=0.4)

                if i_comp == 0:
                    ax.set_title(ml, fontsize=10)
                if j_ml == 0:
                    ax.set_ylabel(f"{cl.replace('_', ' ')}\n{ylabel}", fontsize=7)
                if i_comp == n_mrows - 1:
                    xlabel = 'Time (s)' if plot_type == 'timelock' else 'Frequency (Hz)'
                    ax.set_xlabel(xlabel)
                ax.legend(fontsize=5)

        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, wspace=0.25, hspace=0.25)
        plt.suptitle(f"{pair_label} | {plot_type} - Merged arrays", fontsize=14)
        plt.savefig(os.path.join(summary_output_dir, f"{plot_type}_merged_arrays_comparisons.pdf"))
        plt.close()

    print(f"  Summary plots saved to {summary_output_dir}")

print(f"\nDone. Results saved under:\n  Plots: {output_dir}\n  Data:  {results_data_dir}")
