"""
Summary
-------
This script compares time-locked ERPs, power spectra, and FOOOF residual power
between cognitive states, restricted to the first 10% of trials after each
block exit (event marker 3091).

For each session:
- Block end trials are detected from the log file (event 3091 = exit button).
- The first 10% of trials in each new block are selected.
- LFP data for these post-exit trials are segmented and grouped by state.
- Three signal representations are computed:
    • Time-domain ERPs,
    • Power spectra (2–100 Hz, multitaper FFT),
    • FOOOF-derived residual spectra.

Across sessions:
- Trial-wise data are pooled by cognitive state.
- Pairwise permutation tests are performed between states at channel,
  array, and combined-array levels.
"""

# -----------------------------
# Imports
# -----------------------------
import os
import sys
import math
from datetime import datetime
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
raw_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/raw_data'
output_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_lfp/post_exit_10pct/200_600/erp_spectra'
results_dir = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp'
results_data_dir = os.path.join(results_dir, "post_exit_10pct", "200_600", "erp_spectra")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_data_dir, exist_ok=True)

sessions = ['20230203', '20230208', '20230209', '20230213', '20230214']
N_STATES_TO_USE = 4
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
# Main data collection
# -----------------------------
state_data_timelock = {}
state_data_spectra = {}
state_data_residuals = {}

for session_name in sessions:
    print(f"\n=== Processing session {session_name} ===")
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

    # --- Detect block end trials from log file (event 3091) ---
    session_date_str = datetime.strptime(session_name, "%Y%m%d").strftime("%Y_%m_%d")
    session_log_dir = os.path.join(raw_data_dir, session_name)
    log_files = [f for f in os.listdir(session_log_dir) if f.endswith('.log') and session_date_str in f]
    log_filepath = os.path.join(session_log_dir, log_files[0])
    with TextLog(log_filepath) as log:
        evt, ts, evt_desc, true_ts = log.parse_eventmarkers()
    trial_onset = ts[np.where(evt == 3000)[0]]
    block_idx = np.where(evt == 3091)[0]
    block_end_trial_indices = sorted(set(
        np.searchsorted(trial_onset, ts[block_idx], side='right') - 1
    ))

    # --- Compute first 10% of trials after each block exit ---
    n_total_trials = len(trial_onset)
    block_starts = [be + 1 for be in block_end_trial_indices if be + 1 < n_total_trials]
    block_boundaries = sorted(block_starts + [n_total_trials])
    post_exit_trial_set = set()
    for i in range(len(block_starts)):
        blk_start = block_starts[i]
        blk_end = block_boundaries[i + 1] if i + 1 < len(block_boundaries) else n_total_trials
        blk_len = blk_end - blk_start
        n_post = max(1, math.ceil(0.10 * blk_len))
        for t in range(blk_start, blk_start + n_post):
            post_exit_trial_set.add(t)
    print(f"  Block ends at trials: {block_end_trial_indices}")
    print(f"  Post-exit (first 10%) trial count: {len(post_exit_trial_set)}")

    # load LFP data
    datalfp = spy.load(lfp_path)
    ensure_trialindex_in_trialdefinition(datalfp)
    cfg = spy.StructDict(latency=[-0.2, 0.6])
    data = spy.selectdata(cfg, datalfp)
    selected_trials = data.trialdefinition[:, 3].astype(int)
    states_trial_info_filt = combined_df[combined_df['TrialIndex'].isin(selected_trials)]

    # --- Filter to post-exit trials and group by state ---
    post_exit_mask = states_trial_info_filt['TrialIndex'].isin(post_exit_trial_set)
    states_post_filt = states_trial_info_filt[post_exit_mask]
    if len(states_post_filt) == 0:
        print(f"  No post-exit trials overlap with LFP data for {session_name}")
        continue

    print(f"  Post-exit trials in LFP: {len(states_post_filt)}")
    unique_states = np.sort(np.unique(states_post_filt['States'].to_numpy()))[:N_STATES_TO_USE]
    for state_value in unique_states:
        state_rows = states_post_filt[states_post_filt['States'] == state_value]
        post_trial_positions = np.where(
            states_trial_info_filt['TrialIndex'].isin(state_rows['TrialIndex'])
        )[0]
        if len(post_trial_positions) == 0:
            continue
        cfg_sel = spy.StructDict(trials=post_trial_positions)
        datas_state = spy.selectdata(cfg_sel, data)
        datas_state_clean, valid_channels = remove_nan_trials_channels(datas_state)
        if datas_state_clean is None:
            continue

        # --- Timelock trials ---
        trials_array = np.stack(datas_state_clean.trials, axis=0)
        time_vec = datas_state_clean.time[0]

        # --- Spectra ---
        cfg_low = spy.StructDict(method='mtmfft', foilim=[2, 30], out='pow', keeptrials=True, taper='hann')
        freq_low = spy.freqanalysis(datas_state_clean, cfg_low)
        low_power, freqs_low, channels_low = extract_power_trials(freq_low)

        cfg_high = spy.StructDict(method='mtmfft', foilim=[30, 100], out='pow', keeptrials=True, tapsmofrq=4)
        freq_high = spy.freqanalysis(datas_state_clean, cfg_high)
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
                print(f"FOOOF failed {session_name}, ch {ch_name}: {e}")

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


# -----------------------------
# Permutation tests (pairwise)
# -----------------------------
if not state_data_timelock:
    print("\nNo post-exit trial data collected; skipping permutation tests.")
else:
    print("\n=== Running permutation tests for post-exit trials (first 10%) ===")
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
                # --- Per-channel plots ---
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
                            diff=diff, sig=sig, thr=thr,
                            mean1=mean1, mean2=mean2, x_axis=x_axis,
                            s1=s1, s2=s2, plot_type=plot_type,
                            ch_name=ch_name, array_index=i_arr + 1
                        )

                    ax.plot(x_axis, diff, color='k')
                    ax.fill_between(x_axis, diff, where=sig, color='red', alpha=0.4)
                    ax.axhline(0, color='gray', lw=0.5)
                    ax.set_title(ch_name, fontsize=7)
                for j in range(len(ch_names), 36):
                    axes[j].set_visible(False)
                fig.suptitle(f"Post-exit {plot_type} {s1} vs {s2} - Array {i_arr+1}")
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                fname = os.path.join(output_dir, f"perm_{plot_type}_pair{s1}_{s2}_array{i_arr+1}.pdf")
                fig.savefig(fname)
                plt.close(fig)

                # --- Array-level combined analysis ---
                print(f"  --> Array-level stats for Array {i_arr+1} ({plot_type})")
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
                    data1_array = np.stack(vals1_array, axis=0)
                    data2_array = np.stack(vals2_array, axis=0)
                    diff_array, sig_array, thr_array = permutation_test(
                        data1_array, data2_array, n_perms=n_perms, alpha=alpha, rng=rng)
                    fig_arr, ax_arr = plt.subplots(figsize=(6, 4))
                    ax_arr.plot(x_axis, diff_array, color='k', lw=1.5)
                    ax_arr.fill_between(x_axis, diff_array, where=sig_array, color='red', alpha=0.4)
                    ax_arr.axhline(0, color='gray', lw=0.8)
                    ax_arr.set_title(f"Post-exit Array {i_arr+1} ({plot_type}) {s1} vs {s2}")
                    ax_arr.set_xlabel('Time (s)' if plot_type == 'timelock' else 'Frequency (Hz)')
                    ax_arr.set_ylabel('ΔAmplitude' if plot_type == 'timelock' else 'ΔResidual Power')
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
                            diff=diff_array, sig=sig_array, thr=thr_array,
                            mean1=mean1_array, mean2=mean2_array, x_axis=x_axis,
                            s1=s1, s2=s2, plot_type=plot_type,
                            array_index=i_arr + 1
                        )

    # -----------------------------
    # array-level, grouped
    # -----------------------------
    print("\n=== Running permutation tests (array-level grouping) ===")
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
                    data1_array = np.stack(vals1_array, axis=0)
                    data2_array = np.stack(vals2_array, axis=0)
                    diff_array, sig_array, thr_array = permutation_test(
                        data1_array, data2_array, n_perms=n_perms, alpha=alpha, rng=rng)

                    fig_arr, ax_arr = plt.subplots(figsize=(6, 4))
                    ax_arr.plot(x_axis, diff_array, color='k', lw=1.5)
                    ax_arr.fill_between(x_axis, diff_array, where=sig_array, color='red', alpha=0.4)
                    ax_arr.axhline(0, color='gray', lw=0.8)
                    array_label = f"Array {i_arr+1}" if i_arr >= 3 else "Array 1-3"
                    ax_arr.set_title(f"Post-exit {array_label} ({plot_type}) {s1} vs {s2}")
                    ax_arr.set_xlabel('Time (s)' if plot_type == 'timelock' else 'Frequency (Hz)')
                    ax_arr.set_ylabel('ΔAmplitude' if plot_type == 'timelock' else 'ΔResidual Power')
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
                            diff=diff_array, sig=sig_array, thr=thr_array,
                            mean1=mean1_array, mean2=mean2_array, x_axis=x_axis,
                            s1=s1, s2=s2, plot_type=plot_type,
                            array_index=i_arr + 1
                        )

    print(f"Permutation-test results saved under {output_dir}")
