"""
Summary
-------
RT-centered version of erp_spectra_stats.py.

This script performs the same trial-wise, nonparametric statistical comparisons
of time-locked ERPs, power spectra, and aperiodic-adjusted (FOOOF) residual
power between cognitive states — but with LFP segments centered on each trial's
reaction time (RT) instead of stimulus onset.

For each session:
- LFP data are loaded (stimulus-aligned, full length).
- RT values are loaded from emissions.npy.
- For each trial, a fixed window around RT is extracted:
    [RT - 0.45 s, RT + 0.45 s]  (time 0 = RT).
- Trials and channels containing only NaNs are removed.
- Trials are grouped by cognitive state.
- Three signal representations are computed:
    • Time-domain ERPs (RT-locked LFP trials),
    • Power spectra (2 to 100 Hz, using periodogram, keeping trials),
    • FOOOF-derived residual spectra computed from mean power per channel.

Across sessions:
- Trial-wise data are pooled by cognitive state.
- Pairwise permutation tests (shuffling trials) are performed between states:
    • at the single-channel level,
    • at the array level (channels grouped into 6 arrays),
    • and at a combined-array level (arrays 1 to 3 merged, others separate).
- Max/min-based permutation thresholds are used to control for multiple
  comparisons across time or frequency.

Outputs:
- Permutation statistics, significance masks, and summary measures are saved
  for each channel and array.
- Diagnostic plots show state differences and statistically significant
  time/frequency regions.
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
output_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_lfp/RT_aligned/all_trials/erp_spectra'
results_dir = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp'
results_data_dir = os.path.join(results_dir,"RT_aligned", "all_trials", "erp_spectra")

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

N_STATES_TO_USE = 4
n_perms = 1000
alpha = 0.05
rng = np.random.default_rng(42)

# RT-centering parameters
pre_rt = 0.45   # seconds before RT
post_rt = 0.45  # seconds after RT

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
    T = nsamples / fs                       # duration in seconds
    NW = tapsmofrq * T / 2                  # time-bandwidth product
    K = max(int(2 * NW - 1), 1)             # number of tapers
    tapers = dpss(nsamples, NW, Kmax=K)     # (K, nsamples)

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
    Compute power spectrum for each trial and channel, matching the original
    syncopy approach:
      - 2–30 Hz:  single Hann taper (mtmfft, taper='hann')
      - 30–100 Hz: DPSS multitaper with 4 Hz smoothing (mtmfft, tapsmofrq=4)
    Results are concatenated along the frequency axis.

    Parameters
    ----------
    trials_3d : ndarray, shape (nTrials, nSamples, nChannels)
    fs : float, sampling rate

    Returns
    -------
    power : ndarray, shape (nTrials, nFreqs, nChannels)
    freqs_combined : ndarray, concatenated frequency vector
    """
    ntrials, nsamples, nchan = trials_3d.shape

    # Determine output frequency axes (use dummy signal)
    dummy = np.zeros(nsamples)
    freqs_low, _ = _periodogram_hann(dummy, fs, (2, 30))
    freqs_high, _ = _periodogram_multitaper(dummy, fs, (30, 100), tapsmofrq=4)
    freqs_combined = np.concatenate((freqs_low, freqs_high))
    n_low, n_high = len(freqs_low), len(freqs_high)

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
    emissions_path = os.path.join(processed_dir, session_folders[session_name], 'emissions.npy')

    if not os.path.exists(lfp_path) or not os.path.exists(trial_info_path):
        continue
    if not os.path.exists(emissions_path):
        print(f"  emissions.npy not found for {session_name}, skipping")
        continue

    # load RT data
    rt_values = np.load(emissions_path).flatten()

    # state info
    predicted_states = session_to_probs[session_name]
    trial_info_df = pd.read_pickle(trial_info_path)
    trial_info_df.iloc[:, 0] = (trial_info_df.iloc[:, 0] - 1000).astype('Int64')
    stim_df = pd.DataFrame({
        'TrialIndex': np.arange(len(predicted_states)),
        'States': predicted_states,
        'RT': rt_values[:len(predicted_states)]
    })
    combined_df = pd.merge(trial_info_df, stim_df, left_on='Trial_Number', right_on='TrialIndex', how='inner')

    # load LFP data (full length, no latency cut)
    datalfp = spy.load(lfp_path)
    ensure_trialindex_in_trialdefinition(datalfp)
    fs = datalfp.samplerate
    all_channels = list(datalfp.channel)

    # Get trial indices available in LFP
    lfp_trial_indices = datalfp.trialdefinition[:, 3].astype(int)
    states_trial_info_filt = combined_df[combined_df['TrialIndex'].isin(lfp_trial_indices)]
    unique_states = np.sort(np.unique(states_trial_info_filt['States'].to_numpy()))[:N_STATES_TO_USE]

    for state_value in unique_states:
        state_trials = states_trial_info_filt[states_trial_info_filt['States'] == state_value]

        rt_centered_trials = []
        for _, row in state_trials.iterrows():
            trial_idx = row['TrialIndex']
            rt = row['RT']

            # Find which LFP trial corresponds to this trial index
            lfp_trial_pos = np.where(lfp_trial_indices == trial_idx)[0]
            if len(lfp_trial_pos) == 0:
                continue
            lfp_trial_pos = lfp_trial_pos[0]

            # Get this trial's data and time vector
            trial_data = datalfp.trials[lfp_trial_pos]   # (nSamples, nChannels)
            trial_time = datalfp.time[lfp_trial_pos]      # (nSamples,)

            if np.all(np.isnan(trial_data)):
                continue

            # Find samples within [RT - pre_rt, RT + post_rt]
            t_start = rt - pre_rt
            t_end = rt + post_rt
            time_mask = (trial_time >= t_start) & (trial_time <= t_end)

            if np.sum(time_mask) < 10:  # need minimum samples
                continue

            segment = trial_data[time_mask, :]  # (nSamples_seg, nChannels)
            rt_centered_trials.append(segment)

        if not rt_centered_trials:
            continue

        # Only keep trials with at least 0.45s on both sides of RT
        expected_len = int(np.round((pre_rt + post_rt) * fs))
        seg_lengths = np.array([seg.shape[0] for seg in rt_centered_trials])
        n_before = len(rt_centered_trials)
        rt_centered_trials = [seg[:expected_len, :] for seg in rt_centered_trials
                              if seg.shape[0] >= expected_len]
        n_after = len(rt_centered_trials)
        n_discarded = n_before - n_after
        total_trials_kept += n_after
        total_trials_discarded += n_discarded
        print(f"  State {state_value}: kept {n_after}/{n_before} trials "
              f"(discarded {n_discarded} with <{expected_len} samples for "
              f"{pre_rt}+{post_rt}s window)")

        if not rt_centered_trials:
            continue

        trials_array = np.stack(rt_centered_trials, axis=0)  # (nTrials, nSamples, nChannels)

        # Create RT-centered time vector (time 0 = RT)
        time_vec = np.linspace(-pre_rt, post_rt, expected_len)

        # Remove all-NaN channels
        valid_ch_mask = ~np.all(np.isnan(trials_array), axis=(0, 1))
        valid_ch_idx = np.where(valid_ch_mask)[0]
        if len(valid_ch_idx) == 0:
            continue
        trials_array = trials_array[:, :, valid_ch_idx]
        valid_channels = [all_channels[i] for i in valid_ch_idx]

        # Remove all-NaN trials
        trial_mask = ~np.all(np.isnan(trials_array), axis=(1, 2))
        trials_array = trials_array[trial_mask]
        if trials_array.shape[0] == 0:
            continue

        print(f"  State {state_value}: {trials_array.shape[0]} trials, "
              f"{len(valid_channels)} channels, {expected_len} samples")

        # --- Spectra (2–100 Hz, periodogram with Hann window) ---
        power_trials, freqs_combined = compute_spectrum_trials(trials_array, fs)

        # --- FOOOF on mean spectrum per channel ---
        mean_spec = np.nanmean(power_trials, axis=0)  # (nFreqs, nChannels)
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
print(f"  Fraction kept:          {total_trials_kept / (total_trials_kept + total_trials_discarded):.1%}")

# -----------------------------
# Permutation tests (pairwise)
# -----------------------------
print("\n=== Running permutation tests across states ===")
perm_results_cache = {}  # (plot_type, s1, s2, i_arr, ch_name) → dict
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
                    # Trim to common sample length across sessions
                    min_samples = min(v.shape[1] for v in vals1 + vals2)
                    vals1 = [v[:, :min_samples] for v in vals1]
                    vals2 = [v[:, :min_samples] for v in vals2]
                    x_axis = x_axis[:min_samples]
                    data1 = np.concatenate(vals1, axis=0)
                    data2 = np.concatenate(vals2, axis=0)

                diff, sig, thr = permutation_test(data1, data2, n_perms=n_perms, alpha=alpha, rng=rng)

                # Save channel level results
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

                # Plotting
                ax.plot(x_axis, diff, color='k')
                ax.fill_between(x_axis, diff, where=sig, color='red', alpha=0.4)
                ax.axhline(0, color='gray', lw=0.5)
                ax.set_title(ch_name, fontsize=7)
                if plot_type == 'timelock':
                    ax.set_ylim(-15, 15)
            for j in range(len(ch_names), 36):
                axes[j].set_visible(False)
            fig.suptitle(f"{plot_type} {s1} vs {s2} - Array {i_arr+1} (RT-centered)")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fname = os.path.join(output_dir, f"perm_{plot_type}_pair{s1}_{s2}_array{i_arr+1}.pdf")
            fig.savefig(fname)
            plt.close(fig)

            # --- Array-level combined analysis ---
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
                # Trim to common sample length across sessions
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
                ax_arr.set_title(f"Array {i_arr+1} ({plot_type}) {s1} vs {s2} (RT-centered)")
                ax_arr.set_xlabel('Time rel. RT (s)' if plot_type == 'timelock' else 'Frequency (Hz)')
                ax_arr.set_ylabel('ΔAmplitude' if plot_type == 'timelock' else 'ΔResidual Power')
                if plot_type == 'timelock':
                    ax_arr.set_ylim(-15, 15)
                plt.tight_layout()
                fname_arr = os.path.join(
                    output_dir, f"perm_{plot_type}_pair{s1}_{s2}_ARRAYCOMBINED_array{i_arr+1}.pdf")
                fig_arr.savefig(fname_arr)
                plt.close(fig_arr)

                # save array level results
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
# array-level, grouped
# -----------------------------
print("\n=== Running permutation tests across states (array-level grouping) ===")
perm_results_cache = {}  # (plot_type, s1, s2, array_index) → dict
pairs = list(itertools.combinations(sorted(state_data_timelock.keys()), 2))

for plot_type, store in [('timelock', state_data_timelock),
                         ('spectra', state_data_spectra),
                         ('residual', state_data_residuals)]:
    if not store:
        continue

    # First session channels to define arrays
    first_channels = store[next(iter(store))][0]['channels']
    Sig_CH = np.array_split(first_channels, 6)  # split channels into 6 arrays

    for (s1, s2) in pairs:
        print(f"--> Testing pair ({s1} vs {s2}) for {plot_type}")
        for i_arr, ch_names in enumerate(Sig_CH):
            # --- Combine arrays 1, 2, 3 ---
            if i_arr < 3:
                # combine channels from array 1,2,3
                if i_arr == 0:
                    combined_ch_names = np.concatenate(Sig_CH[:3])
                else:
                    continue  # skip arrays 2 and 3 since they are merged into array 1
            else:
                combined_ch_names = ch_names  # keep arrays 4,5,6 separate

            # --- Collect trials across sessions ---
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

            # --- Run permutation test ---
            if vals1_array and vals2_array:
                # Trim to common sample length across sessions
                min_samples = min(v.shape[0] for v in vals1_array + vals2_array)
                vals1_array = [v[:min_samples] for v in vals1_array]
                vals2_array = [v[:min_samples] for v in vals2_array]
                x_axis = x_axis[:min_samples]
                data1_array = np.stack(vals1_array, axis=0)
                data2_array = np.stack(vals2_array, axis=0)
                diff_array, sig_array, thr_array = permutation_test(
                    data1_array, data2_array, n_perms=n_perms, alpha=alpha, rng=rng)

                # --- Plot array-level results ---
                fig_arr, ax_arr = plt.subplots(figsize=(6, 4))
                ax_arr.plot(x_axis, diff_array, color='k', lw=1.5)
                ax_arr.fill_between(x_axis, diff_array, where=sig_array, color='red', alpha=0.4)
                ax_arr.axhline(0, color='gray', lw=0.8)
                array_label = f"Array {i_arr+1}" if i_arr >= 3 else "Array 1-3"
                ax_arr.set_title(f"{array_label} ({plot_type}) {s1} vs {s2} (RT-centered)")
                ax_arr.set_xlabel('Time rel. RT (s)' if plot_type == 'timelock' else 'Frequency (Hz)')
                ax_arr.set_ylabel('ΔAmplitude' if plot_type == 'timelock' else 'ΔResidual Power')
                if plot_type == 'timelock':
                    ax_arr.set_ylim(-15, 15)
                plt.tight_layout()
                fname_arr = os.path.join(
                    output_dir, f"cb_perm_{plot_type}_pair{s1}_{s2}_{array_label.replace('-', '')}.pdf")
                fig_arr.savefig(fname_arr)
                plt.close(fig_arr)

                # --- Save array-level results ---
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
states = [s for s in range(N_STATES_TO_USE) if s != 1]  # exclude state 1 (can be added back later)
arrays_list = [1, 2, 3, 4, 5, 6]
state_colors = {
    0: (0.55, 0.0, 0.55),   # purple
    1: (0.0, 0.39, 0.39),   # teal
    2: (0.8, 0.33, 0.0),    # orange
    3: (0.25, 0.35, 0.55)   # slate blue
}
sig_color = '#8dd3c7'
teal_cmap = LinearSegmentedColormap.from_list('teal_map', ['white', '#1f9e89'])


def load_permdata(plot_type, s1, s2, array_index):
    """Load permutation test results for a given array and state pair."""
    fname = f"permdata_{plot_type}_pair{s1}_{s2}_ARRAY_array{array_index}.npz"
    fpath = os.path.join(results_data_dir, fname)
    if not os.path.exists(fpath):
        return None
    return np.load(fpath)


def load_permdata_merged(plot_type, s1, s2, array_label):
    """Load permutation data for merged array sets."""
    fname = f"cb_permdata_{plot_type}_pair{s1}_{s2}_{array_label.replace('-', '')}.npz"
    fpath = os.path.join(results_data_dir, fname)
    if not os.path.exists(fpath):
        return None
    return np.load(fpath)


# -----------------------------
# Figure 1: Real data + significance masks (per array)
# -----------------------------
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
                xlabel = 'Time rel. RT (s)' if plot_type == 'timelock' else 'Frequency (Hz)'
                ax.set_xlabel(xlabel)
            ax.legend(fontsize=6)

            summary_mask_array.append(sig_mask)
        summary_mask.append(summary_mask_array)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.25, hspace=0.25)
    plt.suptitle(f"{plot_type} - Mean across all sessions (RT-centered)", fontsize=14)
    plt.savefig(os.path.join(summary_output_dir, f"{plot_type}_all_arrays_pairs.pdf"))
    plt.close()

    # -----------------------------
    # Figure 2: Summary heatmap
    # -----------------------------
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
            ax.set_xlabel('Time rel. RT (s)' if plot_type == 'timelock' else 'Frequency (Hz)')
            if j_pair == 0:
                ax.set_ylabel('Arrays')
                ax.set_yticks(range(1, n_arrays_fig + 1))
                ax.set_yticklabels([str(a) for a in arrays_list])

        plt.subplots_adjust(left=0.05, right=0.88, top=0.88, bottom=0.12, wspace=0.3)
        cbar_ax = fig_sum.add_axes([0.90, 0.12, 0.02, 0.76])
        cbar = fig_sum.colorbar(im, cax=cbar_ax)
        cbar.set_label('Significant (1=pairwise)')

        plt.suptitle(f"{plot_type} - Pairwise significance across arrays (RT-centered)", fontsize=14)
        plt.savefig(os.path.join(summary_output_dir, f"{plot_type}_pairwise_summary_all_arrays.pdf"))
        plt.close()

# -----------------------------
# Figure 3: Merged Array plots (arrays 1-3 combined)
# -----------------------------
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
                xlabel = 'Time rel. RT (s)' if plot_type == 'timelock' else 'Frequency (Hz)'
                ax.set_xlabel(xlabel)
            ax.legend(fontsize=6)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.25, hspace=0.25)
    plt.suptitle(f"{plot_type} - Mean across all sessions, merged arrays (RT-centered)", fontsize=14)
    plt.savefig(os.path.join(summary_output_dir, f"{plot_type}_merged_arrays_pairs.pdf"))
    plt.close()

print(f"Summary plots saved under {summary_output_dir}")
