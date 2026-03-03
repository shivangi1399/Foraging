import os
import numpy as np
import matplotlib.pyplot as plt
from fooof import FOOOF
import syncopy as spy

# -------------------------
# User config
# -------------------------
lfp_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/clean_full_length'
output_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/timelock_spectra/200_600'
sessions = ['20230202', '20230203', '20230208', '20230209', '20230213', '20230214']
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Helper functions
# -------------------------
def get_channel_index(ch_list, ch_name):
    if isinstance(ch_list, list):
        return ch_list.index(ch_name)
    return int(np.where(np.array(ch_list) == ch_name)[0][0])

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

def extract_power_trials(freq_analysis, data):
    try:
        freqs = getattr(freq_analysis, 'freq', None)
        channels = getattr(freq_analysis, 'channel', None)
        if channels is None:
            channels = data.channel
        candidate = np.squeeze(freq_analysis.trials)
        if candidate.ndim == 3:
            return candidate, freqs, channels
        if candidate.ndim == 2:
            if len(freqs) * len(channels) == candidate.shape[1]:
                reshaped = candidate.reshape((candidate.shape[0], len(freqs), len(channels)))
                return reshaped, freqs, channels
        return None, None, None
    except Exception as e:
        print(f"extract_power_trials error: {e}")
        return None, None, None
    
def array_mean_sem(data_array, channel_indices):
    array_data = data_array[:, channel_indices]
    mean = np.nanmean(array_data, axis=1)
    sem = np.nanstd(array_data, axis=1) / np.sqrt(len(channel_indices))
    return mean, sem

# -------------------------
# Containers for session-averaged analysis
# -------------------------
all_timelocks = []
all_spectra = []
all_residuals = []

# -------------------------
# Main loop: process each session
# -------------------------
for session_name in sessions:
    lfp_path = os.path.join(lfp_data_dir, session_name, 'Cleaned_lfp_FT.spy')
    if not os.path.exists(lfp_path):
        print(f"Missing LFP for {session_name}")
        continue

    session_output_dir = os.path.join(output_dir, session_name)
    os.makedirs(session_output_dir, exist_ok=True)
    print(f"\n--- Processing session {session_name} ---")

    try:
        datalfp = spy.load(lfp_path)
    except Exception as e:
        print(f"Failed to load {lfp_path}: {e}")
        continue

    # Ensure trialdefinition has at least 4 columns
    try:
        if datalfp.trialdefinition.shape[1] < 4:
            nTrials = datalfp.trialdefinition.shape[0]
            datalfp.trialdefinition = np.hstack((datalfp.trialdefinition, np.arange(nTrials).reshape(-1, 1)))
    except Exception as e:
        print(f"Trialdefinition handling error: {e}")

    # Select latency window
    cfg = spy.StructDict(latency=[-0.2, 0.6])
    try:
        data = spy.selectdata(cfg, datalfp)
    except Exception as e:
        print(f"Latency select error: {e}")
        continue

    # Convert trials to array
    try:
        trials_array = np.array(data.trials)
    except Exception as e:
        print(f"Trials array error: {e}")
        continue

    if trials_array.size == 0:
        print(f"No trials for {session_name}")
        continue

    n_trials = trials_array.shape[0]
    print(f"Found {n_trials} trials for {session_name}")

    mean_timelock = np.nanmean(trials_array, axis=0)
    sem_timelock = np.nanstd(trials_array, axis=0) / np.sqrt(max(1, n_trials))
    variance_time = np.nanvar(trials_array, axis=0)
    Sig_CH = np.array_split(data.channel, 6)
    time_vec = data.time[0] if hasattr(data, 'time') and len(data.time) > 0 else np.arange(mean_timelock.shape[0])

    # -------------------------
    # Timelock plots
    # -------------------------
    for i in range(6):
        fig, axes = plt.subplots(6, 6, figsize=(15, 12))
        ch_names = Sig_CH[i]
        for ichan, ch_name in enumerate(ch_names):
            row, col = divmod(ichan, 6)
            ax = axes[row, col]
            try:
                ch_idx = get_channel_index(data.channel, ch_name)
                mean_trace = mean_timelock[:, ch_idx]
                sem_trace = sem_timelock[:, ch_idx]
                ax.plot(time_vec, mean_trace)
                ax.fill_between(time_vec, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.25)
                ax.axvline(0, linestyle='--', linewidth=0.7)
                ax.set_title(ch_name, fontsize=8)
            except Exception:
                ax.set_visible(False)
        for j in range(len(ch_names), 36):
            r, c = divmod(j, 6)
            axes[r, c].set_visible(False)
        plt.tight_layout()
        fig.savefig(os.path.join(session_output_dir, f"timelocklfp_alltrials_array_{i+1}.pdf"))
        plt.close(fig)

    # -------------------------
    # Variance plots
    # -------------------------
    for i in range(6):
        fig, axes = plt.subplots(6, 6, figsize=(15, 12))
        ch_names = Sig_CH[i]
        for ichan, ch_name in enumerate(ch_names):
            row, col = divmod(ichan, 6)
            ax = axes[row, col]
            try:
                ch_idx = get_channel_index(data.channel, ch_name)
                ax.plot(time_vec, variance_time[:, ch_idx])
                ax.axvline(0, linestyle='--', linewidth=0.7)
                ax.set_title(ch_name, fontsize=8)
            except Exception:
                ax.set_visible(False)
        for j in range(len(ch_names), 36):
            r, c = divmod(j, 6)
            axes[r, c].set_visible(False)
        plt.tight_layout()
        fig.savefig(os.path.join(session_output_dir, f"variance_alltrials_array_{i+1}.pdf"))
        plt.close(fig)

    # -------------------------
    # Hybrid spectral analysis
    # -------------------------
    print("Hybrid spectral analysis")
    spectra_output_dir = os.path.join(session_output_dir, 'spectra_plots')
    os.makedirs(spectra_output_dir, exist_ok=True)

    try:
        # MTMFFT 2–30 Hz - hann taper
        cfg_low = spy.StructDict(method='mtmfft', foilim=[2, 30], out='pow', keeptrials=True, taper='hann')
        freq_low = spy.freqanalysis(data, cfg_low)
        low_power, freqs_low, channels = extract_power_trials(freq_low, data)

        # MTMFFT 30–100 Hz - multitaper
        cfg_high = spy.StructDict(method='mtmfft', foilim=[30, 100], out='pow', keeptrials=True, tapsmofrq=4)
        freq_high = spy.freqanalysis(data, cfg_high)
        high_power, freqs_high, _ = extract_power_trials(freq_high, data)

        if low_power is None or high_power is None:
            print(f"Spectral extraction failed for {session_name}")
            continue

        power_trials = np.concatenate((low_power, high_power), axis=1)
        freqs_combined = np.concatenate((freqs_low, freqs_high))

        mean_spec = np.nanmean(power_trials, axis=0)
        sem_spec = np.nanstd(power_trials, axis=0) / np.sqrt(power_trials.shape[0])
        Sig_CH_spec = np.array_split(channels, 6)

        # Store residuals per session
        resid_session = np.zeros_like(mean_spec)

        # -------------------------
        # Plotting (Raw + FOOOF residual)
        # -------------------------
        for i in range(6):
            fig_raw, axes_raw = plt.subplots(6, 6, figsize=(15, 12))
            fig_resid, axes_resid = plt.subplots(6, 6, figsize=(15, 12))
            ch_names = Sig_CH_spec[i]
            for ichan, ch_name in enumerate(ch_names):
                row, col = divmod(ichan, 6)
                ax_raw, ax_resid = axes_raw[row, col], axes_resid[row, col]
                try:
                    ch_idx = get_channel_index(channels, ch_name)
                    spec_mean = mean_spec[:, ch_idx]
                    spec_sem = sem_spec[:, ch_idx]
                    if np.all(np.isnan(spec_mean)) or np.nanmax(spec_mean) == 0:
                        ax_raw.set_visible(False)
                        ax_resid.set_visible(False)
                        continue
                    # Raw spectra
                    ax_raw.plot(freqs_combined, spec_mean)
                    ax_raw.fill_between(freqs_combined, spec_mean - spec_sem, spec_mean + spec_sem, alpha=0.25)
                    ax_raw.axvline(30, linestyle='--', linewidth=0.7)
                    ax_raw.set_xlim([2, 100])
                    ax_raw.set_title(ch_name, fontsize=8)

                    # FOOOF residuals
                    freq_res = np.median(np.diff(freqs_combined))
                    fm = FOOOF(peak_width_limits=[max(2 * freq_res, 1.0), 12], max_n_peaks=6,
                               min_peak_height=0.05, peak_threshold=1.5, aperiodic_mode='fixed')
                    fm.fit(freqs_combined, spec_mean)
                    resid = fm._spectrum_flat
                    resid_session[:, ch_idx] = resid
                    ax_resid.plot(freqs_combined, resid)
                    ax_resid.axvline(30, linestyle='--', linewidth=0.7)
                    ax_resid.set_xlim([2, 100])
                    ax_resid.set_title(ch_name, fontsize=8)
                except Exception as e:
                    print(f"FOOOF failed {ch_name}: {e}")
                    ax_resid.set_visible(False)

            for j in range(len(ch_names), 36):
                r, c = divmod(j, 6)
                axes_raw[r, c].set_visible(False)
                axes_resid[r, c].set_visible(False)
            fig_raw.suptitle(f'Hybrid Spectra (mean ± SEM) - Array {i+1}')
            fig_resid.suptitle(f'Hybrid Residual (FOOOF) - Array {i+1}')
            fig_raw.tight_layout(rect=[0, 0, 1, 0.95])
            fig_resid.tight_layout(rect=[0, 0, 1, 0.95])
            fig_raw.savefig(os.path.join(spectra_output_dir, f'hybrid_spectra_array_{i+1}.pdf'))
            fig_resid.savefig(os.path.join(spectra_output_dir, f'hybrid_residual_array_{i+1}.pdf'))
            plt.close(fig_raw)
            plt.close(fig_resid)

        # -------------------------
        # Store session means for session-averaged analysis
        # -------------------------
        all_timelocks.append(mean_timelock)
        all_spectra.append(mean_spec)
        all_residuals.append(resid_session)

    except Exception as e:
        print(f"Hybrid spectral analysis failed: {e}")

    print(f"Finished session {session_name}. Outputs saved to {session_output_dir}")

# -------------------------
# Session-averaged analysis per array
# -------------------------
averages_dir = os.path.join(output_dir, 'averages')
os.makedirs(averages_dir, exist_ok=True)
print("\n--- Session-averaged analysis per array ---")

if all_timelocks:
    timelock_stack = np.stack(all_timelocks, axis=0)
    timelock_grand_mean = np.nanmean(timelock_stack, axis=0)
    timelock_grand_sem = np.nanstd(timelock_stack, axis=0) / np.sqrt(timelock_stack.shape[0])

    # Split channels into arrays
    arrays_timelock = np.array_split(data.channel, 6)

    for i, array_chs in enumerate(arrays_timelock, start=1):
        fig, axes = plt.subplots(6, 6, figsize=(15, 12))
        for ichan, ch_name in enumerate(array_chs):
            row, col = divmod(ichan, 6)
            ax = axes[row, col]
            try:
                ch_idx = get_channel_index(data.channel, ch_name)
                ax.plot(time_vec, timelock_grand_mean[:, ch_idx])
                ax.fill_between(time_vec,
                                timelock_grand_mean[:, ch_idx] - timelock_grand_sem[:, ch_idx],
                                timelock_grand_mean[:, ch_idx] + timelock_grand_sem[:, ch_idx],
                                alpha=0.25)
                ax.axvline(0, linestyle='--', linewidth=0.7)
                ax.set_title(ch_name, fontsize=8)
            except Exception:
                ax.set_visible(False)
        for j in range(len(array_chs), 36):
            r, c = divmod(j, 6)
            axes[r, c].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(averages_dir, f"timelock_grand_average_array_{i}.pdf"))
        plt.close(fig)

if all_spectra:
    spectra_stack = np.stack(all_spectra, axis=2)
    spectra_grand_mean = np.nanmean(spectra_stack, axis=2)
    spectra_grand_sem = np.nanstd(spectra_stack, axis=2) / np.sqrt(spectra_stack.shape[2])
    arrays_spec = np.array_split(data.channel, 6)

    for i, array_chs in enumerate(arrays_spec, start=1):
        fig, axes = plt.subplots(6, 6, figsize=(15, 12))
        for ichan, ch_name in enumerate(array_chs):
            row, col = divmod(ichan, 6)
            ax = axes[row, col]
            try:
                ch_idx = get_channel_index(data.channel, ch_name)
                ax.plot(freqs_combined, spectra_grand_mean[:, ch_idx])
                ax.fill_between(freqs_combined,
                                spectra_grand_mean[:, ch_idx] - spectra_grand_sem[:, ch_idx],
                                spectra_grand_mean[:, ch_idx] + spectra_grand_sem[:, ch_idx],
                                alpha=0.25)
                ax.axvline(30, linestyle='--', linewidth=0.7)
                ax.set_xlim([2, 100])
                ax.set_title(ch_name, fontsize=8)
            except Exception:
                ax.set_visible(False)
        for j in range(len(array_chs), 36):
            r, c = divmod(j, 6)
            axes[r, c].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(averages_dir, f"spectra_grand_average_array_{i}.pdf"))
        plt.close(fig)

if all_residuals:
    residuals_stack = np.stack(all_residuals, axis=2)
    residuals_grand_mean = np.nanmean(residuals_stack, axis=2)
    residuals_grand_sem = np.nanstd(residuals_stack, axis=2) / np.sqrt(residuals_stack.shape[2])
    arrays_resid = np.array_split(data.channel, 6)

    for i, array_chs in enumerate(arrays_resid, start=1):
        fig, axes = plt.subplots(6, 6, figsize=(15, 12))
        for ichan, ch_name in enumerate(array_chs):
            row, col = divmod(ichan, 6)
            ax = axes[row, col]
            try:
                ch_idx = get_channel_index(data.channel, ch_name)
                ax.plot(freqs_combined, residuals_grand_mean[:, ch_idx])
                ax.fill_between(freqs_combined,
                                residuals_grand_mean[:, ch_idx] - residuals_grand_sem[:, ch_idx],
                                residuals_grand_mean[:, ch_idx] + residuals_grand_sem[:, ch_idx],
                                alpha=0.25)
                ax.axvline(30, linestyle='--', linewidth=0.7)
                ax.set_xlim([2, 100])
                ax.set_title(ch_name, fontsize=8)
            except Exception:
                ax.set_visible(False)
        for j in range(len(array_chs), 36):
            r, c = divmod(j, 6)
            axes[r, c].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(averages_dir, f"residuals_grand_average_array_{i}.pdf"))
        plt.close(fig)

print("Session-averaged timelock, spectra, and residuals per array saved in 'averages' folder.")

# -------------------------
# Session-averaged analysis per array (with channel averages)
# -------------------------

averages_dir = os.path.join(output_dir, 'averages')
os.makedirs(averages_dir, exist_ok=True)
print("\n--- Session-averaged analysis per array (with channel averages) ---")

if all_timelocks:
    timelock_stack = np.stack(all_timelocks, axis=0)
    timelock_grand_mean = np.nanmean(timelock_stack, axis=0)
    timelock_grand_sem = np.nanstd(timelock_stack, axis=0) / np.sqrt(timelock_stack.shape[0])
    arrays_timelock = np.array_split(data.channel, 6)

    for i, array_chs in enumerate(arrays_timelock, start=1):
        # Channel-wise plots (existing)
        fig, axes = plt.subplots(6, 6, figsize=(15, 12))
        for ichan, ch_name in enumerate(array_chs):
            row, col = divmod(ichan, 6)
            ax = axes[row, col]
            try:
                ch_idx = get_channel_index(data.channel, ch_name)
                ax.plot(time_vec, timelock_grand_mean[:, ch_idx])
                ax.fill_between(time_vec,
                                timelock_grand_mean[:, ch_idx] - timelock_grand_sem[:, ch_idx],
                                timelock_grand_mean[:, ch_idx] + timelock_grand_sem[:, ch_idx],
                                alpha=0.25)
                ax.axvline(0, linestyle='--', linewidth=0.7)
                ax.set_title(ch_name, fontsize=8)
            except Exception:
                ax.set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(averages_dir, f"timelock_grand_average_array_{i}.pdf"))
        plt.close(fig)

        # Array-averaged across channels
        channel_indices = [get_channel_index(data.channel, ch) for ch in array_chs]
        mean_array, sem_array = array_mean_sem(timelock_grand_mean, channel_indices)
        fig_avg, ax_avg = plt.subplots(figsize=(8,4))
        ax_avg.plot(time_vec, mean_array)
        ax_avg.fill_between(time_vec, mean_array - sem_array, mean_array + sem_array, alpha=0.25)
        ax_avg.axvline(0, linestyle='--', linewidth=0.7)
        ax_avg.set_title(f'Timelock Grand Average - Array {i}', fontsize=10)
        plt.tight_layout()
        fig_avg.savefig(os.path.join(averages_dir, f"timelock_grand_average_array_{i}_channels_avg.pdf"))
        plt.close(fig_avg)

if all_spectra:
    spectra_stack = np.stack(all_spectra, axis=2)
    spectra_grand_mean = np.nanmean(spectra_stack, axis=2)
    spectra_grand_sem = np.nanstd(spectra_stack, axis=2) / np.sqrt(spectra_stack.shape[2])
    arrays_spec = np.array_split(data.channel, 6)

    for i, array_chs in enumerate(arrays_spec, start=1):
        # Channel-wise plots (existing)
        fig, axes = plt.subplots(6, 6, figsize=(15, 12))
        for ichan, ch_name in enumerate(array_chs):
            row, col = divmod(ichan, 6)
            ax = axes[row, col]
            try:
                ch_idx = get_channel_index(data.channel, ch_name)
                ax.plot(freqs_combined, spectra_grand_mean[:, ch_idx])
                ax.fill_between(freqs_combined,
                                spectra_grand_mean[:, ch_idx] - spectra_grand_sem[:, ch_idx],
                                spectra_grand_mean[:, ch_idx] + spectra_grand_sem[:, ch_idx],
                                alpha=0.25)
                ax.axvline(30, linestyle='--', linewidth=0.7)
                ax.set_xlim([2, 100])
                ax.set_title(ch_name, fontsize=8)
            except Exception:
                ax.set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(averages_dir, f"spectra_grand_average_array_{i}.pdf"))
        plt.close(fig)

        # Array-averaged across channels
        channel_indices = [get_channel_index(data.channel, ch) for ch in array_chs]
        mean_array, sem_array = array_mean_sem(spectra_grand_mean, channel_indices)
        fig_avg, ax_avg = plt.subplots(figsize=(8,4))
        ax_avg.plot(freqs_combined, mean_array)
        ax_avg.fill_between(freqs_combined, mean_array - sem_array, mean_array + sem_array, alpha=0.25)
        ax_avg.axvline(30, linestyle='--', linewidth=0.7)
        ax_avg.set_xlim([2,100])
        ax_avg.set_title(f'Spectra Grand Average - Array {i}', fontsize=10)
        plt.tight_layout()
        fig_avg.savefig(os.path.join(averages_dir, f"spectra_grand_average_array_{i}_channels_avg.pdf"))
        plt.close(fig_avg)

if all_residuals:
    residuals_stack = np.stack(all_residuals, axis=2)
    residuals_grand_mean = np.nanmean(residuals_stack, axis=2)
    residuals_grand_sem = np.nanstd(residuals_stack, axis=2) / np.sqrt(residuals_stack.shape[2])
    arrays_resid = np.array_split(data.channel, 6)

    for i, array_chs in enumerate(arrays_resid, start=1):
        # Channel-wise plots (existing)
        fig, axes = plt.subplots(6, 6, figsize=(15, 12))
        for ichan, ch_name in enumerate(array_chs):
            row, col = divmod(ichan, 6)
            ax = axes[row, col]
            try:
                ch_idx = get_channel_index(data.channel, ch_name)
                ax.plot(freqs_combined, residuals_grand_mean[:, ch_idx])
                ax.fill_between(freqs_combined,
                                residuals_grand_mean[:, ch_idx] - residuals_grand_sem[:, ch_idx],
                                residuals_grand_mean[:, ch_idx] + residuals_grand_sem[:, ch_idx],
                                alpha=0.25)
                ax.axvline(30, linestyle='--', linewidth=0.7)
                ax.set_xlim([2, 100])
                ax.set_title(ch_name, fontsize=8)
            except Exception:
                ax.set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(averages_dir, f"residuals_grand_average_array_{i}.pdf"))
        plt.close(fig)

        # Array-averaged across channels
        channel_indices = [get_channel_index(data.channel, ch) for ch in array_chs]
        mean_array, sem_array = array_mean_sem(residuals_grand_mean, channel_indices)
        fig_avg, ax_avg = plt.subplots(figsize=(8,4))
        ax_avg.plot(freqs_combined, mean_array)
        ax_avg.fill_between(freqs_combined, mean_array - sem_array, mean_array + sem_array, alpha=0.25)
        ax_avg.axvline(30, linestyle='--', linewidth=0.7)
        ax_avg.set_xlim([2,100])
        ax_avg.set_title(f'Residuals Grand Average - Array {i}', fontsize=10)
        plt.tight_layout()
        fig_avg.savefig(os.path.join(averages_dir, f"residuals_grand_average_array_{i}_channels_avg.pdf"))
        plt.close(fig_avg)

print("Session-averaged timelock, spectra, and residuals per array (with channel averages) saved in 'averages' folder.")
