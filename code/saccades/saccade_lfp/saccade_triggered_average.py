"""
Saccade-triggered average (STA) of the LFP, pooled over ALL non-micro saccades,
with a per-time-point significance test telling us WHICH time points carry a
saccade-evoked deflection (i.e. differ from the pre-saccade baseline).

Question
--------
"Is there a saccade-locked LFP response, and at which latencies?" -- the
one-sample counterpart of saccade_erp_rf_split.py (which compares two RF groups).

Method
------
1. Extract saccade-locked LFP epochs (reusing the detection + alignment logic of
   saccade_erp_rf_split.py), pooling every non-micro saccade -- no RF grouping.
   Epochs are produced for BOTH alignments:
     'onset'   -> locked to the eye movement itself,
     'landing' -> locked to fixation onset.
2. Baseline-correct each epoch by subtracting its mean over BASELINE_WIN (a
   pre-event window). Under the null "no saccade-evoked response", the
   baseline-corrected values then fluctuate around 0.
3. STA = mean epoch (per channel, and per array = pooled over the array's
   channels).
4. Significance via a TRIGGER-SHIFT permutation with max-statistic correction
   across time: a null bank of epochs locked to RANDOM in-trial times (not
   saccades) is built once; each permutation averages N of them (N = number of
   real epochs) and stores the maximum |mean| across time. A time point is
   significant if |STA| exceeds the (1-alpha) quantile of that max-null --
   family-wise controlled over the whole window, two-sided. This keeps the LFP's
   full temporal structure / autocorrelation intact and makes no symmetry
   assumption (cf. the circular-shift null in states_beh_stats.py): it asks
   directly whether locking to real saccades beats locking to random times.

Note: the null bank is alignment-independent (random times), so the SAME bank is
shared by the 'onset' and 'landing' STAs. The NEXT_SACC_GAP_MS filter optionally
removes saccades whose neighbour is closer than the post-window, limiting overlap
contamination of the real STA tail.

Run in the warping env (needs parse_logfile, time_conversion, syncopy, h5py-free).
"""

# -----------------------------
# Imports
# -----------------------------
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import syncopy as spy

sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions')
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/eyetracking')
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
import time_conversion as tc                 # noqa: E402
from parse_logfile import TextLog            # noqa: E402

# -----------------------------
# User Config
# -----------------------------
lfp_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/clean_full_length'
eye_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data'
sacc_npz_path = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/saccade_detection/stitched_sessions.npz'

sessions = ['20230203', '20230208', '20230209', '20230213', '20230214']

session_logfiles = {
    '20230203': '2023_02_03-11_35_57_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230208': '2023_02_08-10_58_17_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230209': '2023_02_09-11_19_51_Cosmos_LeafForaging_001_KAS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230213': '2023_02_13-11_13_43_Cosmos_LeafForaging_002_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230214': '2023_02_14-11_42_27_Cosmos_LeafForaging_001_PAF_GrassyLandscapeWithBackgroundDark_Cont.log',
}

# ---- epoch window (s) around the alignment event ----
PRE  = 0.2     # s, before the alignment event (also covers the baseline)
POST = 0.3     # s, after  the alignment event (covers the post-saccadic potential)

# ---- alignments to compute (both, per request) ----
ALIGN_EVENTS = ['onset', 'landing']

# ---- baseline window (s, relative to the alignment event) for the one-sample
# test. Ends before 0 to avoid the pre-saccadic motor ramp leaking in. For
# 'landing' alignment this pre-landing interval contains the saccade itself --
# documented limitation; widen/shift if it matters.
BASELINE_WIN = (-0.20, -0.05)

# ---- overlap filter: drop a saccade if its nearest neighbour (onset-to-onset)
# is closer than this, so the post window is not contaminated by the next
# saccade. 0.0 disables. 
NEXT_SACC_GAP_MS = 300

# microsaccade exclusion (same as saccade_erp_rf_split.py)
MIN_SACC_DUR_MS = 6.0
MICRO_AMP_DEG   = 1.0

n_perms = 1000
alpha   = 0.05
# random in-trial triggers drawn per trial to build the trigger-shift null bank;
# pooled across trials/sessions/channels the bank becomes large.
N_NULL_TRIGGERS_PER_TRIAL = 20
rng     = np.random.default_rng(42)

output_dir = ('/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/saccade_lfp/'
              f'saccade_triggered_average_{int(PRE*1000)}_{int(POST*1000)}')
results_data_dir = os.path.join(output_dir, 'data')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_data_dir, exist_ok=True)

# -----------------------------
# Helpers (LFP cleaning) -- shared with saccade_erp_rf_split.py
# -----------------------------
def remove_nan_trials_channels(datas):
    trial_mask = [not np.all(np.isnan(tr)) for tr in datas.trials]
    if not any(trial_mask):
        return None, [], None
    cfg = spy.StructDict(trials=np.where(trial_mask)[0])
    datas_clean = spy.selectdata(cfg, datas)
    n_ch = datas_clean.trials[0].shape[1]
    ch_all_nan = np.ones(n_ch, dtype=bool)
    for tr in datas_clean.trials:
        ch_all_nan &= np.all(np.isnan(tr), axis=0)
    valid_ch_idx = np.where(~ch_all_nan)[0]
    if len(valid_ch_idx) == 0:
        return None, [], None
    cfg = spy.StructDict(channel=valid_ch_idx)
    datas_clean = spy.selectdata(cfg, datas_clean)
    valid_channels = [datas.channel[i] for i in valid_ch_idx]
    return datas_clean, valid_channels, valid_ch_idx


def ensure_trialindex_in_trialdefinition(datalfp):
    if datalfp.trialdefinition.shape[1] < 4:
        nTrials = datalfp.trialdefinition.shape[0]
        datalfp.trialdefinition = np.hstack(
            (datalfp.trialdefinition, np.arange(nTrials).reshape(-1, 1)))


# -----------------------------
# Saccade helpers -- shared with saccade_erp_rf_split.py
# -----------------------------
def detect_saccades(sacc_npz, session, fs_eye):
    """(onset, offset) sample pairs of non-micro saccades (500 Hz / iRec rows)."""
    pred = np.nan_to_num(sacc_npz[f'{session}__pred_orig']).astype(int)
    x = sacc_npz[f'{session}__x_orig']
    y = sacc_npz[f'{session}__y_orig']
    nan_mask = sacc_npz[f'{session}__nan_mask']
    min_dur_samp = int(round(MIN_SACC_DUR_MS / 1000 * fs_eye))

    d = np.diff(pred)
    onsets  = np.where(d == 1)[0] + 1
    offsets = np.where(d == -1)[0] + 1
    if offsets.size and (onsets.size == 0 or offsets[0] < onsets[0]):
        offsets = offsets[1:]
    if onsets.size and (offsets.size == 0 or onsets[-1] > offsets[-1]):
        onsets = onsets[:-1]
    n = min(len(onsets), len(offsets))
    onsets, offsets = onsets[:n], offsets[:n]

    keep_on, keep_off = [], []
    for on, off in zip(onsets, offsets):
        if off - on < min_dur_samp:
            continue
        if nan_mask[on:off].any():
            continue
        if np.hypot(x[off] - x[on], y[off] - y[on]) < MICRO_AMP_DEG:
            continue
        keep_on.append(on)
        keep_off.append(off)
    return np.asarray(keep_on, dtype=int), np.asarray(keep_off, dtype=int)


def samples_to_log_time(session, samples):
    """500 Hz / iRec-row sample indices -> log/Unreal time (s); NaN past the file."""
    folder = os.path.join(eye_data_dir, session)
    log_path = os.path.join(folder, session_logfiles[session])
    eye_file = next(os.path.basename(f).replace('.csv', '')
                    for f in glob.glob(os.path.join(folder, '*.csv'))
                    if 'net.csv' not in os.path.basename(f))
    net_csv = os.path.join(folder, eye_file + 'net.csv')
    pos_csv = os.path.join(folder, eye_file + '.csv')
    log_irec_offset = tc.align_irec(log_path, net_csv)
    pos_t = pd.read_csv(pos_csv, usecols=[0]).to_numpy().ravel()
    valid = samples < len(pos_t)
    out = np.full(len(samples), np.nan)
    out[valid] = pos_t[samples[valid]] + log_irec_offset
    return out


def session_target_onset(session):
    """Per-trial stim-onset times (eventmarker 3011); LFP trial time is 0 here."""
    log_path = os.path.join(eye_data_dir, session, session_logfiles[session])
    with TextLog(log_path) as log:
        evt, ts, _, _ = log.parse_eventmarkers()
    return ts[np.where(evt == 3011)[0]]


# -----------------------------
# STA statistics
# -----------------------------
def baseline_correct(epochs, epoch_t, base_win):
    """Subtract each epoch's mean over the baseline window."""
    bmask = (epoch_t >= base_win[0]) & (epoch_t < base_win[1])
    if not bmask.any():
        return epochs
    base = np.nanmean(epochs[:, bmask], axis=1, keepdims=True)
    return epochs - base


def sta_trigger_shift(real_epochs, null_bank, n_perms=1000, alpha=0.05, rng=None):
    """One-sample STA significance via a trigger-shift (random-trigger) null.

    real_epochs : (N, nTime) baseline-corrected, locked to real saccades.
    null_bank   : (M, nTime) baseline-corrected, locked to RANDOM in-trial times.
    Each permutation averages N epochs drawn (with replacement) from the null
    bank -- an STA at random triggers -- and the max |.| across time forms the
    null. Two-sided: significant where |STA| exceeds the (1-alpha) quantile of
    that max-null. Keeps each epoch's full time course intact; nothing is
    flipped or scrambled.
    """
    sta = np.nanmean(real_epochs, axis=0)
    N = real_epochs.shape[0]
    M = null_bank.shape[0]
    max_dist = np.empty(n_perms)
    for i in range(n_perms):
        idx = rng.integers(0, M, size=N)
        null_sta = np.nanmean(null_bank[idx], axis=0)
        max_dist[i] = np.nanmax(np.abs(null_sta))
    thr = float(np.percentile(max_dist, 100 * (1 - alpha)))
    return sta, np.abs(sta) > thr, thr


# -----------------------------
# Main: per-session epoch extraction (all saccades pooled)
# -----------------------------
print('Loading saccade dataset ...')
sacc_npz = np.load(sacc_npz_path, allow_pickle=True)
fs_eye = int(sacc_npz['fs'])

session_epochs = []     # per session: {'channels', 'epoch_t', 'epochs'}
epoch_t_ref = None
n_events_used = {a: 0 for a in ALIGN_EVENTS}

for session_name in sessions:
    print(f'\n=== Processing session {session_name} ===')
    lfp_path = os.path.join(lfp_data_dir, session_name, 'Cleaned_lfp_FT.spy')
    if not os.path.exists(lfp_path):
        print('  missing LFP, skipping')
        continue
    if f'{session_name}__pred_orig' not in sacc_npz:
        print('  no saccade data, skipping')
        continue

    datalfp = spy.load(lfp_path)
    ensure_trialindex_in_trialdefinition(datalfp)
    fs = float(datalfp.samplerate)
    pre_samples  = int(round(PRE * fs))
    post_samples = int(round(POST * fs))
    epoch_len = pre_samples + post_samples
    epoch_t = (np.arange(epoch_len) - pre_samples) / fs
    if epoch_t_ref is None:
        epoch_t_ref = epoch_t

    data_clean, valid_channels, valid_ch_idx = remove_nan_trials_channels(datalfp)
    if data_clean is None:
        print('  all-NaN data, skipping')
        continue
    sel_trials = data_clean.trialdefinition[:, 3].astype(int)
    n_trials = len(data_clean.trials)

    target_onset = session_target_onset(session_name)
    n_log_trials = len(target_onset)

    onsets, offsets = detect_saccades(sacc_npz, session_name, fs_eye)
    on_log  = samples_to_log_time(session_name, onsets)
    off_log = samples_to_log_time(session_name, offsets)
    good = ~(np.isnan(on_log) | np.isnan(off_log))
    on_log, off_log = on_log[good], off_log[good]

    # optional overlap filter (onset-to-onset neighbour gap)
    if NEXT_SACC_GAP_MS > 0 and len(on_log) > 2:
        order = np.argsort(on_log)
        on_log, off_log = on_log[order], off_log[order]
        gaps = np.diff(on_log)
        keep = np.ones(len(on_log), dtype=bool)
        keep[:-1] &= gaps >= (NEXT_SACC_GAP_MS / 1000.0)
        on_log, off_log = on_log[keep], off_log[keep]
    print(f'  {len(on_log)} non-micro saccades '
          f'(>= {MIN_SACC_DUR_MS:.0f} ms, amp >= {MICRO_AMP_DEG:.0f} deg'
          f'{f", gap >= {NEXT_SACC_GAP_MS:.0f} ms" if NEXT_SACC_GAP_MS > 0 else ""})')

    epochs_this = {}        # {(align, c_i): list[np.array]}  real saccade epochs
    null_epochs_this = {}   # {c_i: list[np.array]}  random-trigger (null) epochs
    for tr_i in range(n_trials):
        trl = sel_trials[tr_i]
        if trl >= n_log_trials:
            continue
        trial_lfp  = data_clean.trials[tr_i]      # (n_t, n_ch)
        trial_time = data_clean.time[tr_i]
        n_t = trial_lfp.shape[0]

        on_rel  = on_log  - target_onset[trl]
        off_rel = off_log - target_onset[trl]
        in_trl = np.where((on_rel >= trial_time[0]) & (on_rel <= trial_time[-1]))[0]
        if in_trl.size == 0:
            continue

        on_smp  = np.clip(np.searchsorted(trial_time, on_rel[in_trl]),  0, n_t - 1)
        off_smp = np.clip(np.searchsorted(trial_time, off_rel[in_trl]), 0, n_t - 1)

        for s_on, s_off in zip(on_smp, off_smp):
            for align in ALIGN_EVENTS:
                align_smp = s_on if align == 'onset' else s_off
                win_start = align_smp - pre_samples
                win_end   = align_smp + post_samples
                if win_start < 0 or win_end > n_t:
                    continue
                for c_i in range(trial_lfp.shape[1]):
                    epoch = np.asarray(trial_lfp[win_start:win_end, c_i],
                                       dtype=np.float32)
                    if np.any(np.isnan(epoch)):
                        continue
                    epochs_this.setdefault((align, c_i), []).append(epoch)
                    n_events_used[align] += 1

        # random in-trial triggers for the (alignment-independent) null bank,
        # drawn only from trials that contributed real epochs.
        lo, hi = pre_samples, n_t - post_samples
        if hi > lo:
            for a_smp in rng.integers(lo, hi, size=N_NULL_TRIGGERS_PER_TRIAL):
                seg = trial_lfp[a_smp - pre_samples: a_smp + post_samples]
                for c_i in range(trial_lfp.shape[1]):
                    epoch = np.asarray(seg[:, c_i], dtype=np.float32)
                    if np.any(np.isnan(epoch)):
                        continue
                    null_epochs_this.setdefault(c_i, []).append(epoch)

    epochs_arr = {key: np.stack(v, axis=0) for key, v in epochs_this.items() if v}
    null_arr = {c_i: np.stack(v, axis=0) for c_i, v in null_epochs_this.items() if v}
    session_epochs.append({'session': session_name, 'channels': valid_channels,
                           'epoch_t': epoch_t, 'epochs': epochs_arr,
                           'null': null_arr})

print('\n=== Saccade-channel epoch counts (pooled across sessions) ===')
for a in ALIGN_EVENTS:
    print(f'  align={a:8s}  epochs(channel-events)={n_events_used[a]:>10d}')
if not session_epochs:
    raise RuntimeError('No sessions processed for epoch extraction.')


# -----------------------------
# Pooling helpers
# -----------------------------
ref_channels = session_epochs[0]['channels']
Sig_CH = np.array_split(ref_channels, 6)


def collect_channel_epochs(ch_name, align):
    """Pool epochs across sessions for one channel and alignment."""
    vals = []
    for sd in session_epochs:
        if ch_name not in sd['channels']:
            continue
        c_i = sd['channels'].index(ch_name)
        arr = sd['epochs'].get((align, c_i))
        if arr is not None and arr.size:
            vals.append(arr)
    return np.concatenate(vals, axis=0) if vals else None


def collect_array_epochs(ch_names, align):
    """Pool epochs across all channels-in-array and sessions, for one alignment."""
    vals = []
    for sd in session_epochs:
        for c in ch_names:
            if c not in sd['channels']:
                continue
            c_i = sd['channels'].index(c)
            arr = sd['epochs'].get((align, c_i))
            if arr is not None and arr.size:
                vals.append(arr)
    return np.concatenate(vals, axis=0) if vals else None


def collect_channel_null(ch_name):
    """Pool random-trigger (null) epochs across sessions for one channel."""
    vals = []
    for sd in session_epochs:
        if ch_name not in sd['channels']:
            continue
        c_i = sd['channels'].index(ch_name)
        arr = sd['null'].get(c_i)
        if arr is not None and arr.size:
            vals.append(arr)
    return np.concatenate(vals, axis=0) if vals else None


def collect_array_null(ch_names):
    """Pool random-trigger (null) epochs across channels-in-array and sessions."""
    vals = []
    for sd in session_epochs:
        for c in ch_names:
            if c not in sd['channels']:
                continue
            c_i = sd['channels'].index(c)
            arr = sd['null'].get(c_i)
            if arr is not None and arr.size:
                vals.append(arr)
    return np.concatenate(vals, axis=0) if vals else None


# -----------------------------
# STA + significance per alignment
# -----------------------------
x_axis = epoch_t_ref

for align in ALIGN_EVENTS:
    print(f'\n=== STA + trigger-shift significance: align={align} ===')
    xlabel = f'Time from saccade {align} (s)'

    for i_arr, ch_names in enumerate(Sig_CH):
        # ---- per channel (6x6 grid) ----
        fig, axes = plt.subplots(6, 6, figsize=(15, 12))
        axes = axes.flatten()
        for ichan, ch_name in enumerate(ch_names):
            ax = axes[ichan]
            ep = collect_channel_epochs(ch_name, align)
            nb = collect_channel_null(ch_name)
            if ep is None or ep.shape[0] < 5 or nb is None or nb.shape[0] < 10:
                ax.set_title(f'{ch_name}\n(n/a)', fontsize=6)
                continue
            ep = baseline_correct(ep, x_axis, BASELINE_WIN)
            nb = baseline_correct(nb, x_axis, BASELINE_WIN)
            sta, sig, thr = sta_trigger_shift(ep, nb, n_perms=n_perms, alpha=alpha, rng=rng)

            npz_path = os.path.join(
                results_data_dir, f'sta_{align}_array{i_arr+1}_{ch_name}.npz')
            if not os.path.exists(npz_path):
                np.savez_compressed(npz_path, sta=sta, sig=sig, thr=thr,
                                    x_axis=x_axis, n_epochs=ep.shape[0],
                                    align=align, ch_name=ch_name,
                                    array_index=i_arr + 1)

            ax.plot(x_axis, sta, color=(0.55, 0.0, 0.55), lw=0.8)
            ax.axhline(thr, color='gray', lw=0.4, ls=':')
            ax.axhline(-thr, color='gray', lw=0.4, ls=':')
            ylo, yhi = ax.get_ylim()
            if sig.any():
                ax.fill_between(x_axis, ylo, yhi, where=sig,
                                color='#8dd3c7', alpha=0.4, zorder=0)
                ax.set_ylim(ylo, yhi)
            ax.axhline(0, color='gray', lw=0.5)
            ax.axvline(0, color='gray', lw=0.5, ls=':')
            ax.set_title(f'{ch_name} (n={ep.shape[0]})', fontsize=6)
        for j in range(len(ch_names), 36):
            axes[j].set_visible(False)
        fig.suptitle(f'Saccade-triggered average ({align}) - Array {i_arr+1} '
                     f'(shaded: |STA| > max-stat thr, a={alpha})')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(output_dir, f'sta_{align}_array{i_arr+1}.pdf'))
        plt.close(fig)

        # ---- array level (pooled over the array's channels) ----
        ep_a = collect_array_epochs(ch_names, align)
        nb_a = collect_array_null(ch_names)
        if ep_a is None or ep_a.shape[0] < 5 or nb_a is None or nb_a.shape[0] < 10:
            continue
        ep_a = baseline_correct(ep_a, x_axis, BASELINE_WIN)
        nb_a = baseline_correct(nb_a, x_axis, BASELINE_WIN)
        sta_a, sig_a, thr_a = sta_trigger_shift(ep_a, nb_a, n_perms=n_perms, alpha=alpha, rng=rng)

        fig_a, ax_a = plt.subplots(figsize=(6, 4))
        ax_a.plot(x_axis, sta_a, color=(0.55, 0.0, 0.55), lw=1.5, label='STA')
        ax_a.axhline(thr_a, color='gray', lw=0.8, ls=':', label='max-stat thr')
        ax_a.axhline(-thr_a, color='gray', lw=0.8, ls=':')
        ylo, yhi = ax_a.get_ylim()
        if sig_a.any():
            ax_a.fill_between(x_axis, ylo, yhi, where=sig_a,
                              color='#8dd3c7', alpha=0.4, zorder=0)
            ax_a.set_ylim(ylo, yhi)
        ax_a.axhline(0, color='gray', lw=0.8)
        ax_a.axvline(0, color='gray', lw=0.8, ls=':')
        ax_a.set_title(f'Array {i_arr+1} STA ({align}, n={ep_a.shape[0]} epochs)')
        ax_a.set_xlabel(xlabel)
        ax_a.set_ylabel('Baseline-corrected LFP')
        ax_a.legend(fontsize=8)
        plt.tight_layout()
        fig_a.savefig(os.path.join(
            output_dir, f'sta_{align}_ARRAYCOMBINED_array{i_arr+1}.pdf'))
        plt.close(fig_a)

        npz_path = os.path.join(
            results_data_dir, f'sta_{align}_ARRAY_array{i_arr+1}.npz')
        if not os.path.exists(npz_path):
            np.savez_compressed(npz_path, sta=sta_a, sig=sig_a, thr=thr_a,
                                x_axis=x_axis, n_epochs=ep_a.shape[0],
                                align=align, array_index=i_arr + 1)

print(f'\nDone. Plots -> {output_dir}\n        Stats -> {results_data_dir}')
