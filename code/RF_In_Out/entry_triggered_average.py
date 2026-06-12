"""
RF-entry-triggered average (ETA) of the LFP, with a per-time-point significance
test telling us WHICH latencies carry an entry-evoked deflection.

This is the one-sample counterpart of erp_spectra_stats_rf_inout.py (which
compares two entry TYPES against each other) and the direct analog of
saccades/saccade_lfp/saccade_triggered_average.py -- only the trigger differs:
here the event is an RF ENTRY (a stimulus entering the channel's receptive
field) rather than a saccade.

Method
------
1. Derive the per-channel RF state on the LFP time axis and detect ENTRIES
   (prev_state -> new_state held for >= MIN_DWELL_S), exactly as in
   erp_spectra_stats_rf_inout.py. Slice LFP [-PRE, +POST] around each entry.
   Entry types:
       (0,1) target_from_background      (2,1) target_from_distractor
       (0,2) distractor_from_background  (1,2) distractor_from_target
   Plus a pooled 'ALL_entries' group.
2. Baseline-correct each epoch on a pre-entry window (BASELINE_WIN).
3. ETA = mean epoch (per channel, and per array = pooled over channels).
4. Significance via a TRIGGER-SHIFT permutation with max-statistic correction
   across time (identical to the saccade STA): a null bank of epochs locked to
   RANDOM in-trial times where RF data exists (not entries) is built once; each
   permutation averages N of them (N = number of real entry epochs) and stores
   the max |mean| across time. A time point is significant if |ETA| exceeds the
   (1-alpha) quantile of that max-null -- family-wise controlled over the whole
   window, two-sided. Keeps the LFP's full temporal structure intact and asks
   directly whether locking to real entries beats locking to random times.

State derivation matches RF_In_Out/erp_spectra_stats_rf_inout.py.
Run in the warping env (needs parse_logfile, syncopy, h5py).
"""

# -----------------------------
# Imports
# -----------------------------
import os
import sys
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import syncopy as spy

sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog  # noqa: E402

# -----------------------------
# User Config
# -----------------------------
lfp_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/clean_full_length'
eye_data_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data'
rf_stim_dir  = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping_no_reset/RFarea_stim'

sessions = ['20230203', '20230208', '20230209', '20230213', '20230214']

session_logfiles = {
    '20230203': '2023_02_03-11_35_57_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230208': '2023_02_08-10_58_17_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230209': '2023_02_09-11_19_51_Cosmos_LeafForaging_001_KAS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230213': '2023_02_13-11_13_43_Cosmos_LeafForaging_002_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230214': '2023_02_14-11_42_27_Cosmos_LeafForaging_001_PAF_GrassyLandscapeWithBackgroundDark_Cont.log',
}

n_stimuli = 5
stim_name = 'ImageStimulus'

# ---- epoch window (s) around the entry ----
# POST and MIN_DWELL are capped by RF dwell (short), so they stay at 0.05 s.
# PRE is unconstrained (it only needs LFP before the entry), so it is lengthened
# to give a longer, more stable baseline and more pre-entry context.
PRE         = 0.15     # s, pre-entry (covers the baseline; free to lengthen)
POST        = 0.05     # s, post-entry (the entry-evoked response)
MIN_DWELL_S = 0.05     # new state must hold this long after entry (== POST so the
                       # post window stays in-state)

# baseline window (s, relative to entry); ends before 0 to avoid the transition.
BASELINE_WIN = (-0.15, -0.05)

ENTRY_TYPES = {
    (0, 1): 'target_from_background',
    (2, 1): 'target_from_distractor',
    (0, 2): 'distractor_from_background',
    (1, 2): 'distractor_from_target',
}
STATE_LABELS = {0: 'background', 1: 'target_in', 2: 'distractor_in'}
ENTRY_LABELS = list(ENTRY_TYPES.values())
# groups to average + test: each entry type, plus all entries pooled.
ETA_GROUPS = ENTRY_LABELS + ['ALL_entries']

n_perms = 1000
alpha   = 0.05
# random in-trial triggers (during RF-tracked periods) per (trial, channel) for
# the trigger-shift null bank.
N_NULL_TRIGGERS_PER_TRIAL = 20
rng = np.random.default_rng(42)

output_dir = ('/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/RF VR mapping_no_reset/'
              f'RF_In_Out/entry_triggered_average_{int(PRE*1000)}_{int(POST*1000)}')
results_data_dir = os.path.join(output_dir, 'data')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_data_dir, exist_ok=True)

# -----------------------------
# Helpers -- shared with erp_spectra_stats_rf_inout.py
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


def parse_session_log(session):
    """Per-trial timing/identity info needed to look up RF state."""
    log_path = os.path.join(eye_data_dir, session, session_logfiles[session])
    print(f'  Parsing log: {log_path}')
    stim_ts = []
    with TextLog(log_path) as log:
        log.make_id_struct()
        evt, ts, _, _ = log.parse_eventmarkers()
        indx = [ii for ii, name in enumerate(log.all_ids['name'])
                if name.startswith(stim_name)]
        for ii, istim in enumerate(indx):
            if ii + n_stimuli == len(indx):
                break
            this_id = log.all_ids[istim]
            next_id = log.all_ids[indx[ii + n_stimuli]]
            _, pos_ts = log.parse_spherical(
                obj_id=this_id['id'], st=this_id['start'], end=next_id['start'])
            stim_ts.append(pos_ts)
    with TextLog(log_path) as log:
        trial_data = log.get_info_per_trial(return_eventmarkers=True, return_loc=False)
    trial_df = pd.DataFrame(trial_data).sort_values('TrialIndex').reset_index(drop=True)
    target_stim = np.where(trial_df['Right'].values == 1, 'A', 'B')
    target_onset = ts[np.where(evt == 3011)[0]]
    reach_mask   = np.isin(evt, [3013, 3023])
    reach_ts_all  = ts[reach_mask]
    reach_evt_all = evt[reach_mask]
    n_log_trials = len(target_onset)
    n_stim_trials = len(stim_ts) // n_stimuli
    aligned_stim_times = []
    for itrl in range(n_stim_trials):
        t = stim_ts[itrl * n_stimuli + 3].T
        aligned_stim_times.append(t - t[0])
    return {
        'target_onset':       target_onset,
        'target_stim':        target_stim,
        'reach_ts_all':       reach_ts_all,
        'reach_evt_all':      reach_evt_all,
        'aligned_stim_times': aligned_stim_times,
        'n_log_trials':       n_log_trials,
    }


def trial_channel_rf_states(hf, trial_name, ch_indices, target_stim,
                            target_onset_t, reach_ts_all, reach_evt_all,
                            aligned_stim_t, lfp_time):
    """Per-channel RF state at every LFP time sample for one trial.
    0 = neither, 1 = target_in, 2 = distractor_in. -1 means no RF data."""
    n_t  = len(lfp_time)
    n_ch = len(ch_indices)
    states = np.full((n_t, n_ch), -1, dtype=np.int8)
    if trial_name not in hf:
        return states
    trl_grp  = hf[trial_name]
    tp_names = sorted(trl_grp.keys(), key=lambda n: int(n.split('_')[-1]))
    n_tp = min(len(tp_names), len(aligned_stim_t))
    if n_tp == 0:
        return states
    tp_times  = aligned_stim_t[:n_tp]
    tp_states = np.full((n_ch, n_tp), -1, dtype=np.int8)
    ch_indices_int = [int(c) for c in ch_indices]

    for tp_i in range(n_tp):
        tp_grp = trl_grp[tp_names[tp_i]]
        is_collapse = bool(tp_grp.attrs.get('collapsed_case', False))
        collapse_reached_stim = None
        if is_collapse and len(reach_ts_all) > 0:
            abs_tp_time = target_onset_t + aligned_stim_t[tp_i]
            nearest_idx = np.argmin(np.abs(reach_ts_all - abs_tp_time))
            collapse_reached_stim = 'A' if reach_evt_all[nearest_idx] == 3013 else 'B'

        ch_to_ptname = {}
        for nm in tp_grp.keys():
            if not nm.startswith('Point_'):
                continue
            try:
                ch_to_ptname[int(nm.split('_', 2)[1]) - 1] = nm
            except (ValueError, IndexError):
                continue

        if is_collapse:
            if collapse_reached_stim == 'A':
                collapse_in_target   = (target_stim == 'A')
                collapse_in_distract = (target_stim == 'B')
            elif collapse_reached_stim == 'B':
                collapse_in_target   = (target_stim == 'B')
                collapse_in_distract = (target_stim == 'A')
            else:
                collapse_in_target = collapse_in_distract = False

        for c_i, ch_idx in enumerate(ch_indices_int):
            pt_name = ch_to_ptname.get(ch_idx)
            if pt_name is None:
                continue
            pt_grp = tp_grp[pt_name]
            if is_collapse:
                in_target, in_distract = collapse_in_target, collapse_in_distract
            else:
                in_A = bool(pt_grp['inside_transformed_outline_A'][()])
                in_B = bool(pt_grp['inside_transformed_outline_B'][()])
                if target_stim == 'A':
                    in_target   = in_A
                    in_distract = in_B and not in_A
                elif target_stim == 'B':
                    in_target   = in_B
                    in_distract = in_A and not in_B
                else:
                    in_target = in_distract = False
            if in_target:
                tp_states[c_i, tp_i] = 1
            elif in_distract:
                tp_states[c_i, tp_i] = 2
            else:
                tp_states[c_i, tp_i] = 0

    valid_mask = lfp_time >= tp_times[0]
    if valid_mask.any():
        idx = np.searchsorted(tp_times, lfp_time[valid_mask], side='right') - 1
        idx = np.clip(idx, 0, n_tp - 1)
        states[valid_mask, :] = tp_states[:, idx].T
    return states


def find_runs(s):
    """Runs of valid (>=0) states. Returns list of (start, end_excl, value)."""
    runs = []
    n = len(s)
    i = 0
    while i < n:
        if s[i] < 0:
            i += 1
            continue
        v = int(s[i])
        j = i + 1
        while j < n and s[j] == v:
            j += 1
        runs.append((i, j, v))
        i = j
    return runs


# -----------------------------
# ETA statistics -- shared with saccade_triggered_average.py
# -----------------------------
def baseline_correct(epochs, epoch_t, base_win):
    bmask = (epoch_t >= base_win[0]) & (epoch_t < base_win[1])
    if not bmask.any():
        return epochs
    base = np.nanmean(epochs[:, bmask], axis=1, keepdims=True)
    return epochs - base


def eta_trigger_shift(real_epochs, null_bank, n_perms=1000, alpha=0.05, rng=None):
    """One-sample ETA significance via a trigger-shift (random-trigger) null.

    real_epochs : (N, nTime) baseline-corrected, locked to real RF entries.
    null_bank   : (M, nTime) baseline-corrected, locked to RANDOM in-trial times.
    Each permutation averages N epochs drawn (with replacement) from the null
    bank -- an ETA at random triggers -- and the max |.| across time forms the
    null. Two-sided: significant where |ETA| exceeds the (1-alpha) quantile.
    """
    eta = np.nanmean(real_epochs, axis=0)
    N = real_epochs.shape[0]
    M = null_bank.shape[0]
    max_dist = np.empty(n_perms)
    for i in range(n_perms):
        idx = rng.integers(0, M, size=N)
        null_eta = np.nanmean(null_bank[idx], axis=0)
        max_dist[i] = np.nanmax(np.abs(null_eta))
    thr = float(np.percentile(max_dist, 100 * (1 - alpha)))
    return eta, np.abs(eta) > thr, thr


# -----------------------------
# Main: per-session epoch extraction
# -----------------------------
session_epochs = []
n_entries_used = {et: 0 for et in ENTRY_LABELS}
epoch_t_ref = None

for session_name in sessions:
    print(f'\n=== Processing session {session_name} ===')
    lfp_path = os.path.join(lfp_data_dir, session_name, 'Cleaned_lfp_FT.spy')
    rf_path  = os.path.join(rf_stim_dir, session_name, 'RF_stim_collapse.h5')
    if not (os.path.exists(lfp_path) and os.path.exists(rf_path)):
        print('  missing LFP or RF file, skipping')
        continue

    datalfp = spy.load(lfp_path)
    ensure_trialindex_in_trialdefinition(datalfp)
    fs = float(datalfp.samplerate)
    pre_samples  = int(round(PRE  * fs))
    post_samples = int(round(POST * fs))
    epoch_len = pre_samples + post_samples
    epoch_t = (np.arange(epoch_len) - pre_samples) / fs
    min_dwell_samples = int(round(MIN_DWELL_S * fs))
    if epoch_t_ref is None:
        epoch_t_ref = epoch_t

    data_clean, valid_channels, valid_ch_idx = remove_nan_trials_channels(datalfp)
    if data_clean is None:
        print('  all-NaN data, skipping')
        continue
    sel_trials = data_clean.trialdefinition[:, 3].astype(int)
    n_trials = len(data_clean.trials)

    log_info = parse_session_log(session_name)
    n_max_trial = min(log_info['n_log_trials'], len(log_info['aligned_stim_times']))

    epochs_this = {}        # {(c_i, entry_type): list[np.array]}
    null_this   = {}        # {c_i: list[np.array]}  random-trigger null epochs
    n_used_sess = {et: 0 for et in ENTRY_LABELS}

    print(f'  Scanning {n_trials} trials for RF entries (min dwell {MIN_DWELL_S*1000:.0f} ms) ...')
    try:
        hf = h5py.File(rf_path, 'r')
        trial_names = sorted(hf.keys(), key=lambda n: int(n.split('_')[-1]))
    except (OSError, RuntimeError) as e:
        print(f'  !! cannot read RF HDF5 ({e.__class__.__name__}: {e}); skipping')
        try:
            hf.close()
        except Exception:
            pass
        continue

    with hf:
        for tr_i in range(n_trials):
            trl = sel_trials[tr_i]
            if trl >= n_max_trial or trl >= len(trial_names):
                continue
            trial_lfp  = data_clean.trials[tr_i]   # (n_t, n_ch)
            trial_time = data_clean.time[tr_i]
            n_t = trial_lfp.shape[0]

            states_t = trial_channel_rf_states(
                hf, trial_names[trl],
                [int(c) for c in valid_ch_idx],
                log_info['target_stim'][trl],
                log_info['target_onset'][trl],
                log_info['reach_ts_all'],
                log_info['reach_evt_all'],
                log_info['aligned_stim_times'][trl],
                trial_time,
            )  # (n_t, n_ch)

            for c_i in range(states_t.shape[1]):
                s = states_t[:, c_i]
                runs = find_runs(s)

                # --- real entry epochs ---
                for k in range(1, len(runs)):
                    p_start, p_end, p_v = runs[k-1]
                    c_start, c_end, c_v = runs[k]
                    if c_start != p_end:
                        continue                  # gap (-1) between runs
                    et = ENTRY_TYPES.get((p_v, c_v))
                    if et is None:
                        continue
                    if (c_end - c_start) < min_dwell_samples:
                        continue
                    win_start = c_start - pre_samples
                    win_end   = c_start + post_samples
                    if win_start < 0 or win_end > n_t:
                        continue
                    epoch = np.asarray(trial_lfp[win_start:win_end, c_i], dtype=np.float32)
                    if np.any(np.isnan(epoch)):
                        continue
                    epochs_this.setdefault((c_i, et), []).append(epoch)
                    n_entries_used[et] += 1
                    n_used_sess[et]    += 1

                # --- random in-trial null triggers (during RF-tracked periods) ---
                hi = n_t - post_samples
                valid_pos = np.where(s[pre_samples:hi] >= 0)[0] + pre_samples if hi > pre_samples else np.array([], dtype=int)
                if valid_pos.size:
                    k_draw = min(N_NULL_TRIGGERS_PER_TRIAL, valid_pos.size)
                    for p in rng.choice(valid_pos, size=k_draw, replace=False):
                        epoch = np.asarray(trial_lfp[p-pre_samples:p+post_samples, c_i],
                                           dtype=np.float32)
                        if np.any(np.isnan(epoch)):
                            continue
                        null_this.setdefault(c_i, []).append(epoch)

    epochs_arr = {key: np.stack(v, axis=0) for key, v in epochs_this.items() if v}
    null_arr = {c_i: np.stack(v, axis=0) for c_i, v in null_this.items() if v}
    session_epochs.append({'session': session_name, 'channels': valid_channels,
                           'epoch_t': epoch_t, 'epochs': epochs_arr, 'null': null_arr})
    print('  Entries this session - used:', n_used_sess)


# -----------------------------
# Counts summary
# -----------------------------
print('\n=== Entry-triggered epoch counts (pooled across sessions) ===')
for et in ENTRY_LABELS:
    print(f'  {et:28s}  used={n_entries_used[et]:>10d}')
if not session_epochs:
    raise RuntimeError('No sessions processed for epoch extraction.')


# -----------------------------
# Pooling helpers
# -----------------------------
ref_channels = session_epochs[0]['channels']
Sig_CH = np.array_split(ref_channels, 6)


def _channel_epochs_for_types(sd, c_i, group):
    """Epochs for one channel index, for an entry type or 'ALL_entries'."""
    types = ENTRY_LABELS if group == 'ALL_entries' else [group]
    arrs = [sd['epochs'].get((c_i, t)) for t in types]
    arrs = [a for a in arrs if a is not None and a.size]
    return np.concatenate(arrs, axis=0) if arrs else None


def collect_channel_epochs(ch_name, group):
    vals = []
    for sd in session_epochs:
        if ch_name not in sd['channels']:
            continue
        a = _channel_epochs_for_types(sd, sd['channels'].index(ch_name), group)
        if a is not None:
            vals.append(a)
    return np.concatenate(vals, axis=0) if vals else None


def collect_array_epochs(ch_names, group):
    vals = []
    for sd in session_epochs:
        for c in ch_names:
            if c not in sd['channels']:
                continue
            a = _channel_epochs_for_types(sd, sd['channels'].index(c), group)
            if a is not None:
                vals.append(a)
    return np.concatenate(vals, axis=0) if vals else None


def collect_channel_null(ch_name):
    vals = []
    for sd in session_epochs:
        if ch_name not in sd['channels']:
            continue
        a = sd['null'].get(sd['channels'].index(ch_name))
        if a is not None and a.size:
            vals.append(a)
    return np.concatenate(vals, axis=0) if vals else None


def collect_array_null(ch_names):
    vals = []
    for sd in session_epochs:
        for c in ch_names:
            if c not in sd['channels']:
                continue
            a = sd['null'].get(sd['channels'].index(c))
            if a is not None and a.size:
                vals.append(a)
    return np.concatenate(vals, axis=0) if vals else None


# -----------------------------
# ETA + significance per entry group
# -----------------------------
x_axis = epoch_t_ref

for group in ETA_GROUPS:
    print(f'\n=== ETA + trigger-shift significance: {group} ===')

    for i_arr, ch_names in enumerate(Sig_CH):
        # ---- per channel (6x6 grid) ----
        fig, axes = plt.subplots(6, 6, figsize=(15, 12))
        axes = axes.flatten()
        for ichan, ch_name in enumerate(ch_names):
            ax = axes[ichan]
            ep = collect_channel_epochs(ch_name, group)
            nb = collect_channel_null(ch_name)
            if ep is None or ep.shape[0] < 5 or nb is None or nb.shape[0] < 10:
                ax.set_title(f'{ch_name}\n(n/a)', fontsize=6)
                continue
            ep = baseline_correct(ep, x_axis, BASELINE_WIN)
            nb = baseline_correct(nb, x_axis, BASELINE_WIN)
            eta, sig, thr = eta_trigger_shift(ep, nb, n_perms=n_perms, alpha=alpha, rng=rng)

            npz_path = os.path.join(
                results_data_dir, f'eta_{group}_array{i_arr+1}_{ch_name}.npz')
            if not os.path.exists(npz_path):
                np.savez_compressed(npz_path, eta=eta, sig=sig, thr=thr,
                                    x_axis=x_axis, n_epochs=ep.shape[0],
                                    group=group, ch_name=ch_name, array_index=i_arr+1)

            ax.plot(x_axis, eta, color=(0.55, 0.0, 0.55), lw=0.8)
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
        fig.suptitle(f'Entry-triggered average [{group}] - Array {i_arr+1} '
                     f'(shaded: |ETA| > max-stat thr, a={alpha})')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(output_dir, f'eta_{group}_array{i_arr+1}.pdf'))
        plt.close(fig)

        # ---- array level (pooled over the array's channels) ----
        ep_a = collect_array_epochs(ch_names, group)
        nb_a = collect_array_null(ch_names)
        if ep_a is None or ep_a.shape[0] < 5 or nb_a is None or nb_a.shape[0] < 10:
            continue
        ep_a = baseline_correct(ep_a, x_axis, BASELINE_WIN)
        nb_a = baseline_correct(nb_a, x_axis, BASELINE_WIN)
        eta_a, sig_a, thr_a = eta_trigger_shift(ep_a, nb_a, n_perms=n_perms, alpha=alpha, rng=rng)

        fig_a, ax_a = plt.subplots(figsize=(6, 4))
        ax_a.plot(x_axis, eta_a, color=(0.55, 0.0, 0.55), lw=1.5, label='ETA')
        ax_a.axhline(thr_a, color='gray', lw=0.8, ls=':', label='max-stat thr')
        ax_a.axhline(-thr_a, color='gray', lw=0.8, ls=':')
        ylo, yhi = ax_a.get_ylim()
        if sig_a.any():
            ax_a.fill_between(x_axis, ylo, yhi, where=sig_a,
                              color='#8dd3c7', alpha=0.4, zorder=0)
            ax_a.set_ylim(ylo, yhi)
        ax_a.axhline(0, color='gray', lw=0.8)
        ax_a.axvline(0, color='gray', lw=0.8, ls=':')
        ax_a.set_title(f'Array {i_arr+1} ETA [{group}] (n={ep_a.shape[0]} epochs)')
        ax_a.set_xlabel('Time from RF entry (s)')
        ax_a.set_ylabel('Baseline-corrected LFP')
        ax_a.legend(fontsize=8)
        plt.tight_layout()
        fig_a.savefig(os.path.join(
            output_dir, f'eta_{group}_ARRAYCOMBINED_array{i_arr+1}.pdf'))
        plt.close(fig_a)

        npz_path = os.path.join(
            results_data_dir, f'eta_{group}_ARRAY_array{i_arr+1}.npz')
        if not os.path.exists(npz_path):
            np.savez_compressed(npz_path, eta=eta_a, sig=sig_a, thr=thr_a,
                                x_axis=x_axis, n_epochs=ep_a.shape[0],
                                group=group, array_index=i_arr+1)

print(f'\nDone. Plots -> {output_dir}\n        Stats -> {results_data_dir}')
