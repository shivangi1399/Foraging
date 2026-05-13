"""
RF-entry-locked LFP analysis.

For every (trial, channel), step-fill the 60 Hz RF state onto the LFP time axis
and detect transitions where the new state holds for at least MIN_DWELL_S. Each
transition becomes an event with t = 0 at the entry sample. LFP is sliced
[-PRE, +POST] around the entry, pooled across (trial, channel, session) by
entry type, and compared with the same max/min permutation thresholding as
erp_spectra_stats.py.

Entry types (prev_state -> new_state):
    (0, 1)  target_from_neither
    (2, 1)  target_from_distractor
    (0, 2)  distractor_from_neither
    (1, 2)  distractor_from_target

Also reports:
- Dwell-time histogram per RF state (target / distractor / neither) computed
  across all (trial, channel) runs.

State derivation matches RF_In_Out/RF_inout_channel_raster.py.
"""

# -----------------------------
# Imports
# -----------------------------
import os
import sys
import itertools
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

output_dir       = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/RF VR mapping_no_reset/RF_In_Out/entry_locked'
results_data_dir = os.path.join(output_dir, 'data')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_data_dir, exist_ok=True)

sessions = ['20230203', '20230208', '20230209', '20230213', '20230214']

session_logfiles = {
    '20230203': '2023_02_03-11_35_57_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230208': '2023_02_08-10_58_17_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230209': '2023_02_09-11_19_51_Cosmos_LeafForaging_001_KAS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230213': '2023_02_13-11_13_43_Cosmos_LeafForaging_002_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230214': '2023_02_14-11_42_27_Cosmos_LeafForaging_001_PAF_GrassyLandscapeWithBackgroundDark_Cont.log',
}

n_stimuli   = 5
stim_name   = 'ImageStimulus'

PRE          = 0.2     # s, pre-entry window
POST         = 0.3     # s, post-entry window
MIN_DWELL_S  = 0.05    # 50 ms — new state must hold this long after entry

# If True, walk all transitions, generate diagnostic histograms (dwell + pre/post
# per entry type) and counts, then exit BEFORE epoch extraction / permutation
# tests. Use to pick PRE / POST. Set to False to run the full pipeline.
STOP_AFTER_HISTOGRAMS = False

ENTRY_TYPES = {
    (0, 1): 'target_from_neither',
    (2, 1): 'target_from_distractor',
    (0, 2): 'distractor_from_neither',
    (1, 2): 'distractor_from_target',
}
STATE_LABELS = {0: 'neither', 1: 'target_in', 2: 'distractor_in'}

n_perms = 1000
alpha   = 0.05
rng     = np.random.default_rng(42)


# -----------------------------
# Helpers
# -----------------------------
def remove_nan_trials_channels(datas):
    """Drop all-NaN trials, then drop channels that are all-NaN across every
    remaining trial. Handles variable-length trials (no np.stack)."""
    trial_mask = [not np.all(np.isnan(tr)) for tr in datas.trials]
    if not any(trial_mask):
        return None, [], None
    cfg = spy.StructDict(trials=np.where(trial_mask)[0])
    datas_clean = spy.selectdata(cfg, datas)

    # Per-channel: invalid only if every (trial, time) sample is NaN
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
            (datalfp.trialdefinition, np.arange(nTrials).reshape(-1, 1))
        )


def permutation_test(data1, data2, n_perms=1000, alpha=0.05, rng=None):
    n1, n2 = data1.shape[0], data2.shape[0]
    pooled = np.vstack([data1, data2])
    labels = np.array([0]*n1 + [1]*n2)
    real_diff = np.nanmean(data1, axis=0) - np.nanmean(data2, axis=0)
    max_dist = np.zeros(n_perms)
    min_dist = np.zeros(n_perms)
    for i in range(n_perms):
        rng.shuffle(labels)
        diff = np.nanmean(pooled[labels == 0], axis=0) - np.nanmean(pooled[labels == 1], axis=0)
        max_dist[i] = np.nanmax(diff)
        min_dist[i] = np.nanmin(diff)
    upper_thr = np.percentile(max_dist, 100 * (1 - alpha/2))
    lower_thr = np.percentile(min_dist, 100 * (alpha/2))
    sig_mask = (real_diff > upper_thr) | (real_diff < lower_thr)
    return real_diff, sig_mask, (lower_thr, upper_thr)


# -----------------------------
# Log + RF helpers
# -----------------------------
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
    0 = neither, 1 = target_in, 2 = distractor_in. -1 means no RF data.
    """
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

        # Build ch_idx → group-name map once per tp (was a full prefix scan per
        # (channel, tp), which dominated runtime).
        ch_to_ptname = {}
        for nm in tp_grp.keys():
            if not nm.startswith('Point_'):
                continue
            try:
                ch_to_ptname[int(nm.split('_', 2)[1]) - 1] = nm
            except (ValueError, IndexError):
                continue

        # Resolve collapse target/distractor membership once per tp
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
# Main: per-session epoch extraction
# -----------------------------
# session_epochs[i] = {
#   'channels':       [ch_name, ...],
#   'epoch_t':        np.array(epoch_len,),
#   'epochs':         {(c_i, entry_type): np.array(n_epochs, epoch_len)},
# }
session_epochs = []
dwell_durations = {0: [], 1: [], 2: []}
# Per-entry-type metadata for ALL detected transitions (independent of PRE/POST/MIN_DWELL):
#   prev_run_dur: duration of the prior state run (s)         → "available pre"
#   new_run_dur:  duration of the new state run (s) = dwell   → "available post"
#   max_pre:      seconds from trial start to entry sample    → trial-edge pre limit
#   max_post:     seconds from entry sample to trial end      → trial-edge post limit
entry_metadata = {et: {'prev_run_dur': [], 'new_run_dur': [],
                       'max_pre':      [], 'max_post':    []}
                  for et in ENTRY_TYPES.values()}
n_entries_total = {et: 0 for et in ENTRY_TYPES.values()}   # all detected transitions
n_entries_used  = {et: 0 for et in ENTRY_TYPES.values()}   # passing dwell + window filters
fs_ref = None
epoch_t_ref = None

for sess_idx, session_name in enumerate(sessions):
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
    epoch_len    = pre_samples + post_samples
    epoch_t      = (np.arange(epoch_len) - pre_samples) / fs
    min_dwell_samples = int(round(MIN_DWELL_S * fs))

    if fs_ref is None:
        fs_ref      = fs
        epoch_t_ref = epoch_t
    elif epoch_len != len(epoch_t_ref):
        print(f'  WARNING: epoch length differs from reference '
              f'({epoch_len} vs {len(epoch_t_ref)}); using session-local axis')

    data_clean, valid_channels, valid_ch_idx = remove_nan_trials_channels(datalfp)
    if data_clean is None:
        print('  all-NaN data, skipping')
        continue
    sel_trials = data_clean.trialdefinition[:, 3].astype(int)
    n_trials = len(data_clean.trials)

    log_info = parse_session_log(session_name)
    n_max_trial = min(log_info['n_log_trials'], len(log_info['aligned_stim_times']))

    epochs_this   = {}                       # {(c_i, entry_type): list[np.array]}
    n_total_sess  = {et: 0 for et in ENTRY_TYPES.values()}
    n_used_sess   = {et: 0 for et in ENTRY_TYPES.values()}

    print(f'  Scanning {n_trials} trials for RF entries (min dwell {MIN_DWELL_S*1000:.0f} ms) …')
    try:
        hf = h5py.File(rf_path, 'r')
        trial_names = sorted(hf.keys(), key=lambda n: int(n.split('_')[-1]))
    except (OSError, RuntimeError) as e:
        print(f'  !! cannot read RF HDF5 ({e.__class__.__name__}: {e}); '
              f'skipping session {session_name}')
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

                for start, end, v in runs:
                    dwell_durations[v].append((end - start) / fs)

                for k in range(1, len(runs)):
                    p_start, p_end, p_v = runs[k-1]
                    c_start, c_end, c_v = runs[k]
                    if c_start != p_end:
                        continue              # gap (-1) between runs
                    et = ENTRY_TYPES.get((p_v, c_v))
                    if et is None:
                        continue

                    # Record diagnostic metadata for every detected transition
                    entry_metadata[et]['prev_run_dur'].append((p_end - p_start) / fs)
                    entry_metadata[et]['new_run_dur' ].append((c_end - c_start) / fs)
                    entry_metadata[et]['max_pre'     ].append(c_start / fs)
                    entry_metadata[et]['max_post'    ].append((n_t - c_start) / fs)
                    n_entries_total[et] += 1
                    n_total_sess[et]    += 1

                    if STOP_AFTER_HISTOGRAMS:
                        continue              # skip epoch extraction in diagnostic mode

                    # Analysis filters
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

    if not STOP_AFTER_HISTOGRAMS:
        epochs_arr = {key: np.stack(v, axis=0) for key, v in epochs_this.items() if v}
        session_epochs.append({
            'session':  session_name,
            'channels': valid_channels,
            'epoch_t':  epoch_t,
            'fs':       fs,
            'epochs':   epochs_arr,
        })
    print('  Entries this session — total:', n_total_sess)
    if not STOP_AFTER_HISTOGRAMS:
        print('                        used:',  n_used_sess)


# -----------------------------
# Counts summary
# -----------------------------
print('\n=== Entry counts (pooled across sessions) ===')
print(f'  {"entry type":28s} {"detected":>10s}   {"used":>8s}')
for et in ENTRY_TYPES.values():
    print(f'  {et:28s} {n_entries_total[et]:>10d}   '
          f'{n_entries_used[et] if not STOP_AFTER_HISTOGRAMS else "—":>8}')


# -----------------------------
# Dwell-time histograms (per RF state)
# -----------------------------
print('\n=== Dwell-time histograms (per state) ===')
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, st in zip(axes, [1, 2, 0]):
    durs = np.array(dwell_durations[st]) if dwell_durations[st] else np.array([])
    if durs.size:
        ax.hist(durs, bins=60, range=(0, 3.0), color='steelblue', edgecolor='k', linewidth=0.4)
        med = np.median(durs)
        ax.axvline(med, color='red', linestyle='--', lw=1, label=f'median = {med*1000:.0f} ms')
        ax.legend(fontsize=8)
    ax.set_title(f'{STATE_LABELS[st]}  (n = {durs.size})')
    ax.set_xlabel('Dwell time (s)')
    ax.set_ylabel('Count')
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'dwell_time_histogram.pdf'))
plt.close(fig)


# -----------------------------
# Per-entry-type pre/post histograms
# -----------------------------
# 4 metrics × 4 entry types. Rows = metric, cols = entry type.
print('\n=== Pre/post-window histograms (per entry type) ===')
metrics = [
    ('prev_run_dur', 'Prior-state dwell (= available pre, s)'),
    ('new_run_dur',  'New-state dwell (= available post, s)'),
    ('max_pre',      'Max pre to trial start (s)'),
    ('max_post',     'Max post to trial end (s)'),
]
fig, axes = plt.subplots(len(metrics), len(ENTRY_TYPES), figsize=(20, 14))
for col, et in enumerate(ENTRY_TYPES.values()):
    for row, (metric, label) in enumerate(metrics):
        ax = axes[row, col]
        d = np.array(entry_metadata[et][metric])
        if d.size:
            ax.hist(d, bins=60, range=(0, 5.0), color='steelblue',
                    edgecolor='k', linewidth=0.4)
            med = np.median(d)
            p10, p90 = np.percentile(d, [10, 90])
            ax.axvline(med, color='red', linestyle='--', lw=1)
            ax.axvline(p10, color='orange', linestyle=':', lw=0.8)
            ax.axvline(p90, color='orange', linestyle=':', lw=0.8)
            stat_txt = (f'n={d.size}\nmed={med*1000:.0f} ms\n'
                        f'p10={p10*1000:.0f} ms\np90={p90*1000:.0f} ms')
            ax.text(0.98, 0.97, stat_txt, transform=ax.transAxes,
                    ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.85))
        if row == 0:
            ax.set_title(et, fontsize=10)
        if col == 0:
            ax.set_ylabel(label, fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
fig.suptitle('Available pre/post per entry type (red=median, orange=10/90th pctile)',
             fontsize=11, y=1.00)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'entry_window_histograms.pdf'))
plt.close(fig)

# Save raw arrays for downstream use
np.savez_compressed(
    os.path.join(results_data_dir, 'entry_window_metadata.npz'),
    **{f'{et}__{m}': np.array(entry_metadata[et][m])
       for et in ENTRY_TYPES.values() for m, _ in metrics},
    n_entries_total=np.array([n_entries_total[et] for et in ENTRY_TYPES.values()]),
    entry_types=np.array(list(ENTRY_TYPES.values())),
)
print(f'  Saved → {output_dir}/entry_window_histograms.pdf')

if STOP_AFTER_HISTOGRAMS:
    print('\nSTOP_AFTER_HISTOGRAMS=True → exiting before epoch extraction. '
          'Set to False once you have chosen PRE/POST.')
    sys.exit(0)

if not session_epochs:
    raise RuntimeError('No sessions processed for epoch extraction.')

np.savez_compressed(
    os.path.join(results_data_dir, 'dwell_durations.npz'),
    target_in     = np.array(dwell_durations[1]),
    distractor_in = np.array(dwell_durations[2]),
    neither       = np.array(dwell_durations[0]),
)
print('  Saved dwell-time histogram and raw durations.')


# -----------------------------
# Pooling helpers
# -----------------------------
ref_channels = session_epochs[0]['channels']
Sig_CH = np.array_split(ref_channels, 6)


def collect_channel_epochs(ch_name, entry_type):
    """Pool epochs across sessions for one channel and entry type."""
    vals = []
    for sd in session_epochs:
        if ch_name not in sd['channels']:
            continue
        c_i = sd['channels'].index(ch_name)
        arr = sd['epochs'].get((c_i, entry_type))
        if arr is None or arr.size == 0:
            continue
        vals.append(arr)
    if not vals:
        return None
    return np.concatenate(vals, axis=0)


def collect_array_session_means(ch_names, entry_type):
    """One mean trace per session (across channels-in-array, all epochs)."""
    vals = []
    for sd in session_epochs:
        ch_valid = [c for c in ch_names if c in sd['channels']]
        if not ch_valid:
            continue
        eps = []
        for c in ch_valid:
            c_i = sd['channels'].index(c)
            arr = sd['epochs'].get((c_i, entry_type))
            if arr is None or arr.size == 0:
                continue
            eps.append(arr)
        if not eps:
            continue
        all_eps = np.concatenate(eps, axis=0)
        vals.append(np.nanmean(all_eps, axis=0))
    if not vals:
        return None
    return np.stack(vals, axis=0)


# -----------------------------
# Permutation tests across entry-type pairs
# -----------------------------
print('\n=== Permutation tests across entry-type pairs ===')
entry_labels = list(ENTRY_TYPES.values())
pairs = list(itertools.combinations(entry_labels, 2))
print(f'  Pairs: {len(pairs)}')

x_axis = epoch_t_ref

for (l1, l2) in pairs:
    print(f'  {l1} vs {l2}')

    for i_arr, ch_names in enumerate(Sig_CH):
        # ---- per channel ----
        fig, axes = plt.subplots(6, 6, figsize=(15, 12))
        axes = axes.flatten()
        for ichan, ch_name in enumerate(ch_names):
            ax = axes[ichan]
            d1 = collect_channel_epochs(ch_name, l1)
            d2 = collect_channel_epochs(ch_name, l2)
            if d1 is None or d2 is None or d1.shape[0] < 3 or d2.shape[0] < 3:
                ax.set_title(f'{ch_name}\n(n/a)', fontsize=6)
                continue
            diff, sig, thr = permutation_test(d1, d2, n_perms=n_perms, alpha=alpha, rng=rng)

            npz_path = os.path.join(
                results_data_dir,
                f'permdata_pair_{l1}_VS_{l2}_array{i_arr+1}_{ch_name}.npz')
            if not os.path.exists(npz_path):
                np.savez_compressed(
                    npz_path, diff=diff, sig=sig, thr=thr,
                    mean1=np.nanmean(d1, axis=0), mean2=np.nanmean(d2, axis=0),
                    x_axis=x_axis, group1=l1, group2=l2,
                    n1=d1.shape[0], n2=d2.shape[0],
                    ch_name=ch_name, array_index=i_arr+1)

            ax.plot(x_axis, diff, color='k', lw=0.8)
            ax.fill_between(x_axis, diff, where=sig, color='red', alpha=0.4)
            ax.axhline(0, color='gray', lw=0.5)
            ax.axvline(0, color='gray', lw=0.5, linestyle=':')
            ax.set_title(f'{ch_name} (n={d1.shape[0]}/{d2.shape[0]})', fontsize=6)
        for j in range(len(ch_names), 36):
            axes[j].set_visible(False)
        fig.suptitle(f'entry-locked  {l1}  vs  {l2}  - Array {i_arr+1}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(
            output_dir, f'perm_entry_pair_{l1}_VS_{l2}_array{i_arr+1}.pdf'))
        plt.close(fig)

        # ---- single-array level ----
        d1a = collect_array_session_means(ch_names, l1)
        d2a = collect_array_session_means(ch_names, l2)
        if d1a is None or d2a is None or d1a.shape[0] < 2 or d2a.shape[0] < 2:
            continue
        diff_a, sig_a, thr_a = permutation_test(d1a, d2a, n_perms=n_perms, alpha=alpha, rng=rng)
        fig_a, ax_a = plt.subplots(figsize=(6, 4))
        ax_a.plot(x_axis, diff_a, color='k', lw=1.5)
        ax_a.fill_between(x_axis, diff_a, where=sig_a, color='red', alpha=0.4)
        ax_a.axhline(0, color='gray', lw=0.8)
        ax_a.axvline(0, color='gray', lw=0.8, linestyle=':')
        ax_a.set_title(f'Array {i_arr+1}  {l1} vs {l2}')
        ax_a.set_xlabel('Time from RF entry (s)')
        ax_a.set_ylabel('ΔAmplitude')
        plt.tight_layout()
        fig_a.savefig(os.path.join(
            output_dir, f'perm_entry_pair_{l1}_VS_{l2}_ARRAYCOMBINED_array{i_arr+1}.pdf'))
        plt.close(fig_a)

        npz_path = os.path.join(
            results_data_dir,
            f'permdata_entry_pair_{l1}_VS_{l2}_ARRAY_array{i_arr+1}.npz')
        if not os.path.exists(npz_path):
            np.savez_compressed(
                npz_path, diff=diff_a, sig=sig_a, thr=thr_a,
                mean1=np.nanmean(d1a, axis=0), mean2=np.nanmean(d2a, axis=0),
                x_axis=x_axis, group1=l1, group2=l2,
                n_sessions1=d1a.shape[0], n_sessions2=d2a.shape[0],
                array_index=i_arr+1)


# -----------------------------
# Combined-array level (Arrays 1-3 merged)
# -----------------------------
print('\n=== Combined arrays 1-3 ===')
for (l1, l2) in pairs:
    for i_arr, ch_names in enumerate(Sig_CH):
        if i_arr < 3:
            if i_arr == 0:
                combined_ch_names = list(np.concatenate(Sig_CH[:3]))
            else:
                continue
        else:
            combined_ch_names = list(ch_names)

        d1a = collect_array_session_means(combined_ch_names, l1)
        d2a = collect_array_session_means(combined_ch_names, l2)
        if d1a is None or d2a is None or d1a.shape[0] < 2 or d2a.shape[0] < 2:
            continue
        diff_a, sig_a, thr_a = permutation_test(d1a, d2a, n_perms=n_perms, alpha=alpha, rng=rng)

        array_label = f'Array {i_arr+1}' if i_arr >= 3 else 'Array 13'
        fig_a, ax_a = plt.subplots(figsize=(6, 4))
        ax_a.plot(x_axis, diff_a, color='k', lw=1.5)
        ax_a.fill_between(x_axis, diff_a, where=sig_a, color='red', alpha=0.4)
        ax_a.axhline(0, color='gray', lw=0.8)
        ax_a.axvline(0, color='gray', lw=0.8, linestyle=':')
        ax_a.set_title(f'{array_label}  {l1} vs {l2}')
        ax_a.set_xlabel('Time from RF entry (s)')
        ax_a.set_ylabel('ΔAmplitude')
        plt.tight_layout()
        fig_a.savefig(os.path.join(
            output_dir,
            f'cb_perm_entry_pair_{l1}_VS_{l2}_{array_label.replace(" ", "")}.pdf'))
        plt.close(fig_a)

        npz_path = os.path.join(
            results_data_dir,
            f'cb_permdata_entry_pair_{l1}_VS_{l2}_{array_label.replace(" ", "")}.npz')
        if not os.path.exists(npz_path):
            np.savez_compressed(
                npz_path, diff=diff_a, sig=sig_a, thr=thr_a,
                mean1=np.nanmean(d1a, axis=0), mean2=np.nanmean(d2a, axis=0),
                x_axis=x_axis, group1=l1, group2=l2,
                n_sessions1=d1a.shape[0], n_sessions2=d2a.shape[0],
                array_index=i_arr+1)

print(f'\nDone. Plots → {output_dir}\n        Stats → {results_data_dir}')
