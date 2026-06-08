"""
Saccade-triggered LFP ERP, split by RF *transition* (framing B), with a dwell
filter so the post-saccade state actually covers the analysis window.

For every (saccade, channel) we read that channel's RF state at TWO samples,
using the SAME RF-state derivation as RF_In_Out/erp_spectra_stats_rf_inout.py
and saccade_rf_transition_counts.py (RF_stim_collapse.h5 + log):
    pre  = state at saccade ONSET   (0=background, 1=target_in, 2=distractor_in)
    post = state at saccade LANDING (offset + POST_SETTLE_MS)
The (pre, post) pair defines the group, e.g. target_enter = (0->1),
target_stay = (1->1).  This lets us separate saccades that BROUGHT a stimulus
into the RF from saccades where it was already there.

A dwell filter requires the post-landing state to persist >= DWELL_MIN_MS, so an
"enter" only counts if the stimulus stays in the RF long enough to cover the
chosen POST window.  Optionally groups are balanced (subsampled to equal n) per
tested pair so a difference can't be driven by count or by dwell-distribution
mismatch.

Alignment (all in one consistent timebase):
    saccade onset/offset sample (500 Hz, == iRec position-file row)
      -> iRec time  pos_t[sample]
      -> log/Unreal time  + log_irec_offset  (tc.align_irec)
      -> per trial:  rel_t = log_time - target_onset[trial]   (== LFP trial time,
         which is 0 at stim onset)  -> nearest LFP sample -> slice epoch.
Epochs are aligned to ALIGN_EVENT ('onset' or 'landing') and pooled across
(saccade, channel, session) by group, then compared with max/min permutation
thresholding.

Microsaccades (< MIN_SACC_DUR_MS or amplitude < MICRO_AMP_DEG) are excluded.

Run in the warping env (needs parse_logfile, time_conversion, syncopy, h5py).
"""

# -----------------------------
# Imports
# -----------------------------
import os
import sys
import glob
import itertools
import numpy as np
import pandas as pd
import h5py
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
rf_stim_dir  = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping_no_reset/RFarea_stim'
sacc_npz_path = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/saccade_detection/stitched_sessions.npz'

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

# ---- epoch window (s) around the alignment event ----
PRE          = 0.2     # s, before alignment event
POST         = 0.15    # s, after alignment event

# ---- alignment event for the epoch ----
#   'onset'   : align to saccade onset  (activity locked to the eye movement itself, including the movement transient)
#   'landing' : align to saccade offset (evoked response to a stimulus entering the RF)
ALIGN_EVENT  = 'landing'

# ---- dwell filter ----
# require the post-landing RF state to persist at least this long, in ms.
# Set to 0.0 to disable.  A natural choice is >= POST*1000 so the post state
# covers the whole post-alignment window.
DWELL_MIN_MS    = 150.0
# small settle added after the saccade offset before reading the post state (ms)
POST_SETTLE_MS  = 0.0

# ---- group balancing ----
# subsample the larger group to the smaller group's n, per tested pair, so a
# difference can't be driven by unequal counts / dwell distributions.
BALANCE_GROUPS  = True

# microsaccade exclusion
MIN_SACC_DUR_MS = 6.0
MICRO_AMP_DEG   = 1.0

# ---- groups: name -> (pre_state, post_state) ----
# states: 0 = background, 1 = target_in, 2 = distractor_in
GROUP_TRANSITIONS = {
    'target_enter':     (0, 1),
    'distractor_enter': (0, 2),
    'target_stay':      (1, 1),
    'distractor_stay':  (2, 2),
    'background':       (0, 0),
}
GROUP_LABELS = list(GROUP_TRANSITIONS.keys())
# (pre, post) -> group name, for fast lookup during extraction
TRANSITION_TO_GROUP = {v: k for k, v in GROUP_TRANSITIONS.items()}

# ---- pairs to test (must be names from GROUP_TRANSITIONS) ----
# leave empty [] to test every combination of the groups above.
PAIRS = [
    ('target_enter',     'distractor_enter'),
    ('target_enter',     'background'),
    ('distractor_enter', 'background'),
    ('target_enter',     'target_stay'),
    ('distractor_enter', 'distractor_stay'),
]

output_dir       = (f'/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/saccade_lfp/'
                    f'saccade_locked_rf_transition_{ALIGN_EVENT}_'
                    f'{int(PRE*1000)}_{int(POST*1000)}_dwell{int(DWELL_MIN_MS)}')
results_data_dir = os.path.join(output_dir, 'data')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_data_dir, exist_ok=True)

n_perms = 1000
alpha   = 0.05
rng     = np.random.default_rng(42)

# -----------------------------
# Helpers (LFP cleaning / stats) -- shared with erp_spectra_stats_rf_inout.py
# -----------------------------
def remove_nan_trials_channels(datas):
    """Drop all-NaN trials, then drop channels that are all-NaN across every
    remaining trial. Handles variable-length trials (no np.stack)."""
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


def balance_pair(d1, d2, rng):
    """Subsample the larger array to the smaller's n along axis 0."""
    n = min(d1.shape[0], d2.shape[0])
    if d1.shape[0] > n:
        d1 = d1[rng.choice(d1.shape[0], n, replace=False)]
    if d2.shape[0] > n:
        d2 = d2[rng.choice(d2.shape[0], n, replace=False)]
    return d1, d2


# -----------------------------
# Log + RF helpers -- shared with erp_spectra_stats_rf_inout.py
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


# -----------------------------
# Saccade helpers
# -----------------------------
def detect_saccades(sacc_npz, session, fs_eye):
    """(onset, offset) sample pairs of non-micro saccades (500 Hz / iRec-row space)."""
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
        amp = np.hypot(x[off] - x[on], y[off] - y[on])
        if amp < MICRO_AMP_DEG:                 # microsaccade
            continue
        keep_on.append(on)
        keep_off.append(off)
    return np.asarray(keep_on, dtype=int), np.asarray(keep_off, dtype=int)


def samples_to_log_time(session, samples):
    """Convert 500 Hz / iRec-row sample indices -> log/Unreal time (s).
    Samples beyond the position file get NaN."""
    folder = os.path.join(eye_data_dir, session)
    log_path = os.path.join(folder, session_logfiles[session])
    eye_file = next(os.path.basename(f).replace('.csv', '')
                    for f in glob.glob(os.path.join(folder, '*.csv'))
                    if 'net.csv' not in os.path.basename(f))
    net_csv = os.path.join(folder, eye_file + 'net.csv')
    pos_csv = os.path.join(folder, eye_file + '.csv')

    log_irec_offset = tc.align_irec(log_path, net_csv)
    pos_t = pd.read_csv(pos_csv, usecols=[0]).to_numpy().ravel()  # iRec time per sample
    valid = samples < len(pos_t)
    out = np.full(len(samples), np.nan)
    out[valid] = pos_t[samples[valid]] + log_irec_offset
    return out


# -----------------------------
# Main: per-session epoch extraction
# -----------------------------
session_epochs = []
n_events_used  = {g: 0 for g in GROUP_LABELS}
fs_ref = None
epoch_t_ref = None

print('Loading saccade dataset …')
sacc_npz = np.load(sacc_npz_path, allow_pickle=True)
fs_eye = int(sacc_npz['fs'])

for sess_idx, session_name in enumerate(sessions):
    print(f'\n=== Processing session {session_name} ===')
    lfp_path = os.path.join(lfp_data_dir, session_name, 'Cleaned_lfp_FT.spy')
    rf_path  = os.path.join(rf_stim_dir, session_name, 'RF_stim_collapse.h5')
    if not (os.path.exists(lfp_path) and os.path.exists(rf_path)):
        print('  missing LFP or RF file, skipping')
        continue
    if f'{session_name}__pred_orig' not in sacc_npz:
        print('  no saccade data for this session, skipping')
        continue

    datalfp = spy.load(lfp_path)
    ensure_trialindex_in_trialdefinition(datalfp)
    fs = float(datalfp.samplerate)
    pre_samples  = int(round(PRE  * fs))
    post_samples = int(round(POST * fs))
    epoch_len    = pre_samples + post_samples
    epoch_t      = (np.arange(epoch_len) - pre_samples) / fs
    settle       = int(round(POST_SETTLE_MS / 1000 * fs))
    dwell_min_samp = int(round(DWELL_MIN_MS / 1000 * fs))

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

    # saccade onsets + offsets -> log time (once per session)
    onsets, offsets = detect_saccades(sacc_npz, session_name, fs_eye)
    on_log  = samples_to_log_time(session_name, onsets)
    off_log = samples_to_log_time(session_name, offsets)
    good = ~(np.isnan(on_log) | np.isnan(off_log))
    on_log, off_log = on_log[good], off_log[good]
    print(f'  {len(on_log)} non-micro saccades (>= {MIN_SACC_DUR_MS:.0f} ms, '
          f'amp >= {MICRO_AMP_DEG:.0f} deg)')

    epochs_this  = {}                          # {(c_i, group): list[np.array]}
    n_used_sess  = {g: 0 for g in GROUP_LABELS}

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

            # saccades whose onset falls inside this trial's LFP window
            on_rel  = on_log  - log_info['target_onset'][trl]
            off_rel = off_log - log_info['target_onset'][trl]
            in_trl = np.where((on_rel >= trial_time[0]) & (on_rel <= trial_time[-1]))[0]
            if in_trl.size == 0:
                continue

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

            on_smp  = np.clip(np.searchsorted(trial_time, on_rel[in_trl]),  0, n_t - 1)
            off_smp = np.clip(np.searchsorted(trial_time, off_rel[in_trl]) + settle,
                              0, n_t - 1)

            for s_on, s_off in zip(on_smp, off_smp):
                align_smp = s_on if ALIGN_EVENT == 'onset' else s_off
                win_start = align_smp - pre_samples
                win_end   = align_smp + post_samples
                if win_start < 0 or win_end > n_t:
                    continue
                for c_i in range(states_t.shape[1]):
                    pre  = int(states_t[s_on,  c_i])
                    post = int(states_t[s_off, c_i])
                    if pre < 0 or post < 0:
                        continue                # no RF data at one of the samples
                    group = TRANSITION_TO_GROUP.get((pre, post))
                    if group is None:
                        continue                # transition not requested

                    # post-landing dwell: contiguous samples of `post` from s_off
                    if dwell_min_samp > 0:
                        col = states_t[s_off:, c_i]
                        chg = np.where(col != post)[0]
                        dwell_samp = chg[0] if chg.size else len(col)
                        if dwell_samp < dwell_min_samp:
                            continue

                    epoch = np.asarray(trial_lfp[win_start:win_end, c_i],
                                       dtype=np.float32)
                    if np.any(np.isnan(epoch)):
                        continue
                    epochs_this.setdefault((c_i, group), []).append(epoch)
                    n_events_used[group] += 1
                    n_used_sess[group]   += 1

    epochs_arr = {key: np.stack(v, axis=0) for key, v in epochs_this.items() if v}
    session_epochs.append({
        'session':  session_name,
        'channels': valid_channels,
        'epoch_t':  epoch_t,
        'fs':       fs,
        'epochs':   epochs_arr,
    })
    print('  Saccade-channel epochs this session — used:', n_used_sess)


# -----------------------------
# Counts summary
# -----------------------------
print('\n=== Saccade-channel epoch counts (pooled across sessions) ===')
print(f'  align={ALIGN_EVENT}  PRE={PRE}s  POST={POST}s  '
      f'dwell>={DWELL_MIN_MS:.0f}ms  balance={BALANCE_GROUPS}')
for g in GROUP_LABELS:
    print(f'  {g:18s}  used={n_events_used[g]:>8d}')

if not session_epochs:
    raise RuntimeError('No sessions processed for epoch extraction.')


# -----------------------------
# Pooling helpers -- shared with erp_spectra_stats_rf_inout.py
# -----------------------------
ref_channels = session_epochs[0]['channels']
Sig_CH = np.array_split(ref_channels, 6)


def collect_channel_epochs(ch_name, group):
    """Pool epochs across sessions for one channel and group."""
    vals = []
    for sd in session_epochs:
        if ch_name not in sd['channels']:
            continue
        c_i = sd['channels'].index(ch_name)
        arr = sd['epochs'].get((c_i, group))
        if arr is None or arr.size == 0:
            continue
        vals.append(arr)
    if not vals:
        return None
    return np.concatenate(vals, axis=0)


def collect_array_session_means(ch_names, group):
    """One mean trace per session (across channels-in-array, all epochs)."""
    vals = []
    for sd in session_epochs:
        ch_valid = [c for c in ch_names if c in sd['channels']]
        if not ch_valid:
            continue
        eps = []
        for c in ch_valid:
            c_i = sd['channels'].index(c)
            arr = sd['epochs'].get((c_i, group))
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
# Permutation tests across group pairs
# -----------------------------
print('\n=== Permutation tests across group pairs ===')
pairs = PAIRS if PAIRS else list(itertools.combinations(GROUP_LABELS, 2))
print(f'  Pairs: {pairs}')

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
            if BALANCE_GROUPS:
                d1, d2 = balance_pair(d1, d2, rng)
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

            m1 = np.nanmean(d1, axis=0)
            m2 = np.nanmean(d2, axis=0)
            ax.plot(x_axis, m1, color=(0.55, 0.0, 0.55), lw=0.8, label=l1)
            ax.plot(x_axis, m2, color=(0.0, 0.39, 0.39), lw=0.8, label=l2)
            ylo, yhi = ax.get_ylim()
            if sig.any():
                ax.fill_between(x_axis, ylo, yhi, where=sig,
                                color='#8dd3c7', alpha=0.4, zorder=0)
                ax.set_ylim(ylo, yhi)
            ax.axhline(0, color='gray', lw=0.5)
            ax.axvline(0, color='gray', lw=0.5, linestyle=':')
            ax.set_title(f'{ch_name} (n={d1.shape[0]}/{d2.shape[0]})', fontsize=6)
        for j in range(len(ch_names), 36):
            axes[j].set_visible(False)
        fig.suptitle(f'saccade-locked ({ALIGN_EVENT})  {l1}  vs  {l2}  - Array {i_arr+1}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(
            output_dir, f'perm_sacc_pair_{l1}_VS_{l2}_array{i_arr+1}.pdf'))
        plt.close(fig)

        # ---- single-array level ----
        d1a = collect_array_session_means(ch_names, l1)
        d2a = collect_array_session_means(ch_names, l2)
        if d1a is None or d2a is None or d1a.shape[0] < 2 or d2a.shape[0] < 2:
            continue
        diff_a, sig_a, thr_a = permutation_test(d1a, d2a, n_perms=n_perms, alpha=alpha, rng=rng)
        m1a = np.nanmean(d1a, axis=0)
        m2a = np.nanmean(d2a, axis=0)
        fig_a, ax_a = plt.subplots(figsize=(6, 4))
        ax_a.plot(x_axis, m1a, color=(0.55, 0.0, 0.55), lw=1.5, label=l1)
        ax_a.plot(x_axis, m2a, color=(0.0, 0.39, 0.39), lw=1.5, label=l2)
        ylo, yhi = ax_a.get_ylim()
        if sig_a.any():
            ax_a.fill_between(x_axis, ylo, yhi, where=sig_a,
                              color='#8dd3c7', alpha=0.4, zorder=0)
            ax_a.set_ylim(ylo, yhi)
        ax_a.axhline(0, color='gray', lw=0.8)
        ax_a.axvline(0, color='gray', lw=0.8, linestyle=':')
        ax_a.set_title(f'Array {i_arr+1}  {l1} vs {l2}  ({ALIGN_EVENT})')
        ax_a.set_xlabel(f'Time from saccade {ALIGN_EVENT} (s)')
        ax_a.set_ylabel('Amplitude')
        ax_a.legend(fontsize=8)
        plt.tight_layout()
        fig_a.savefig(os.path.join(
            output_dir, f'perm_sacc_pair_{l1}_VS_{l2}_ARRAYCOMBINED_array{i_arr+1}.pdf'))
        plt.close(fig_a)

        npz_path = os.path.join(
            results_data_dir,
            f'permdata_sacc_pair_{l1}_VS_{l2}_ARRAY_array{i_arr+1}.npz')
        if not os.path.exists(npz_path):
            np.savez_compressed(
                npz_path, diff=diff_a, sig=sig_a, thr=thr_a,
                mean1=np.nanmean(d1a, axis=0), mean2=np.nanmean(d2a, axis=0),
                x_axis=x_axis, group1=l1, group2=l2,
                n_sessions1=d1a.shape[0], n_sessions2=d2a.shape[0],
                array_index=i_arr+1)

print(f'\nDone. Plots → {output_dir}\n        Stats → {results_data_dir}')
