"""
Counting + histograms for saccade x RF-content TRANSITIONS (framing B).

For every (saccade, channel) we read the channel's retinotopic RF state at the
saccade-onset sample (PRE-saccade) and at the saccade-landing/offset sample
(POST-saccade), using the SAME RF-state derivation as
RF_In_Out/erp_spectra_stats_rf_inout.py and saccade_erp_rf_split.py
(RF_stim_collapse.h5 + log; stimulus already converted to eye coordinates, so
"inside RF" is gaze-dependent and a saccade can move a stimulus in/out).

We DO NOT epoch LFP here. We only tabulate:
  1. transition category counts (pre_state -> post_state), pooled + per session
  2. the distribution of how long the POST-landing state persists (dwell), so you
     can pick a dwell threshold / post-window rule before committing to the ERP.

States: 0 = background, 1 = target_in, 2 = distractor_in, -1 = no RF data.

Transition categories (pre -> post):
  target_enters (0->1), distractor_enters (0->2),
  target_exits (1->0), distractor_exits (2->0),
  target_to_distractor (1->2), distractor_to_target (2->1),
  stays_target (1->1), stays_distractor (2->2), stays_background (0->0)

Run in the warping env (parse_logfile, time_conversion, syncopy, h5py).
"""

# -----------------------------
# Imports
# -----------------------------
import os
import sys
import glob
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
# User Config  (mirrors saccade_erp_rf_split.py)
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

# microsaccade exclusion (same as ERP script)
MIN_SACC_DUR_MS = 6.0
MICRO_AMP_DEG   = 1.0

# small settle after landing before reading post_state (samples added at offset)
POST_SETTLE_MS  = 0.0

output_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/saccade_lfp/saccade_rf_transition_counts'
os.makedirs(output_dir, exist_ok=True)

STATE_NAME = {-1: 'no_data', 0: 'background', 1: 'target', 2: 'distractor'}

# (pre, post) -> category label
TRANSITIONS = {
    (0, 1): 'target_enters',
    (0, 2): 'distractor_enters',
    (1, 0): 'target_exits',
    (2, 0): 'distractor_exits',
    (1, 2): 'target_to_distractor',
    (2, 1): 'distractor_to_target',
    (1, 1): 'stays_target',
    (2, 2): 'stays_distractor',
    (0, 0): 'stays_background',
}
CATEGORY_ORDER = [
    'target_enters', 'distractor_enters',
    'target_exits', 'distractor_exits',
    'target_to_distractor', 'distractor_to_target',
    'stays_target', 'stays_distractor', 'stays_background',
]


# -----------------------------
# Helpers (shared logic with saccade_erp_rf_split.py)
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
            (datalfp.trialdefinition, np.arange(nTrials).reshape(-1, 1))
        )


def parse_session_log(session):
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
        'n_log_trials':       len(target_onset),
    }


def trial_channel_rf_states(hf, trial_name, ch_indices, target_stim,
                            target_onset_t, reach_ts_all, reach_evt_all,
                            aligned_stim_t, lfp_time):
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


def detect_saccades(sacc_npz, session, fs_eye):
    """Return (onset, offset) sample pairs of non-micro saccades (500 Hz space)."""
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
        if amp < MICRO_AMP_DEG:
            continue
        keep_on.append(on)
        keep_off.append(off)
    return np.asarray(keep_on, dtype=int), np.asarray(keep_off, dtype=int)


def samples_to_log_time(session, samples):
    """Convert 500 Hz / iRec-row sample indices -> log/Unreal time (s)."""
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


# -----------------------------
# Main: per-session transition tabulation
# -----------------------------
print('Loading saccade dataset …')
sacc_npz = np.load(sacc_npz_path, allow_pickle=True)
fs_eye = int(sacc_npz['fs'])

# pooled accumulators
cat_counts_total = {c: 0 for c in CATEGORY_ORDER}
cat_counts_sess  = {s: {c: 0 for c in CATEGORY_ORDER} for s in sessions}
dwell_ms_target     = []   # post_state == target
dwell_ms_distractor = []   # post_state == distractor
n_no_post_data = 0

for session_name in sessions:
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
    settle = int(round(POST_SETTLE_MS / 1000 * fs))

    data_clean, valid_channels, valid_ch_idx = remove_nan_trials_channels(datalfp)
    if data_clean is None:
        print('  all-NaN data, skipping')
        continue
    sel_trials = data_clean.trialdefinition[:, 3].astype(int)
    n_trials = len(data_clean.trials)

    log_info = parse_session_log(session_name)
    n_max_trial = min(log_info['n_log_trials'], len(log_info['aligned_stim_times']))

    onsets, offsets = detect_saccades(sacc_npz, session_name, fs_eye)
    on_log  = samples_to_log_time(session_name, onsets)
    off_log = samples_to_log_time(session_name, offsets)
    good = ~(np.isnan(on_log) | np.isnan(off_log))
    on_log, off_log = on_log[good], off_log[good]
    print(f'  {len(on_log)} non-micro saccades (>= {MIN_SACC_DUR_MS:.0f} ms, '
          f'amp >= {MICRO_AMP_DEG:.0f} deg)')

    try:
        hf = h5py.File(rf_path, 'r')
        trial_names = sorted(hf.keys(), key=lambda n: int(n.split('_')[-1]))
    except (OSError, RuntimeError) as e:
        print(f'  !! cannot read RF HDF5 ({e.__class__.__name__}: {e}); skipping')
        continue

    with hf:
        for tr_i in range(n_trials):
            trl = sel_trials[tr_i]
            if trl >= n_max_trial or trl >= len(trial_names):
                continue

            trial_time = data_clean.time[tr_i]
            n_t = len(trial_time)

            on_rel  = on_log  - log_info['target_onset'][trl]
            off_rel = off_log - log_info['target_onset'][trl]
            # saccades whose onset falls inside this trial's LFP window
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
                for c_i in range(states_t.shape[1]):
                    pre  = int(states_t[s_on,  c_i])
                    post = int(states_t[s_off, c_i])
                    if pre < 0 or post < 0:
                        continue
                    cat = TRANSITIONS.get((pre, post))
                    if cat is None:
                        continue
                    cat_counts_total[cat] += 1
                    cat_counts_sess[session_name][cat] += 1

                    # post-landing dwell: contiguous samples of `post` from s_off
                    if post in (1, 2):
                        col = states_t[s_off:, c_i]
                        same = np.where(col != post)[0]
                        dwell_samp = same[0] if same.size else len(col)
                        dwell_ms = dwell_samp * 1000.0 / fs
                        if post == 1:
                            dwell_ms_target.append(dwell_ms)
                        else:
                            dwell_ms_distractor.append(dwell_ms)

    print('  category counts this session:',
          {c: cat_counts_sess[session_name][c] for c in CATEGORY_ORDER
           if cat_counts_sess[session_name][c]})


# -----------------------------
# Print summary table
# -----------------------------
print('\n=== Transition category counts (pooled across sessions) ===')
total = sum(cat_counts_total.values())
for c in CATEGORY_ORDER:
    n = cat_counts_total[c]
    pct = 100 * n / total if total else 0
    print(f'  {c:22s}  {n:>9d}  ({pct:5.1f}%)')
print(f'  {"TOTAL":22s}  {total:>9d}')

dwell_ms_target     = np.asarray(dwell_ms_target)
dwell_ms_distractor = np.asarray(dwell_ms_distractor)
print('\n=== Post-landing dwell (ms) for events ending IN the RF ===')
for name, arr in [('target', dwell_ms_target), ('distractor', dwell_ms_distractor)]:
    if arr.size:
        print(f'  {name:11s} n={arr.size:>8d}  median={np.median(arr):6.0f}  '
              f'frac>=50ms={np.mean(arr >= 50):.2f}  >=100ms={np.mean(arr >= 100):.2f}  '
              f'>=200ms={np.mean(arr >= 200):.2f}  >=300ms={np.mean(arr >= 300):.2f}')
    else:
        print(f'  {name:11s} n=0')


# -----------------------------
# Plots
# -----------------------------
# 1) transition category bar chart (pooled)
fig, ax = plt.subplots(figsize=(10, 5))
vals = [cat_counts_total[c] for c in CATEGORY_ORDER]
colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a',
          '#66a61e', '#e6ab02', '#a6cee3', '#fb9a99', '#cccccc']
ax.bar(range(len(CATEGORY_ORDER)), vals, color=colors)
ax.set_xticks(range(len(CATEGORY_ORDER)))
ax.set_xticklabels(CATEGORY_ORDER, rotation=40, ha='right', fontsize=8)
ax.set_ylabel('(saccade, channel) count')
ax.set_title('Saccade x RF transitions (pre-onset -> post-landing state), pooled')
for i, v in enumerate(vals):
    ax.text(i, v, str(v), ha='center', va='bottom', fontsize=7)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'transition_category_counts.pdf'))
plt.close(fig)

# 2) post-landing dwell histograms (target vs distractor entries+stays)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
bins = np.arange(0, 605, 20)
for ax, name, arr, col in [
        (axes[0], 'target',     dwell_ms_target,     '#1b9e77'),
        (axes[1], 'distractor', dwell_ms_distractor, '#d95f02')]:
    if arr.size:
        clipped = np.clip(arr, 0, 600)
        ax.hist(clipped, bins=bins, color=col, alpha=0.85)
        ax.axvline(np.median(arr), color='k', ls='--', lw=1,
                   label=f'median={np.median(arr):.0f} ms')
        for thr in (50, 100, 200, 300):
            ax.axvline(thr, color='gray', ls=':', lw=0.6)
        ax.legend(fontsize=8)
    ax.set_title(f'{name} in RF post-landing (n={arr.size})')
    ax.set_xlabel('post-landing dwell (ms, clipped at 600)')
axes[0].set_ylabel('count')
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'post_landing_dwell_hist.pdf'))
plt.close(fig)

# 3) per-session stacked category counts
fig, ax = plt.subplots(figsize=(11, 5))
bottom = np.zeros(len(sessions))
x = np.arange(len(sessions))
for ci, c in enumerate(CATEGORY_ORDER):
    vals = np.array([cat_counts_sess[s][c] for s in sessions])
    ax.bar(x, vals, bottom=bottom, label=c, color=colors[ci])
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels(sessions, rotation=30, ha='right')
ax.set_ylabel('(saccade, channel) count')
ax.set_title('Transition categories per session')
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'transition_category_counts_per_session.pdf'))
plt.close(fig)

print(f'\nDone. Plots -> {output_dir}')
