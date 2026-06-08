"""
Saccade raster across trials.

One row per trial (oldest -> newest on the y axis), the whole trial on the x
axis (time from TRIAL onset, event 3000), and a tick for every (non-micro)
saccade onset.  Stimulus onset (event 3011) is marked per trial, and reward
delivery and the trial outcome (correct / wrong) are marked at the right; block
boundaries are drawn as white gaps -- same trial ordering and filtering as
RF_In_Out/RF_inout_channel_raster_leafid.py.

Saccade onsets come from the U'n'Eye detection (500 Hz) and are converted to
log/Unreal time with the same alignment as saccade_lfp/saccade_erp_rf_split.py,
so they sit in the same clock as the stimulus-onset event markers.

Excluded trials (matching the leaf-id raster):
  * Trials containing event 3091 (monkey exits block to collect apple)
  * Trials with no correct/wrong response

Run in the warping env (needs parse_logfile + time_conversion).
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions')
sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
import time_conversion as tc                 # noqa: E402
from parse_logfile import TextLog            # noqa: E402

# ============================================================
# CONFIG
# ============================================================

session = '20230214'

session_logfiles = {
    '20230203': '2023_02_03-11_35_57_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230208': '2023_02_08-10_58_17_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230209': '2023_02_09-11_19_51_Cosmos_LeafForaging_001_KAS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230213': '2023_02_13-11_13_43_Cosmos_LeafForaging_002_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230214': '2023_02_14-11_42_27_Cosmos_LeafForaging_001_PAF_GrassyLandscapeWithBackgroundDark_Cont.log',
}

eye_data_dir  = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data'
sacc_npz_path = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/saccade_detection/stitched_sessions.npz'
save_dir      = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/saccades'
os.makedirs(save_dir, exist_ok=True)

n_stimuli = 5
stim_name = 'ImageStimulus'
MAX_T     = 8.0      # s -- whole-trial x-axis cap (from trial onset); longer trials clipped
GAP_ROWS  = 2        # blank rows at block boundaries

# microsaccade exclusion (same thresholds as saccade_erp_rf_split.py)
MIN_SACC_DUR_MS = 6.0
MICRO_AMP_DEG   = 1.0

# ============================================================
# 1. PARSE LOG FILE  -> event timestamps
# ============================================================

log_path = os.path.join(eye_data_dir, session, session_logfiles[session])
print(f'Parsing log file: {log_path}')

with TextLog(log_path) as log:
    log.make_id_struct()
    evt, ts, evt_desc, true_ts = log.parse_eventmarkers()

with TextLog(log_path) as log:
    trial_data = log.get_info_per_trial(return_eventmarkers=True, return_loc=False)

trial_df = pd.DataFrame(trial_data).sort_values('TrialIndex').reset_index(drop=True)

# Event timestamps
trial_onset_ts      = ts[np.where(evt == 3000)[0]]   # trial start
target_onset        = ts[np.where(evt == 3011)[0]]   # stimulus onset
trial_end_ts        = ts[np.where(evt == 3090)[0]]
block_exit_ts       = ts[np.where(evt == 3091)[0]]
response_correct_ts = ts[np.where(evt == 1)[0]]
response_wrong_ts   = ts[np.where(evt == 2)[0]]

reward_mask   = (evt >= 5000) & (evt <= 5999)
reward_ts_all = ts[reward_mask]

n_log_trials = len(target_onset)
print(f'  Log trials: {n_log_trials}  |  Block exits: {len(block_exit_ts)}')

# trial start for each trial = last 3000 at or before that trial's stim onset
t_start_per_trial = np.full(n_log_trials, np.nan)
stim_onset_rel    = np.full(n_log_trials, np.nan)   # 3011 time, relative to trial start
for trl in range(n_log_trials):
    idx = np.searchsorted(trial_onset_ts, target_onset[trl], side='right') - 1
    if idx >= 0:
        t_start_per_trial[trl] = trial_onset_ts[idx]
        stim_onset_rel[trl]    = target_onset[trl] - trial_onset_ts[idx]

# first-reward time per trial (relative to stim onset)
rew_t_per_trial = np.full(n_log_trials, np.nan)
for trl in range(n_log_trials):
    t0 = target_onset[trl]
    t1 = trial_end_ts[trl] if trl < len(trial_end_ts) else t0 + MAX_T
    in_window = (reward_ts_all >= t0) & (reward_ts_all <= t1)
    if in_window.any():
        rew_t_per_trial[trl] = reward_ts_all[np.where(in_window)[0][0]] - t0

# ============================================================
# 2. TRIAL FLAGS  (same logic as the leaf-id raster)
# ============================================================

n_trials = n_log_trials

has_block_exit = np.zeros(n_trials, dtype=bool)
for t_exit in block_exit_ts:
    idx = np.searchsorted(target_onset, t_exit, side='right') - 1
    if 0 <= idx < n_trials:
        has_block_exit[idx] = True

is_response_correct = np.zeros(n_trials, dtype=bool)
is_response_wrong   = np.zeros(n_trials, dtype=bool)
for trl in range(n_trials):
    t0 = target_onset[trl]
    t1 = trial_end_ts[trl] if trl < len(trial_end_ts) else t0 + MAX_T
    if np.any((response_correct_ts >= t0) & (response_correct_ts <= t1)):
        is_response_correct[trl] = True
    if np.any((response_wrong_ts >= t0) & (response_wrong_ts <= t1)):
        is_response_wrong[trl] = True

print(f'  Trials: {n_trials}  |  Block-exit: {has_block_exit.sum()}'
      f'  |  Correct: {is_response_correct.sum()}  |  Wrong: {is_response_wrong.sum()}')

# ============================================================
# 3. BLOCK STRUCTURE
# ============================================================

block_ids = np.zeros(n_trials, dtype=int)
current_block = 0
for trl in range(n_trials):
    block_ids[trl] = current_block
    if has_block_exit[trl]:
        current_block += 1
print(f'  Blocks found: {len(set(block_ids))}')

# ============================================================
# 4. FILTER
# ============================================================

correct_mask = is_response_correct & ~has_block_exit
wrong_mask   = is_response_wrong   & ~has_block_exit
plot_mask    = correct_mask | wrong_mask
print(f'  Correct (non-exit): {correct_mask.sum()}  |  Wrong (non-exit): {wrong_mask.sum()}')

# ============================================================
# 5. SACCADE ONSETS  -> log time
# ============================================================

def detect_saccade_onsets(sacc_npz, session, fs_eye):
    """Onset samples (500 Hz / iRec-row space) of non-micro saccades."""
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

    keep_on = []
    for on, off in zip(onsets, offsets):
        if off - on < min_dur_samp:
            continue
        if nan_mask[on:off].any():
            continue
        amp = np.hypot(x[off] - x[on], y[off] - y[on])
        if amp < MICRO_AMP_DEG:                 # microsaccade
            continue
        keep_on.append(on)
    return np.asarray(keep_on, dtype=int)


def samples_to_log_time(session, samples):
    """Convert 500 Hz / iRec-row sample indices -> log/Unreal time (s).
    Samples beyond the position file get NaN."""
    folder = os.path.join(eye_data_dir, session)
    log_p = os.path.join(folder, session_logfiles[session])
    eye_file = next(os.path.basename(f).replace('.csv', '')
                    for f in glob.glob(os.path.join(folder, '*.csv'))
                    if 'net.csv' not in os.path.basename(f))
    net_csv = os.path.join(folder, eye_file + 'net.csv')
    pos_csv = os.path.join(folder, eye_file + '.csv')

    log_irec_offset = tc.align_irec(log_p, net_csv)
    pos_t = pd.read_csv(pos_csv, usecols=[0]).to_numpy().ravel()  # iRec time per sample
    valid = samples < len(pos_t)
    out = np.full(len(samples), np.nan)
    out[valid] = pos_t[samples[valid]] + log_irec_offset
    return out


print('\nLoading saccade dataset and detecting onsets …')
sacc_npz = np.load(sacc_npz_path, allow_pickle=True)
fs_eye = int(sacc_npz['fs'])
if f'{session}__pred_orig' not in sacc_npz:
    raise RuntimeError(f'No saccade data for session {session} in {sacc_npz_path}')

onset_samples = detect_saccade_onsets(sacc_npz, session, fs_eye)
onset_log = samples_to_log_time(session, onset_samples)
onset_log = onset_log[~np.isnan(onset_log)]
print(f'  {len(onset_log)} non-micro saccades '
      f'(>= {MIN_SACC_DUR_MS:.0f} ms, amp >= {MICRO_AMP_DEG:.0f} deg)')

# ============================================================
# 6. BUILD RASTER  (saccade onset times relative to stim onset, per trial)
# ============================================================

row_saccades = []     # list of np.array of saccade times (s, from trial onset) per row
row_outcomes = []     # 'correct' | 'wrong' | 'gap'
row_reward   = []     # reward time (s, from trial onset) per row, NaN if none
row_stim     = []     # stimulus-onset time (s, from trial onset) per row, NaN if none
gap_positions = []
prev_block = -1

for trl in range(n_trials):
    if not plot_mask[trl]:
        continue

    blk = block_ids[trl]
    if blk != prev_block and prev_block != -1:
        for _ in range(GAP_ROWS):
            row_saccades.append(np.array([]))
            row_outcomes.append('gap')
            row_reward.append(np.nan)
            row_stim.append(np.nan)
            gap_positions.append(len(row_saccades) - 1)
    prev_block = blk

    t0 = t_start_per_trial[trl]            # trial onset
    if np.isnan(t0):
        continue
    t_end = trial_end_ts[trl] - t0 if trl < len(trial_end_ts) else MAX_T
    t_end = min(max(t_end, 0.0), MAX_T)

    rel = onset_log - t0
    rel = rel[(rel >= 0) & (rel <= t_end)]

    # reward time was relative to stim onset -> shift to trial onset
    rew_rel = rew_t_per_trial[trl] + stim_onset_rel[trl]

    row_saccades.append(rel)
    row_outcomes.append('correct' if correct_mask[trl] else 'wrong')
    row_reward.append(rew_rel)
    row_stim.append(stim_onset_rel[trl])

n_rows = len(row_saccades)
print(f'  Raster rows (incl. gaps): {n_rows}  '
      f'|  trials plotted: {sum(o != "gap" for o in row_outcomes)}')

# ============================================================
# 7. PLOT
# ============================================================

OUTCOME_COLOR = {'correct': '#2CA02C', 'wrong': '#D62728'}
MARKER_X_FRAC = 0.012

# x extent: longest plotted trial, capped at MAX_T
finite_ends = [s.max() if s.size else 0.0 for s in row_saccades]
stim_max = np.nanmax(row_stim) if np.any(~np.isnan(row_stim)) else 0.0
x_max = min(MAX_T, max(finite_ends + [stim_max]) + 0.2)
x_max = max(x_max, 1.0)

fig, ax = plt.subplots(figsize=(14, max(8, min(n_rows * 0.08 + 2, 40))))

# saccade ticks -- one short vertical line per saccade, per row
ax.eventplot(
    row_saccades,
    lineoffsets=np.arange(n_rows),
    linelengths=0.9,
    colors='black',
    linewidths=0.6,
)

ax.set_ylim(n_rows - 0.5, -0.5)   # row 0 (oldest) at top, like the leaf-id raster

# block boundary lines
for gp in gap_positions:
    ax.axhline(gp, color='0.6', linewidth=1.0)

if gap_positions:
    boundary_ys  = gap_positions[::GAP_ROWS]
    gap_rows_arr = np.array([-1] + list(boundary_ys) + [n_rows])
    for blk_i in range(len(gap_rows_arr) - 1):
        mid = (gap_rows_arr[blk_i] + gap_rows_arr[blk_i + 1]) / 2
        ax.text(x_max * 0.99, mid, f'Blk {blk_i + 1}',
                ha='right', va='center', fontsize=6, color='steelblue',
                fontweight='bold')

# trial onset
ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)

# stimulus-onset marker per trial (blue tick)
stim_xs = [s for s in row_stim if not np.isnan(s) and s <= MAX_T]
stim_ys = [i for i, s in enumerate(row_stim) if not np.isnan(s) and s <= MAX_T]
if stim_xs:
    ax.scatter(stim_xs, stim_ys, marker='|', color='#1F77B4', s=60,
               linewidths=1.0, zorder=4)

# reward markers (white dot, like RT marker in the leaf-id raster)
rew_xs = [r for r in row_reward if not np.isnan(r) and r <= MAX_T]
rew_ys = [i for i, r in enumerate(row_reward) if not np.isnan(r) and r <= MAX_T]
if rew_xs:
    ax.scatter(rew_xs, rew_ys, color='white', s=10, zorder=6,
               linewidths=0.5, edgecolors='black')

# outcome markers (right edge)
marker_x = x_max + x_max * MARKER_X_FRAC
for row_i, outcome in enumerate(row_outcomes):
    if outcome in OUTCOME_COLOR:
        ax.plot(marker_x, row_i, 's', color=OUTCOME_COLOR[outcome],
                markersize=4, clip_on=False, zorder=5)

ax.set_xlim(0, x_max)
ax.set_xlabel('Time from trial onset (s)', fontsize=10)
ax.set_ylabel('Trial (oldest → newest)', fontsize=10)
ax.set_title(f'{session} — saccade raster (whole trial)', fontsize=12, fontweight='bold')

legend_handles = [
    plt.Line2D([0], [0], color='black', lw=1, label='Saccade onset'),
    plt.Line2D([0], [0], marker='|', color='#1F77B4', markersize=9,
               linewidth=0, label='Stimulus onset'),
    Patch(facecolor='#2CA02C', label='Correct response'),
    Patch(facecolor='#D62728', label='Wrong response'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=5, label='Reward', linewidth=0),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=5,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(
    f'{session}  –  saccade timing per trial (aligned to trial onset)\n'
    f'Grey lines = block boundaries  |  trials clipped at {MAX_T:.0f} s',
    fontsize=10, y=1.01,
)

plt.tight_layout()
out_path = os.path.join(save_dir, f'saccade_raster_trial_{session}.pdf')
fig.savefig(out_path, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {out_path}')
print('Done.')
