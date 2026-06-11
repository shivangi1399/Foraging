"""
Color-coded raster of RF-in / RF-out status across trials for each array (channel info condensed), with block boundaries indicated 
by horizontal gaps.

Excluded trials
---------------
  * Trials containing event 3091 (monkey exits block to collect apple)
  * Trials with no reward (Reward == 5000 or no reward code found)
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(1, '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/functions/unreal_logfile')
from parse_logfile import TextLog

# ============================================================
# CONFIG  (change session to run a different recording)
# ============================================================

session = '20230214'

session_logfiles = {
    '20230203': '2023_02_03-11_35_57_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230208': '2023_02_08-10_58_17_Cosmos_LeafForaging_001_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230209': '2023_02_09-11_19_51_Cosmos_LeafForaging_001_KAS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230213': '2023_02_13-11_13_43_Cosmos_LeafForaging_002_MS_GrassyLandscapeWithBackgroundDark_Cont.log',
    '20230214': '2023_02_14-11_42_27_Cosmos_LeafForaging_001_PAF_GrassyLandscapeWithBackgroundDark_Cont.log',
}

session_folders = {
    '20230203': 'Cosmos_20230203_LeafForaging_001',
    '20230208': 'Cosmos_20230208_LeafForaging_001',
    '20230209': 'Cosmos_20230209_LeafForaging_001',
    '20230213': 'Cosmos_20230213_LeafForaging_002',
    '20230214': 'Cosmos_20230214_LeafForaging_001',
}

# RF session closest to the recording session
rf_session_map = {
    '20230203': '20230202',
    '20230208': '20230209',
    '20230209': '20230209',
    '20230213': '20230209',
    '20230214': '20230209',
}

eye_data_dir   = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data'
rf_stim_dir    = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping_no_reset/RFarea_stim'
trial_info_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/full_length'
states_dir     = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis/processed'
rf_center_dir  = '/cs/projects/MWzeronoise/Analysis/4Shivangi/RF_VR_mapping_no_reset/RFs/center_radius'
save_dir       = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/RF VR mapping_no_reset/RF_In_Out'
os.makedirs(save_dir, exist_ok=True)

n_stimuli    = 5         # stimuli per trial in the log
stim_name    = 'ImageStimulus'
TIME_RES     = 0.0167    # s  –  matches the 60 Hz VR RF mapping recording rate
MAX_T        = 5.0      # s  –  trials longer than this are skipped; display clipped here
GAP_ROWS     = 2         # blank rows inserted at each block boundary

# ---- area grouping (192 channels, 6 equal arrays) --------------------------
N_CH = 192
all_ch_idx = np.arange(N_CH)
arrays = np.array_split(all_ch_idx, 6)   # arrays[0..5]

area_defs = {
    'V1 periphery': np.concatenate(arrays[:3]),   # Arrays 1-3  (ch 1-96)
    'V1 fovea':     arrays[3],                    # Array 4     (ch 97-128)
    'V4':           np.concatenate(arrays[4:]),   # Arrays 5-6  (ch 129-192)
}

# ---- color map  ------------------------------------------------------------
# 0=neither(orange), 1=target(purple), 2=distractor(turquoise), nan=after-reward(grey)
COLORS = {
    0: '#F4831F',   # orange  – neither
    1: '#7B2D8B',   # purple  – target
    2: '#00A896',   # turquoise – distractor
}
cmap_vals = [COLORS[0], COLORS[1], COLORS[2]]
cmap = mcolors.ListedColormap(cmap_vals, name='rf_state')
CMAP_BOUNDS = [-0.5, 0.5, 1.5, 2.5]
NORM = mcolors.BoundaryNorm(CMAP_BOUNDS, cmap.N)

legend_handles = [
    Patch(facecolor=COLORS[1], label='Target in RF'),
    Patch(facecolor=COLORS[2], label='Distractor in RF'),
    Patch(facecolor=COLORS[0], label='Neither'),
    Patch(facecolor='#CCCCCC', label='After reward'),
]

# ============================================================
# HELPERS
# ============================================================

def area_rf_state(in_target_ch, in_distract_ch, n_valid):
    """
    Determine the RF state for one area at one time point via majority vote.
    Returns 1 (target), 2 (distractor), or 0 (neither).
    """
    if n_valid == 0:
        return np.nan
    n_neither = n_valid - in_target_ch - in_distract_ch
    # whichever category has the most channels wins
    counts = [n_neither, in_target_ch, in_distract_ch]
    return int(np.argmax(counts))   # 0=neither, 1=target, 2=distractor


# ============================================================
# 1. PARSE LOG FILE  → aligned stimulus times, event timestamps
# ============================================================

log_path = os.path.join(eye_data_dir, session, session_logfiles[session])
print(f'Parsing log file: {log_path}')

stim_ts = []
with TextLog(log_path) as log:
    log.make_id_struct()
    evt, ts, evt_desc, true_ts = log.parse_eventmarkers()
    indx = [ii for ii, name in enumerate(log.all_ids['name']) if name.startswith(stim_name)]
    for ii, istim in enumerate(indx):
        if ii + n_stimuli == len(indx):
            break
        this_id = log.all_ids[istim]
        next_id = log.all_ids[indx[ii + n_stimuli]]
        _, pos_ts = log.parse_spherical(obj_id=this_id['id'], st=this_id['start'], end=next_id['start'])
        stim_ts.append(pos_ts)

# Target identity per trial:
# Stim A is always on the right. Right==1 means target was on the right → A is target.
with TextLog(log_path) as log:
    trial_data = log.get_info_per_trial(return_eventmarkers=True, return_loc=False)

trial_df = pd.DataFrame(trial_data).sort_values('TrialIndex').reset_index(drop=True)
target_stim_per_trial = np.where(trial_df['Right'].values == 1, 'A', 'B')

# event timestamps
target_onset = ts[np.where(evt == 3011)[0]]   # stim on  (align to 0)
trial_end_ts = ts[np.where(evt == 3090)[0]]   # EndTrial
block_exit_ts = ts[np.where(evt == 3091)[0]]  # block exit (apple)

# response outcome events
response_correct_ts = ts[np.where(evt == 1)[0]]   # ResponseCorrect
response_wrong_ts   = ts[np.where(evt == 2)[0]]   # ResponseWrong

# reward events (5000–5999): find their absolute timestamps
reward_mask = (evt >= 5000) & (evt <= 5999)
reward_ts_all = ts[reward_mask]
reward_val_all = evt[reward_mask]

# reach events: 3013 = stim A reached, 3023 = stim B reached
reach_mask = np.isin(evt, [3013, 3023])
reach_ts_all = ts[reach_mask]
reach_evt_all = evt[reach_mask]

n_log_trials = len(target_onset)
print(f'  Log trials: {n_log_trials}  |  Block exits: {len(block_exit_ts)}')

# Aligned RF time series (relative to stim onset)
n_stim_trials = len(stim_ts) // n_stimuli
aligned_stim_times = []
for itrl in range(n_stim_trials):
    t = stim_ts[itrl * n_stimuli + 3].T
    aligned_stim_times.append(t - t[0])

# Compute reward time for each trial: time from stim onset to reward delivery
rew_t_per_trial = np.full(n_log_trials, np.nan)
reward_code_per_trial = np.zeros(n_log_trials, dtype=int)

for trl in range(n_log_trials):
    t0 = target_onset[trl]
    t1 = trial_end_ts[trl] if trl < len(trial_end_ts) else t0 + 5.0
    # find reward events between stim onset and trial end
    in_window = (reward_ts_all >= t0) & (reward_ts_all <= t1)
    if in_window.any():
        first_idx = np.where(in_window)[0][0]
        rew_t_per_trial[trl] = reward_ts_all[first_idx] - t0
        reward_code_per_trial[trl] = reward_val_all[first_idx]

# RT values from emissions.npy (seconds from stim onset, one value per trial)
emissions_path = os.path.join(states_dir, session_folders[session], 'emissions.npy')
if os.path.exists(emissions_path):
    rt_values = np.load(emissions_path).flatten()
    print(f'  RT values loaded: {len(rt_values)}')
else:
    print(f'  WARNING: emissions.npy not found at {emissions_path}, RT will be NaN')
    rt_values = np.full(n_log_trials, np.nan)

# ============================================================
# 2. DERIVE TRIAL FLAGS directly from parsed log event timestamps
# ============================================================

n_trials = min(n_log_trials, len(aligned_stim_times))

# block-exit (3091): map each exit timestamp to its trial index
has_block_exit = np.zeros(n_trials, dtype=bool)
for t_exit in block_exit_ts:
    idx = np.searchsorted(target_onset, t_exit, side='right') - 1
    if 0 <= idx < n_trials:
        has_block_exit[idx] = True

# rewarded trials: reward event (5000–5999) found between stim onset and trial end
has_reward = ~np.isnan(rew_t_per_trial[:n_trials])

# per-trial correct / wrong response from event markers 1 / 2
is_response_correct = np.zeros(n_trials, dtype=bool)
is_response_wrong   = np.zeros(n_trials, dtype=bool)
for trl in range(n_trials):
    t0 = target_onset[trl]
    t1 = trial_end_ts[trl] if trl < len(trial_end_ts) else t0 + 5.0
    if np.any((response_correct_ts >= t0) & (response_correct_ts <= t1)):
        is_response_correct[trl] = True
    if np.any((response_wrong_ts >= t0) & (response_wrong_ts <= t1)):
        is_response_wrong[trl] = True

print(f'  Trials: {n_trials}  |  Block-exit: {has_block_exit.sum()}  |  Rewarded: {has_reward.sum()}'
      f'  |  ResponseCorrect: {is_response_correct.sum()}  |  ResponseWrong: {is_response_wrong.sum()}')

# ============================================================
# 3. BLOCK STRUCTURE
# ============================================================

# A block ends at a trial containing event 3091.
# Trials AFTER a 3091 trial start the next block.
block_ids = np.zeros(n_trials, dtype=int)   # block index per trial
current_block = 0
for trl in range(n_trials):
    block_ids[trl] = current_block
    if has_block_exit[trl]:
        current_block += 1

block_boundaries = sorted(set(block_ids))   # block indices
# trial indices that are the LAST trial of each block (contain 3091)
block_end_trials = np.where(has_block_exit[:n_trials])[0]

print(f'  Blocks found: {len(block_boundaries)}')

# ============================================================
# 4. FILTER: keep only correct, non-block-exit trials
# ============================================================

correct_mask = (
    is_response_correct[:n_trials] &   # event 1 (ResponseCorrect) present in trial
    ~has_block_exit[:n_trials]         # not a block-exit trial
)
wrong_mask = (
    is_response_wrong[:n_trials] &     # event 2 (ResponseWrong) present in trial
    ~has_block_exit[:n_trials]
)
plot_mask = correct_mask | wrong_mask  # all non-exit trials with a response outcome

print(f'  Correct (non-exit) trials: {correct_mask.sum()}  |  Wrong (non-exit): {wrong_mask.sum()}')

# ============================================================
# 5. LOAD RF IN/OUT DATA
# ============================================================

hdf5_path = os.path.join(rf_stim_dir, session, 'RF_stim_collapse.h5')
print(f'Loading RF data: {hdf5_path}')

# time grid for raster (common across all trials)
t_grid = np.arange(0, MAX_T + TIME_RES, TIME_RES)
n_tbins = len(t_grid)

# For each area, build a list of rows (one per correct trial),
# with block-gap rows inserted between blocks.
# Row values: 0=neither, 1=target, 2=distractor, NaN=after reward

def build_raster(area_ch_indices):
    """
    Returns (raster_matrix, gap_positions, row_outcomes, row_rt)
    raster_matrix : (n_display_rows, n_tbins)
    row_outcomes  : list of 'correct', 'wrong', or 'gap' for each display row
    row_rt        : list of RT (s from stim onset) or NaN for gap rows
    """
    rows = []           # list of 1-D arrays (length n_tbins)
    row_outcomes = []   # outcome label per display row
    row_rt = []         # RT (s) per display row, NaN for gap rows
    gap_row = np.full(n_tbins, np.nan)   # visual separator
    prev_block = -1
    gap_positions = []  # y-positions of gap rows (for tick reference)

    with h5py.File(hdf5_path, 'r') as hf:
        trial_names = sorted(hf.keys(), key=lambda n: int(n.split('_')[-1]))

        for trl in range(n_trials):
            if not plot_mask[trl]:
                continue
            if trl >= len(trial_names):
                break

            # insert gap at block boundary
            blk = block_ids[trl]
            if blk != prev_block and prev_block != -1:
                for _ in range(GAP_ROWS):
                    rows.append(gap_row.copy())
                    row_outcomes.append('gap')
                    row_rt.append(np.nan)
                    gap_positions.append(len(rows) - 1)
            prev_block = blk

            # trial timing — skip trials longer than MAX_T
            stim_t = aligned_stim_times[trl]  # RF time points (seconds from onset)
            rt      = rew_t_per_trial[trl]
            if np.isnan(rt) or rt > MAX_T:
                continue
            rt = np.clip(rt, 0, MAX_T)

            # target identity: A if Right==1 (target on right), B otherwise
            target_stim = target_stim_per_trial[trl] if trl < len(target_stim_per_trial) else None
            trial_group = hf[trial_names[trl]]

            # read RF in/out per time point for area channels
            tp_names = sorted(trial_group.keys(), key=lambda n: int(n.split('_')[-1]))
            n_tp = min(len(tp_names), len(stim_t))

            # build per-time-point state (0/1/2)
            tp_times  = stim_t[:n_tp]
            tp_states = np.full(n_tp, np.nan)

            for tp_i in range(n_tp):
                tp_group = trial_group[tp_names[tp_i]]
                is_collapse = bool(tp_group.attrs.get('collapsed_case', False))

                # For collapse time points, resolve which stimulus was reached
                # by finding the nearest 3013 (stim A) or 3023 (stim B) event
                if is_collapse and len(reach_ts_all) > 0:
                    abs_tp_time = target_onset[trl] + stim_t[tp_i]
                    nearest_idx = np.argmin(np.abs(reach_ts_all - abs_tp_time))
                    collapse_reached_stim = 'A' if reach_evt_all[nearest_idx] == 3013 else 'B'
                else:
                    collapse_reached_stim = None

                n_valid = 0
                n_target_in = 0
                n_distract_in = 0

                for ch_idx in area_ch_indices:
                    target_prefix = f'Point_{int(ch_idx) + 1}_'
                    pt_name = next(
                        (nm for nm in tp_group.keys() if nm.startswith(target_prefix)),
                        None
                    )
                    if pt_name is None:
                        continue
                    pt_grp = tp_group[pt_name]
                    in_A = bool(pt_grp['inside_transformed_outline_A'][()])
                    in_B = bool(pt_grp['inside_transformed_outline_B'][()])
                    n_valid += 1

                    if is_collapse:
                        if collapse_reached_stim == 'A':
                            in_target   = (target_stim == 'A')
                            in_distract = (target_stim == 'B')
                        elif collapse_reached_stim == 'B':
                            in_target   = (target_stim == 'B')
                            in_distract = (target_stim == 'A')
                        else:
                            in_target   = False
                            in_distract = False
                    elif target_stim == 'A':
                        in_target = in_A
                        in_distract = in_B and not in_A
                    elif target_stim == 'B':
                        in_target = in_B
                        in_distract = in_A and not in_B
                    else:
                        in_target = False
                        in_distract = False

                    if in_target:
                        n_target_in += 1
                    elif in_distract:
                        n_distract_in += 1

                tp_states[tp_i] = area_rf_state(n_target_in, n_distract_in, n_valid)

            # map RF time points onto common t_grid (step function)
            row = np.full(n_tbins, np.nan)
            for ti, t in enumerate(t_grid):
                if t > rt:
                    break
                # find the last RF time point ≤ t
                before = np.where(tp_times <= t)[0]
                if len(before) == 0:
                    row[ti] = tp_states[0] if len(tp_states) > 0 else 0
                else:
                    row[ti] = tp_states[before[-1]]

            rows.append(row)
            row_outcomes.append('correct' if correct_mask[trl] else 'wrong')
            trial_rt = rt_values[trl] if trl < len(rt_values) else np.nan
            row_rt.append(trial_rt)

    return np.array(rows), gap_positions, row_outcomes, row_rt


# ============================================================
# 6. PLOT  –  one PDF per session, one page per array
# ============================================================

out_path = os.path.join(save_dir, f'RF_inout_block_raster_{session}.pdf')

with PdfPages(out_path) as pdf:
    for arr_i, ch_idx in enumerate(arrays, 1):
        area_name = next(name for name, chs in area_defs.items() if ch_idx[0] in chs)
        page_label = f'Array {arr_i} ({area_name}, ch {ch_idx[0]+1}–{ch_idx[-1]+1})'
        print(f'  Building raster for {page_label} …')
        raster, gap_pos, row_outcomes, row_rt = build_raster(ch_idx)

        fig, ax = plt.subplots(
            figsize=(14, max(8, min(plot_mask.sum() * 0.08 + 2, 40)))
        )

        if raster.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(page_label)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            continue

        n_rows = raster.shape[0]

        # plot with pcolormesh
        # NaN → grey (after reward)
        raster_display = np.where(np.isnan(raster), -1, raster)   # -1 → grey
        grey_mask      = np.isnan(raster)

        # draw main colored raster
        ax.imshow(
            raster_display,
            aspect='auto',
            interpolation='none',
            cmap=cmap,
            norm=NORM,
            extent=[t_grid[0], t_grid[-1], n_rows - 0.5, -0.5],
            origin='upper',
        )

        # overlay grey for after-reward cells using a masked array
        grey_overlay = np.ma.masked_where(~grey_mask, np.ones_like(raster))
        ax.imshow(
            grey_overlay,
            aspect='auto',
            interpolation='none',
            cmap=mcolors.ListedColormap(['#CCCCCC']),
            vmin=0, vmax=1,
            extent=[t_grid[0], t_grid[-1], n_rows - 0.5, -0.5],
            origin='upper',
        )

        # draw horizontal white lines at block gaps
        for gp in gap_pos:
            ax.axhline(gp, color='white', linewidth=1.5)

        # annotate block numbers
        if len(gap_pos) > 0:
            boundary_ys = gap_pos[::GAP_ROWS]
            gap_rows_arr = np.array([-1] + list(boundary_ys) + [n_rows])
            for blk_i in range(len(gap_rows_arr) - 1):
                mid = (gap_rows_arr[blk_i] + gap_rows_arr[blk_i + 1]) / 2
                ax.text(t_grid[-1] * 0.99, mid, f'Blk {blk_i + 1}',
                        ha='right', va='center', fontsize=6, color='white',
                        fontweight='bold')

        # stim-onset line
        ax.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.6)

        # outcome markers: green square = correct, red square = wrong, at row right edge
        MARKER_X = t_grid[-1] + (t_grid[-1] - t_grid[0]) * 0.012
        OUTCOME_COLOR = {'correct': '#2CA02C', 'wrong': '#D62728'}
        for row_i, outcome in enumerate(row_outcomes):
            if outcome in OUTCOME_COLOR:
                ax.plot(
                    MARKER_X, row_i,
                    's', color=OUTCOME_COLOR[outcome],
                    markersize=4, clip_on=False, zorder=5,
                )

        # RT markers: white dot at reaction time for each trial row
        rt_xs = [rt for rt in row_rt if not np.isnan(rt)]
        rt_ys = [row_i for row_i, rt in enumerate(row_rt) if not np.isnan(rt)]
        if rt_xs:
            ax.scatter(rt_xs, rt_ys, color='white', s=8, zorder=6,
                       linewidths=0.5, edgecolors='black')

        ax.set_xlim(t_grid[0], t_grid[-1])
        ax.set_xlabel('Time from stimulus onset (s)', fontsize=10)
        ax.set_ylabel('Trial (stacked, oldest→newest)', fontsize=10)
        ax.set_title(page_label, fontsize=12, fontweight='bold')

        # legend
        outcome_handles = [
            Patch(facecolor='#2CA02C', label='Correct response'),
            Patch(facecolor='#D62728', label='Wrong response'),
        ]
        rt_handle = plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor='white', markeredgecolor='black',
                               markersize=5, label='Reaction time', linewidth=0)
        fig.legend(
            handles=legend_handles + outcome_handles + [rt_handle],
            loc='lower center',
            ncol=7,
            fontsize=9,
            framealpha=0.9,
            bbox_to_anchor=(0.5, -0.02),
        )

        fig.suptitle(
            f'{session}  –  {page_label}  –  RF in/out raster (correct + wrong trials, stim onset → reward)\n'
            f'White gaps = block boundaries  |  Color = majority vote across area channels  |  Trials > {MAX_T:.0f} s skipped',
            fontsize=11, y=1.01
        )

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f'\nSaved: {out_path}')
print('Done.')
