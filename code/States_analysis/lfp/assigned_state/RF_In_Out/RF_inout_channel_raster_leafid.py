"""
Color-coded raster showing which specific leaf image is in the RF at each
time step, for each channel in a selected array.

Leaf identities (MorphTarget codes):
  30  →  state 1  (blue)
  49  →  state 2  (red)
  51  →  state 3  (green)
  70  →  state 4  (orange)
   0  →  neither leaf outline covers the channel RF  (grey)
  NaN →  after reward  (light grey)

Excluded trials:
  * Trials containing event 3091 (monkey exits block to collect apple)
  * Trials with no reward
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

session_folders = {
    '20230203': 'Cosmos_20230203_LeafForaging_001',
    '20230208': 'Cosmos_20230208_LeafForaging_001',
    '20230209': 'Cosmos_20230209_LeafForaging_001',
    '20230213': 'Cosmos_20230213_LeafForaging_002',
    '20230214': 'Cosmos_20230214_LeafForaging_001',
}

eye_data_dir   = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/eye_data'
rf_stim_dir    = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping_no_reset/RFarea_stim'
states_dir     = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/states_analysis/processed'
save_dir       = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/RF VR mapping_no_reset/RF_In_Out'
os.makedirs(save_dir, exist_ok=True)

n_stimuli = 5
stim_name = 'ImageStimulus'
TIME_RES  = 0.0167   # s (60 Hz VR recording rate)
MAX_T     = 5.0      # s – trials longer than this are skipped
GAP_ROWS  = 2        # blank rows at block boundaries

ARRAY_IDX = 3        # 0-indexed: 0=Array1 … 5=Array6

# ---- area grouping ---------------------------------------------------------
N_CH = 192
all_ch_idx = np.arange(N_CH)
arrays = np.array_split(all_ch_idx, 6)

area_defs = {
    'V1 periphery': np.concatenate(arrays[:3]),
    'V1 fovea':     arrays[3],
    'V4':           np.concatenate(arrays[4:]),
}

sel_array      = arrays[ARRAY_IDX]
sel_arr_label  = f'Array {ARRAY_IDX + 1}'
sel_area_name  = next(name for name, chs in area_defs.items() if sel_array[0] in chs)
sel_page_label = f'{sel_arr_label} ({sel_area_name}, ch {sel_array[0]+1}–{sel_array[-1]+1})'

# ---- leaf identity → raster state -----------------------------------------
# MorphTarget string → state integer
LEAF_TO_STATE = {'30': 1, '49': 2, '51': 3, '70': 4}
STATE_TO_LABEL = {0: 'Neither', 1: 'Leaf 30', 2: 'Leaf 49', 3: 'Leaf 51', 4: 'Leaf 70'}

# Paired leaves always appear together (30↔70, 49↔51)
MORPH_PAIRS = {'30': '70', '70': '30', '49': '51', '51': '49'}

COLORS = {
    0: '#AAAAAA',   # grey   – neither
    1: '#1F77B4',   # blue   – leaf 30
    2: '#D62728',   # red    – leaf 49
    3: '#2CA02C',   # green  – leaf 51
    4: '#FF7F0E',   # orange – leaf 70
}

cmap = mcolors.ListedColormap([COLORS[k] for k in range(5)], name='leaf_id')
CMAP_BOUNDS = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
NORM = mcolors.BoundaryNorm(CMAP_BOUNDS, cmap.N)

legend_handles = [
    Patch(facecolor=COLORS[1], label='Leaf 30 in RF'),
    Patch(facecolor=COLORS[2], label='Leaf 49 in RF'),
    Patch(facecolor=COLORS[3], label='Leaf 51 in RF'),
    Patch(facecolor=COLORS[4], label='Leaf 70 in RF'),
    Patch(facecolor=COLORS[0], label='Neither in RF'),
    Patch(facecolor='#CCCCCC', label='After reward'),
]

# ============================================================
# 1. PARSE LOG FILE  → stimulus times, event timestamps, leaf identities
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

with TextLog(log_path) as log:
    trial_data = log.get_info_per_trial(return_eventmarkers=True, return_loc=False)

trial_df = pd.DataFrame(trial_data).sort_values('TrialIndex').reset_index(drop=True)
trial_df['TargetIdentity'] = np.where(trial_df['Right'] == 1, 'A', 'B')

# Build per-trial A/B leaf identity (same logic as warping_stim_mem.py)
agg = {'TrialIndex': [], 'A_Identity': [], 'B_Identity': []}
for trial_index in trial_df['TrialIndex'].unique():
    sub = trial_df[trial_df['TrialIndex'] == trial_index]
    a_morphs = sub[sub['TargetIdentity'] == 'A']['MorphTarget'].tolist()
    b_morphs = sub[sub['TargetIdentity'] == 'B']['MorphTarget'].tolist()
    agg['TrialIndex'].append(trial_index)
    agg['A_Identity'].append(' '.join(map(str, a_morphs)))
    agg['B_Identity'].append(' '.join(map(str, b_morphs)))

stim_id_df = pd.DataFrame(agg).sort_values('TrialIndex').reset_index(drop=True)

def fill_gaps(row):
    a_list = row['A_Identity'].split() if row['A_Identity'] else []
    b_list = row['B_Identity'].split() if row['B_Identity'] else []
    if a_list and not b_list:
        b_list = [MORPH_PAIRS.get(a, '') for a in a_list if MORPH_PAIRS.get(a)]
    if b_list and not a_list:
        a_list = [MORPH_PAIRS.get(b, '') for b in b_list if MORPH_PAIRS.get(b)]
    return pd.Series([' '.join(a_list), ' '.join(b_list)])

stim_id_df[['A_Identity', 'B_Identity']] = stim_id_df.apply(fill_gaps, axis=1)

target_stim_per_trial = np.where(trial_df['Right'].values == 1, 'A', 'B')

# Event timestamps
target_onset        = ts[np.where(evt == 3011)[0]]
trial_end_ts        = ts[np.where(evt == 3090)[0]]
block_exit_ts       = ts[np.where(evt == 3091)[0]]
response_correct_ts = ts[np.where(evt == 1)[0]]
response_wrong_ts   = ts[np.where(evt == 2)[0]]

reward_mask    = (evt >= 5000) & (evt <= 5999)
reward_ts_all  = ts[reward_mask]
reward_val_all = evt[reward_mask]

reach_mask    = np.isin(evt, [3013, 3023])
reach_ts_all  = ts[reach_mask]
reach_evt_all = evt[reach_mask]

n_log_trials = len(target_onset)
print(f'  Log trials: {n_log_trials}  |  Block exits: {len(block_exit_ts)}')

n_stim_trials = len(stim_ts) // n_stimuli
aligned_stim_times = []
for itrl in range(n_stim_trials):
    t = stim_ts[itrl * n_stimuli + 3].T
    aligned_stim_times.append(t - t[0])

rew_t_per_trial       = np.full(n_log_trials, np.nan)
reward_code_per_trial = np.zeros(n_log_trials, dtype=int)
for trl in range(n_log_trials):
    t0 = target_onset[trl]
    t1 = trial_end_ts[trl] if trl < len(trial_end_ts) else t0 + 5.0
    in_window = (reward_ts_all >= t0) & (reward_ts_all <= t1)
    if in_window.any():
        first_idx = np.where(in_window)[0][0]
        rew_t_per_trial[trl]       = reward_ts_all[first_idx] - t0
        reward_code_per_trial[trl] = reward_val_all[first_idx]

emissions_path = os.path.join(states_dir, session_folders[session], 'emissions.npy')
if os.path.exists(emissions_path):
    rt_values = np.load(emissions_path).flatten()
    print(f'  RT values loaded: {len(rt_values)}')
else:
    print(f'  WARNING: emissions.npy not found, RT will be NaN')
    rt_values = np.full(n_log_trials, np.nan)

# ============================================================
# 2. TRIAL FLAGS
# ============================================================

n_trials = min(n_log_trials, len(aligned_stim_times))

has_block_exit = np.zeros(n_trials, dtype=bool)
for t_exit in block_exit_ts:
    idx = np.searchsorted(target_onset, t_exit, side='right') - 1
    if 0 <= idx < n_trials:
        has_block_exit[idx] = True

is_response_correct = np.zeros(n_trials, dtype=bool)
is_response_wrong   = np.zeros(n_trials, dtype=bool)
for trl in range(n_trials):
    t0 = target_onset[trl]
    t1 = trial_end_ts[trl] if trl < len(trial_end_ts) else t0 + 5.0
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

correct_mask = is_response_correct[:n_trials] & ~has_block_exit[:n_trials]
wrong_mask   = is_response_wrong[:n_trials]   & ~has_block_exit[:n_trials]
plot_mask    = correct_mask | wrong_mask

print(f'  Correct (non-exit): {correct_mask.sum()}  |  Wrong (non-exit): {wrong_mask.sum()}')

# ============================================================
# 5. LOAD RF DATA
# ============================================================

hdf5_path = os.path.join(rf_stim_dir, session, 'RF_stim_collapse.h5')
print(f'Loading RF data: {hdf5_path}')

t_grid  = np.arange(0, MAX_T + TIME_RES, TIME_RES)
n_tbins = len(t_grid)


def build_leaf_raster(area_ch_indices):
    """
    Returns (raster_matrix, gap_positions, row_outcomes, row_rt).
    raster values: 0=neither, 1=leaf30, 2=leaf49, 3=leaf51, 4=leaf70, NaN=after-reward.
    """
    rows          = []
    row_outcomes  = []
    row_rt        = []
    gap_row       = np.full(n_tbins, np.nan)
    prev_block    = -1
    gap_positions = []

    with h5py.File(hdf5_path, 'r') as hf:
        trial_names = sorted(hf.keys(), key=lambda n: int(n.split('_')[-1]))

        for trl in range(n_trials):
            if not plot_mask[trl]:
                continue
            if trl >= len(trial_names):
                break

            blk = block_ids[trl]
            if blk != prev_block and prev_block != -1:
                for _ in range(GAP_ROWS):
                    rows.append(gap_row.copy())
                    row_outcomes.append('gap')
                    row_rt.append(np.nan)
                    gap_positions.append(len(rows) - 1)
            prev_block = blk

            stim_t = aligned_stim_times[trl]
            rt     = rew_t_per_trial[trl]
            if np.isnan(rt) or rt > MAX_T:
                continue
            rt = np.clip(rt, 0, MAX_T)

            # Leaf identity for this trial
            if trl < len(stim_id_df):
                a_id_str = stim_id_df.iloc[trl]['A_Identity']
                b_id_str = stim_id_df.iloc[trl]['B_Identity']
                a_id = a_id_str.split()[0] if a_id_str else ''
                b_id = b_id_str.split()[0] if b_id_str else ''
            else:
                a_id, b_id = '', ''

            trial_group = hf[trial_names[trl]]
            tp_names    = sorted(trial_group.keys(), key=lambda n: int(n.split('_')[-1]))
            n_tp        = min(len(tp_names), len(stim_t))

            tp_times  = stim_t[:n_tp]
            tp_states = np.full(n_tp, np.nan)

            for tp_i in range(n_tp):
                tp_group    = trial_group[tp_names[tp_i]]
                is_collapse = bool(tp_group.attrs.get('collapsed_case', False))

                if is_collapse and len(reach_ts_all) > 0:
                    abs_tp_time         = target_onset[trl] + stim_t[tp_i]
                    nearest_idx         = np.argmin(np.abs(reach_ts_all - abs_tp_time))
                    collapse_reached    = 'A' if reach_evt_all[nearest_idx] == 3013 else 'B'
                else:
                    collapse_reached = None

                # Majority-vote across channels: which leaf state wins?
                leaf_votes = {k: 0 for k in range(5)}
                n_valid    = 0

                for ch_idx in area_ch_indices:
                    prefix  = f'Point_{int(ch_idx) + 1}_'
                    pt_name = next(
                        (nm for nm in tp_group.keys() if nm.startswith(prefix)), None
                    )
                    if pt_name is None:
                        continue
                    pt_grp = tp_group[pt_name]
                    in_A   = bool(pt_grp['inside_transformed_outline_A'][()])
                    in_B   = bool(pt_grp['inside_transformed_outline_B'][()])
                    n_valid += 1

                    if is_collapse:
                        # Both stims are at the same position; in_A (== in_B) tells us
                        # whether that position is in the RF.  Use collapse_reached to
                        # identify which leaf was shown.
                        if in_A:
                            leaf_in_rf = LEAF_TO_STATE.get(
                                a_id if collapse_reached == 'A' else b_id, 0)
                        else:
                            leaf_in_rf = 0
                    elif in_A:
                        leaf_in_rf = LEAF_TO_STATE.get(a_id, 0)
                    elif in_B:
                        leaf_in_rf = LEAF_TO_STATE.get(b_id, 0)
                    else:
                        leaf_in_rf = 0

                    leaf_votes[leaf_in_rf] += 1

                if n_valid == 0:
                    tp_states[tp_i] = np.nan
                else:
                    tp_states[tp_i] = float(max(leaf_votes, key=leaf_votes.__getitem__))

            # Map RF time points onto common t_grid (step function)
            row = np.full(n_tbins, np.nan)
            for ti, t in enumerate(t_grid):
                if t > rt:
                    break
                before = np.where(tp_times <= t)[0]
                if len(before) == 0:
                    row[ti] = tp_states[0] if len(tp_states) > 0 else 0.0
                else:
                    row[ti] = tp_states[before[-1]]

            rows.append(row)
            row_outcomes.append('correct' if correct_mask[trl] else 'wrong')
            trial_rt = rt_values[trl] if trl < len(rt_values) else np.nan
            row_rt.append(trial_rt)

    return np.array(rows), gap_positions, row_outcomes, row_rt


def get_good_channels(ch_indices):
    good = []
    with h5py.File(hdf5_path, 'r') as hf:
        trial_names = sorted(hf.keys(), key=lambda n: int(n.split('_')[-1]))
        for trl_name in trial_names[:10]:
            trl_grp  = hf[trl_name]
            tp_names = sorted(trl_grp.keys(), key=lambda n: int(n.split('_')[-1]))
            if not tp_names:
                continue
            tp_grp = trl_grp[tp_names[0]]
            for ch in ch_indices:
                if ch not in good:
                    prefix = f'Point_{int(ch) + 1}_'
                    if any(nm.startswith(prefix) for nm in tp_grp.keys()):
                        good.append(ch)
    return sorted(good)


# ============================================================
# 6. PLOT  – per-channel PDF
# ============================================================

outcome_handles = [
    Patch(facecolor='#2CA02C', label='Correct response'),
    Patch(facecolor='#D62728', label='Wrong response'),
]
OUTCOME_COLOR = {'correct': '#2CA02C', 'wrong': '#D62728'}
MARKER_X_FRAC = 0.012

print(f'\nDetecting good channels in {sel_arr_label} …')
good_channels = get_good_channels(sel_array)
print(f'  Good channels ({len(good_channels)}): ch {[c+1 for c in good_channels]}')

ch_rasters = {}

out_path = os.path.join(
    save_dir,
    f'RF_inout_leafid_array{ARRAY_IDX+1}_per_channel_{session}.pdf'
)

with PdfPages(out_path) as pdf:
    for ch in good_channels:
        ch_label = f'{sel_arr_label} – Ch {ch+1} ({sel_area_name})'
        print(f'  Building raster for {ch_label} …')
        raster, gap_pos, row_outcomes, row_rt = build_leaf_raster([ch])
        ch_rasters[ch] = raster

        fig, ax = plt.subplots(
            figsize=(14, max(8, min(plot_mask.sum() * 0.08 + 2, 40)))
        )

        if raster.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(ch_label)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            continue

        n_rows = raster.shape[0]
        raster_display = np.where(np.isnan(raster), -1, raster)
        grey_mask      = np.isnan(raster)

        ax.imshow(
            raster_display,
            aspect='auto', interpolation='none',
            cmap=cmap, norm=NORM,
            extent=[t_grid[0], t_grid[-1], n_rows - 0.5, -0.5],
            origin='upper',
        )

        # grey overlay for after-reward bins
        grey_overlay = np.ma.masked_where(~grey_mask, np.ones_like(raster))
        ax.imshow(
            grey_overlay,
            aspect='auto', interpolation='none',
            cmap=mcolors.ListedColormap(['#CCCCCC']),
            vmin=0, vmax=1,
            extent=[t_grid[0], t_grid[-1], n_rows - 0.5, -0.5],
            origin='upper',
        )

        for gp in gap_pos:
            ax.axhline(gp, color='white', linewidth=1.5)

        if len(gap_pos) > 0:
            boundary_ys  = gap_pos[::GAP_ROWS]
            gap_rows_arr = np.array([-1] + list(boundary_ys) + [n_rows])
            for blk_i in range(len(gap_rows_arr) - 1):
                mid = (gap_rows_arr[blk_i] + gap_rows_arr[blk_i + 1]) / 2
                ax.text(t_grid[-1] * 0.99, mid, f'Blk {blk_i + 1}',
                        ha='right', va='center', fontsize=6, color='white',
                        fontweight='bold')

        ax.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.6)

        # outcome markers (right edge)
        marker_x = t_grid[-1] + (t_grid[-1] - t_grid[0]) * MARKER_X_FRAC
        for row_i, outcome in enumerate(row_outcomes):
            if outcome in OUTCOME_COLOR:
                ax.plot(marker_x, row_i, 's',
                        color=OUTCOME_COLOR[outcome],
                        markersize=4, clip_on=False, zorder=5)

        # RT markers (white dot)
        rt_xs = [rt for rt in row_rt if not np.isnan(rt)]
        rt_ys = [row_i for row_i, rt in enumerate(row_rt) if not np.isnan(rt)]
        if rt_xs:
            ax.scatter(rt_xs, rt_ys, color='white', s=8, zorder=6,
                       linewidths=0.5, edgecolors='black')

        ax.set_xlim(t_grid[0], t_grid[-1])
        ax.set_xlabel('Time from stimulus onset (s)', fontsize=10)
        ax.set_ylabel('Trial (oldest → newest)', fontsize=10)
        ax.set_title(ch_label, fontsize=12, fontweight='bold')

        rt_handle = plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor='white', markeredgecolor='black',
                               markersize=5, label='Reaction time', linewidth=0)
        fig.legend(
            handles=legend_handles + outcome_handles + [rt_handle],
            loc='lower center', ncol=9,
            fontsize=9, framealpha=0.9,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.suptitle(
            f'{session}  –  {ch_label}  –  Leaf identity in RF (single channel)\n'
            f'White gaps = block boundaries  |  Trials > {MAX_T:.0f} s skipped',
            fontsize=10, y=1.01,
        )

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f'\nSaved: {out_path}')

# ============================================================
# 7. OVERLAP SUMMARY – which morph (49, 51, 70) dominates per channel
# ============================================================

MORPH_STATES = {2: 'Leaf 49', 3: 'Leaf 51', 4: 'Leaf 70'}

print('\n--- RF Overlap Summary: Morph 49 vs 51 vs 70 per channel ---')
print(f'{"Channel":<10} {"Leaf49 bins":>12} {"Leaf51 bins":>12} {"Leaf70 bins":>12} {"Winner":>10}')
print('-' * 60)

for ch in good_channels:
    raster = ch_rasters.get(ch)
    if raster is None or raster.size == 0:
        continue
    counts = {s: int(np.sum(raster == s)) for s in MORPH_STATES}
    winner = MORPH_STATES[max(counts, key=counts.get)]
    print(f'  Ch {ch+1:<6} {counts[2]:>12} {counts[3]:>12} {counts[4]:>12} {winner:>10}')

print('\nDone.')
