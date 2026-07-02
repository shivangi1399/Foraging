"""
=============================================================================================
Visualise the per-array / per-channel GLM contribution summaries.
=============================================================================================
Reads the CSVs written by RunByArray.py (under <results_dir>/_contribution_summaries/) and makes,
for each array, a channel x regressor-family heatmap of unique dR2 -- so you can see at a glance
which families encode on which channels, and how consistent that is across the array. Each array
figure also carries a per-channel full-model R2 bar. A final overview figure shows the mean dR2 per
family for every array together.

dR2 colour: red = positive unique contribution (family helps prediction), blue = negative (hurts /
below chance). The `dummy` column is the shuffled-control null -- compare every other column to it.

Light / login-node script (no acme):  python PlotArraySummary.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# -------------------------
# Config
# -------------------------
# make glm_config (single source of truth for the output tree + sampling rate) importable
for _d in (os.path.dirname(os.path.abspath(__file__)),
           os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    if os.path.exists(os.path.join(_d, 'glm_config.py')):
        sys.path.insert(0, _d)
        break
from glm_config import RESULTS_DIR, PLOTS_DIR
results_dir = RESULTS_DIR
plots_base = PLOTS_DIR
SESSION = '20230214'    
SUMMARY_DIR = os.path.join(results_dir, SESSION, '_contribution_summaries')
plots_dir = os.path.join(plots_base, SESSION, '_contribution_summaries')

METRICS = ['dR2', 'tracevar']   # make a figure set per metric: 'dR2' (unique, honest) + 'tracevar' (gross in-sample swing)
ALPHA = 0.05            # significance level for the circular-shift permutation test (pval:: columns)


def load_summary():
    """Prefer the combined CSV; fall back to concatenating per-array CSVs."""
    combined = os.path.join(SUMMARY_DIR, 'all_arrays_contributions.csv')
    if os.path.exists(combined):
        return pd.read_csv(combined)
    parts = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(SUMMARY_DIR, 'array*_contributions.csv')))]
    if not parts:
        raise SystemExit(f'No summary CSVs found in {SUMMARY_DIR} (run RunByArray.py first)')
    return pd.concat(parts, ignore_index=True)


def display_labels(fams):
    """Tick labels for display. The `state` family is the merged per-trial state levels
    (state_0/state_2/... scored together in RegressorContributions.py), so flag it as grouped."""
    return [f + ' (grouped)' if f == 'state' else f for f in fams]


def sym_vlim(values):
    """Robust symmetric colour limit: 98th pct of |finite values|, always > 0."""
    v = np.abs(np.asarray(values, dtype=float).ravel())
    v = v[np.isfinite(v)]
    return max(float(np.percentile(v, 98)), 1e-9) if v.size else 1e-9


def family_order(df, cols):
    """Consistent family ordering across arrays: by global mean metric (strongest first), dummy last."""
    means = df[cols].mean().sort_values(ascending=False)
    fams = [c.split('::')[1] for c in means.index]
    if 'dummy' in fams:                       # keep dummy at the far right as the reference column
        fams = [f for f in fams if f != 'dummy'] + ['dummy']
    return fams


def matrix_for_array(df_arr, fams, metric):
    """channels x families matrix of `metric`, rows sorted by channel."""
    df_arr = df_arr.sort_values('channel')
    chans = df_arr['channel'].to_numpy()
    M = np.full((len(chans), len(fams)), np.nan)
    for r, (_, row) in enumerate(df_arr.iterrows()):
        for cci, fam in enumerate(fams):
            col = f'{metric}::{fam}'
            if col in df_arr.columns:
                M[r, cci] = row[col]
    return chans, M


def plot_array(arr_idx, df_arr, fams, metric, vlim):
    chans, M = matrix_for_array(df_arr, fams, metric)
    nrow = max(len(chans), 1)
    fig, (axh, axr) = plt.subplots(
        1, 2, figsize=(max(8, 0.45 * len(fams) + 3), max(3, 0.28 * nrow + 1.5)),
        gridspec_kw={'width_ratios': [len(fams), 4]}, squeeze=True)

    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    im = axh.imshow(M, aspect='auto', cmap='RdBu_r', norm=norm)
    axh.set_xticks(range(len(fams)))
    axh.set_xticklabels(display_labels(fams), rotation=90, fontsize=7)
    axh.set_yticks(range(len(chans)))
    axh.set_yticklabels(chans, fontsize=7)
    axh.set_ylabel('channel')
    axh.set_title(f'Array {arr_idx}: unique {metric} per channel  (dot = p<{ALPHA:g}, perm)')
    # mark cells significant by the circular-shift permutation test (family_p < ALPHA)
    _, P = matrix_for_array(df_arr, fams, 'pval')
    ys, xs = np.where(np.isfinite(P) & (P < ALPHA))
    if xs.size:
        axh.scatter(xs, ys, marker='o', s=6, color='k', edgecolors='white',
                    linewidths=0.3, zorder=3)
    if 'dummy' in fams:                       # mark the null-control column
        axh.axvline(len(fams) - 1.5, color='k', lw=1.0)
    fig.colorbar(im, ax=axh, fraction=0.025, pad=0.02, label=metric)

    # per-channel full-model R2 bar, aligned to the heatmap rows
    r2 = df_arr.sort_values('channel')['full_R2'].to_numpy()
    axr.barh(range(len(chans)), r2, color='tab:gray')
    axr.set_ylim(axh.get_ylim())
    axr.set_yticks([])
    axr.set_xlabel('full R2')
    axr.set_title('model R2')

    fig.tight_layout()
    out = os.path.join(plots_dir, f'array{arr_idx}_{metric}_heatmap.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  saved {out}')


def plot_overview(df, fams, metric):
    """arrays x families heatmap of the mean metric across each array's channels."""
    arrs = sorted(df['array'].unique())
    M = np.full((len(arrs), len(fams)), np.nan)
    Fp = np.full((len(arrs), len(fams)), np.nan)   # fraction of the array's channels significant
    for r, a in enumerate(arrs):
        sub = df[df['array'] == a]
        for cci, fam in enumerate(fams):
            col = f'{metric}::{fam}'
            if col in df.columns:
                M[r, cci] = sub[col].mean()
            pcol = f'pval::{fam}'
            if pcol in df.columns:
                pv = sub[pcol].to_numpy(dtype=float)
                pv = pv[np.isfinite(pv)]
                if pv.size:
                    Fp[r, cci] = float(np.mean(pv < ALPHA))
    vlim = sym_vlim(M)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(fams) + 2), 0.5 * len(arrs) + 2))
    im = ax.imshow(M, aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim))
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(display_labels(fams), rotation=90, fontsize=7)
    ax.set_yticks(range(len(arrs))); ax.set_yticklabels([f'array {a}' for a in arrs])
    ax.set_title(f'Mean unique {metric} per family, by array  (dot = >50% channels p<{ALPHA:g})')
    # mark cells where a majority of the array's channels are significant (perm test)
    ys, xs = np.where(np.isfinite(Fp) & (Fp >= 0.5))
    if xs.size:
        ax.scatter(xs, ys, marker='o', s=10, color='k', edgecolors='white',
                   linewidths=0.4, zorder=3)
    if 'dummy' in fams:
        ax.axvline(len(fams) - 1.5, color='k', lw=1.0)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=f'mean {metric}')
    fig.tight_layout()
    out = os.path.join(plots_dir, f'overview_mean_{metric}_by_array.pdf')
    fig.savefig(out); plt.close(fig)
    print(f'  saved {out}')


if __name__ == '__main__':
    os.makedirs(plots_dir, exist_ok=True)
    df = load_summary()
    print(f'Loaded {len(df)} channels across arrays {sorted(df["array"].unique())}')

    for metric in METRICS:
        cols = [c for c in df.columns if c.startswith(f'{metric}::')]
        if not cols:
            print(f'  (no {metric}:: columns in the summary CSV -- skipping)')
            continue
        fams = family_order(df, cols)                     # per-metric ordering (strongest first)
        vlim = sym_vlim(df[cols].to_numpy())              # shared colour scale across arrays for this metric
        print(f'-- {metric}: plotting {len(df["array"].unique())} arrays + overview')
        for a in sorted(df['array'].unique()):
            plot_array(int(a), df[df['array'] == a], fams, metric, vlim)
        plot_overview(df, fams, metric)
    print(f'Done -> {plots_dir}')
