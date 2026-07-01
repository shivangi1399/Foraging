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
results_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp/all_trials/full_length/GLM'
SUMMARY_DIR = os.path.join(results_dir, '_contribution_summaries')
plots_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_lfp/all_trials/full_length/GLM/_contribution_summaries'

METRIC = 'dR2'          # 'dR2' (unique, honest) or 'tracevar' (gross in-sample swing)


def load_summary():
    """Prefer the combined CSV; fall back to concatenating per-array CSVs."""
    combined = os.path.join(SUMMARY_DIR, 'all_arrays_contributions.csv')
    if os.path.exists(combined):
        return pd.read_csv(combined)
    parts = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(SUMMARY_DIR, 'array*_contributions.csv')))]
    if not parts:
        raise SystemExit(f'No summary CSVs found in {SUMMARY_DIR} (run RunByArray.py first)')
    return pd.concat(parts, ignore_index=True)


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
    axh.set_xticklabels(fams, rotation=90, fontsize=7)
    axh.set_yticks(range(len(chans)))
    axh.set_yticklabels(chans, fontsize=7)
    axh.set_ylabel('channel')
    axh.set_title(f'Array {arr_idx}: unique {metric} per channel')
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
    for r, a in enumerate(arrs):
        sub = df[df['array'] == a]
        for cci, fam in enumerate(fams):
            col = f'{metric}::{fam}'
            if col in df.columns:
                M[r, cci] = sub[col].mean()
    vlim = sym_vlim(M)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(fams) + 2), 0.5 * len(arrs) + 2))
    im = ax.imshow(M, aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim))
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=90, fontsize=7)
    ax.set_yticks(range(len(arrs))); ax.set_yticklabels([f'array {a}' for a in arrs])
    ax.set_title(f'Mean unique {metric} per family, by array')
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
    cols = [c for c in df.columns if c.startswith(f'{METRIC}::')]
    if not cols:
        raise SystemExit(f'No {METRIC}:: columns in the summary CSV')
    fams = family_order(df, cols)

    # shared colour scale across arrays so panels are comparable (robust to outliers)
    vlim = sym_vlim(df[cols].to_numpy())

    print(f'Loaded {len(df)} channels across arrays {sorted(df["array"].unique())}')
    for a in sorted(df['array'].unique()):
        plot_array(int(a), df[df['array'] == a], fams, METRIC, vlim)
    plot_overview(df, fams, METRIC)
    print(f'Done -> {plots_dir}')
