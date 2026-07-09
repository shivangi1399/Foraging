"""
=============================================================================================
Visualise the per-array / per-channel GLM contribution summaries.
=============================================================================================
Reads the CSVs written by ArrayRun.py (under <results_dir>/_contribution_summaries/) and makes,
for each array, a channel x regressor-family heatmap of unique dR2 -- so you can see at a glance
which families encode on which channels, and how consistent that is across the array. Each array
figure also carries a per-channel full-model R2 bar. A final overview figure shows the mean dR2 per
family for every array together.

dR2 colour: red = positive unique contribution (family helps prediction), blue = negative (hurts /
below chance). Significance is the circular-shift permutation test (pval:: columns); the shuffled
`dummy` control is excluded from these figures (it was only ever a dR2 null-floor check, and the
permutation test -- whose null IS the shuffled target -- does not depend on it).

Works for a single session OR a POOLED (multi-session) fit: the pooled run writes the same summary
CSVs under a `pooled_<first>_<last>/_contribution_summaries/` folder, so just point SESSION at it.

Light / login-node script (no acme):
    python PlotArraySummary.py                       # default SESSION below
    python PlotArraySummary.py 20230214              # one session
    python PlotArraySummary.py pooled_20230203_20230214   # the concatenated multi-session fit
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
SESSION = 'pooled_all'    # default; override on the command line (a real date OR a 'pooled_...' folder)
if len(sys.argv) > 1:
    SESSION = sys.argv[1]
POOLED = SESSION.startswith('pooled_')   # a concatenated multi-session fit (from AssemblePooled.py)
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
        raise SystemExit(f'No summary CSVs found in {SUMMARY_DIR} (run ArrayRun.py first)')
    return pd.concat(parts, ignore_index=True)


def display_labels(fams):
    """Tick labels for display. `state` is the merged per-trial state levels (state_0/state_2/...
    scored together), so flag it as grouped. In a POOLED fit `session` is the per-session intercept
    (nuisance) -- annotate it so the pooled plot is unambiguous."""
    out = []
    for f in fams:
        if f == 'state':
            out.append(f + ' (grouped)')
        elif POOLED and f == 'session':
            out.append(f + ' (offsets)')
        else:
            out.append(f)
    return out


def sym_vlim(values):
    """Robust symmetric colour limit: 98th pct of |finite values|, always > 0."""
    v = np.abs(np.asarray(values, dtype=float).ravel())
    v = v[np.isfinite(v)]
    return max(float(np.percentile(v, 98)), 1e-9) if v.size else 1e-9


def family_order(df, cols):
    """Consistent family ordering across arrays: by global mean metric (strongest first). The
    shuffled `dummy` control is dropped from every figure -- it was only a dR2 null-floor yardstick,
    never a result, and significance comes from the circular-shift permutation test (which already
    uses the shuffled target as its null, so it does not rely on the dummy regressor)."""
    means = df[cols].mean().sort_values(ascending=False)
    fams = [c.split('::')[1] for c in means.index]
    fams = [f for f in fams if f != 'dummy']   # exclude the shuffled control from all figures
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


def _load_channel_nulls(channel):
    """(family_labels, fam_maxnull[F x P], real_max[F]) for one channel, or None if the null
    distributions aren't in the contributions npz (i.e. it predates the null-saving RegressorContributions)."""
    f = os.path.join(results_dir, SESSION, f'channel{channel}_regressors', 'results',
                     f'{SESSION}_channel{channel}_contributions.npz')
    if not os.path.exists(f):
        return None
    c = np.load(f, allow_pickle=True)
    if 'fam_maxnull' not in c.files or 'real_max' not in c.files:
        return None
    fm = np.asarray(c['fam_maxnull'], dtype=float)
    if fm.ndim != 2 or fm.shape[1] == 0:
        return None
    labs = [str(x) for x in c['group_labels']]
    return labs, fm, np.asarray(c['real_max'], dtype=float)


def array_perm_significance(df):
    """Array-level shared-shift permutation test -> {(array, family): bool}.
    For each array & family: take the MEAN across the array's channels of the real max|beta| as the
    array statistic, then build its null by taking -- for each shuffle index -- the MEAN across
    channels of the per-permutation nulls (pooled by shuffle index; valid because every channel used
    the SAME shift, so the shuffle preserves cross-channel correlation). Significant if the real mean
    exceeds the (1-ALPHA) percentile of that array null. Because the statistic is |beta|, that
    percentile is already the two-sided ALPHA cutoff."""
    out = {}
    for a in sorted(df['array'].unique()):
        sub = df[df['array'] == a]
        loaded = [x for x in (_load_channel_nulls(int(ch)) for ch in sub['channel']) if x is not None]
        if not loaded:
            continue
        nperm = loaded[0][1].shape[1]
        if any(fm.shape[1] != nperm for _, fm, _ in loaded):
            print(f'  array {a}: channels disagree on N_PERM -- skipping array-level test')
            continue
        common = set(loaded[0][0]).intersection(*[set(labs) for labs, _, _ in loaded[1:]])
        for fam in common:
            reals = np.array([rm[labs.index(fam)] for labs, _, rm in loaded])           # (n_chan,)
            nulls = np.vstack([fm[labs.index(fam)] for labs, fm, _ in loaded])          # (n_chan, nperm)
            real_stat = float(np.mean(reals))
            null_dist = np.mean(nulls, axis=0)                                          # (nperm,)
            out[(a, fam)] = bool(real_stat > np.percentile(null_dist, 100 * (1 - ALPHA)))
    return out


def plot_overview(df, fams, metric, arr_sig=None):
    """arrays x families heatmap of the MEAN metric across each array's channels. The dot marks a
    family significant at the array level by the shared-shift permutation test (arr_sig)."""
    arrs = sorted(df['array'].unique())
    M = np.full((len(arrs), len(fams)), np.nan)
    for r, a in enumerate(arrs):
        sub = df[df['array'] == a]
        for cci, fam in enumerate(fams):
            col = f'{metric}::{fam}'
            if col in sub.columns:
                vals = sub[col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    M[r, cci] = float(np.mean(vals))
    vlim = sym_vlim(M)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(fams) + 2), 0.5 * len(arrs) + 2))
    im = ax.imshow(M, aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim))
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(display_labels(fams), rotation=90, fontsize=7)
    ax.set_yticks(range(len(arrs))); ax.set_yticklabels([f'array {a}' for a in arrs])
    ax.set_title(f'mean {metric} per family, by array  (dot = array-level perm test p<{ALPHA:g})')
    if arr_sig is not None:
        ys = [r for r, a in enumerate(arrs) for cci, fam in enumerate(fams) if arr_sig.get((a, fam))]
        xs = [cci for r, a in enumerate(arrs) for cci, fam in enumerate(fams) if arr_sig.get((a, fam))]
        if xs:
            ax.scatter(xs, ys, marker='o', s=10, color='k', edgecolors='white',
                       linewidths=0.4, zorder=3)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=f'mean {metric}')
    fig.tight_layout()
    out = os.path.join(plots_dir, f'overview_mean_{metric}_by_array.pdf')
    fig.savefig(out); plt.close(fig)
    print(f'  saved {out}')


if __name__ == '__main__':
    os.makedirs(plots_dir, exist_ok=True)
    df = load_summary()
    print(f'Loaded {len(df)} channels across arrays {sorted(df["array"].unique())}')

    # array-level significance for the overview dots (computed once; metric-independent)
    arr_sig = array_perm_significance(df)
    if not arr_sig:
        print('  (no fam_maxnull in the contributions npz -- re-run RegressorContributions to get '
              'array-level perm dots; overview will have no dots)')

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
        plot_overview(df, fams, metric, arr_sig)
    print(f'Done -> {plots_dir}')
