"""
=============================================================================================
Plot per-regressor contributions (step 6b).
=============================================================================================
Pipeline position: step 6b (see README.md). For every session/channel, reads the contributions
npz that RegressorContributions.py (step 6a) wrote and, per channel, saves three PDFs into the
plots tree (mirroring the results tree but rooted in plots_dir, like AnalyzeRedundancySubsample.py):

  * _kernels.pdf       - each regressor family's per-lag KERNEL (its weights vs column index):
                         the brain's fitted response shape for that event.
  * _traces.pdf        - each family's per-timepoint CONTRIBUTION over a short window (its columns
                         @ its weights), with the observed LFP and full prediction overlaid:
                         "at which time points does each family drive the signal".
  * _summary.pdf       - bar charts of unique dR2 and contribution-trace variance per family:
                         "how much each family contributes overall".

The contribution traces are recomputed here from betas + the processed design (step 6a does not
store the full n_timepoints x n_families arrays).

Light / login-node script (no acme): `python AnalyzeRegressorContributions.py`.
"""

import os
import re
import glob
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import diags, load_npz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

results_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/states_analysis/states_lfp/all_trials/full_length/GLM'
plots_dir = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/states_lfp/all_trials/full_length/GLM'

# Which session/channel pairs to plot. None -> auto-discover from the results tree.
SESSIONS = None
CHANNELS = None

# Window (in frames) for the contribution-trace plot (match ExamineFit.py defaults).
START_TIME = 10000
WINDOW = 1500


def discover_channels(session):
    chans = []
    for p in glob.glob(str(Path(results_dir) / session / 'channel*_regressors')):
        m = re.search(r'channel(\d+)_regressors$', os.path.basename(p))
        if m:
            chans.append(int(m.group(1)))
    return sorted(chans)


def discover_sessions():
    out = []
    for p in sorted(glob.glob(str(Path(results_dir) / '*'))):
        if os.path.isdir(p) and discover_channels(os.path.basename(p)):
            out.append(os.path.basename(p))
    return out


def normalise_design(fullR_sparse):
    """Column-normalise the design exactly as FittingGLM.py / RegressorContributions.py do."""
    design_mean = np.array(fullR_sparse.mean(axis=0)).ravel()
    design_sq_mean = np.array(fullR_sparse.power(2).mean(axis=0)).ravel()
    design_std = np.sqrt(design_sq_mean - design_mean**2)
    design_std[design_std == 0] = 1.0
    return fullR_sparse @ diags(1 / design_std)


def analyze_channel(session, channel):
    designMatID = f'{session}_channel{channel}'
    SESSION_ROOT = Path(results_dir) / session / f'channel{channel}_regressors'
    SAVE_PATH = SESSION_ROOT / "results"

    contrib_path = SAVE_PATH / f'{designMatID}_contributions.npz'
    if not contrib_path.exists():
        print(f'skip {designMatID}: no {contrib_path.name} (run RegressorContributions.py first)')
        return

    c = np.load(contrib_path, allow_pickle=True)
    betas = c['betas']
    regIdx = c['regIdx'].ravel()
    full_R2 = float(c['full_R2'])
    n_timepoints = int(c['n_timepoints'])
    group_ids = c['group_ids']
    group_labels = c['group_labels']
    dR2 = c['dR2']
    trace_var = c['trace_var']

    print(f'\n===== {designMatID} ({n_timepoints} samples, full cv-R2 {full_R2:.4f}) =====')

    fig_dir = Path(plots_dir) / session / f'channel{channel}_regressors' / 'results'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # per-family pre-window (frames) + frame rate saved by DesignMatrix, so the kernel x-axis can be in
    # SECONDS relative to the event (lag column c -> (c - pre_frames)/frame_rate; event at 0).
    dmeta = np.load(SAVE_PATH / f'{designMatID}_dMatProcessed_metadata.npz', allow_pickle=True)
    if 'lag_pre_frames' in dmeta.files:
        frame_rate = int(dmeta['frame_rate'])
        pre_by_label = dict(zip([str(x) for x in dmeta['regLabels'].ravel()],
                                dmeta['lag_pre_frames'].ravel()))
    else:                      # design predates the seconds axis -> fall back to column index
        frame_rate, pre_by_label = None, {}

    # ---- 1. Kernel shapes: each family's weights vs time from its event (event at t=0) ----
    n = group_ids.size
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.5 * nrow), squeeze=False)
    for j, g in enumerate(group_ids):
        ax = axes[j // ncol][j % ncol]
        kernel = betas[regIdx == g]
        label = str(group_labels[j])
        if frame_rate:
            pre = int(pre_by_label.get(label, 0))
            x = (np.arange(kernel.size) - pre) / frame_rate     # seconds relative to the event
            ax.plot(x, kernel, marker='.')
            ax.axvline(0, color='red', lw=0.6, ls='--')          # event onset
            ax.set_xlabel('time from event (s)')
        else:
            ax.plot(kernel, marker='.')
            ax.set_xlabel('lag (column)')
        ax.axhline(0, color='grey', lw=0.6)
        ax.set_title(label, fontsize=9)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle(f'{designMatID}: regressor kernels (x = seconds from event, dashed = onset)')
    fig.tight_layout()
    fig.savefig(fig_dir / f'{designMatID}_kernels.pdf')
    plt.close(fig)

    # ---- 2. Contribution traces over a window (recomputed from betas + design) ----
    fullR_sparse = load_npz(SAVE_PATH / f'{designMatID}_dMatProcessed_sparse.npz')[:n_timepoints]
    design_norm = normalise_design(fullR_sparse)
    # downsampled target (design's row rate), matching FittingGLM.py / RegressorContributions.py;
    # the raw neural_data.npz is at NATIVE_FS and would be misaligned with the design/preds.
    neural_data = np.load(SAVE_PATH / f'{designMatID}_neural_downsampled.npz')['data'][:n_timepoints].ravel()

    start, end = START_TIME, START_TIME + WINDOW
    full_pred = np.asarray(design_norm[start:end] @ betas).ravel()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(neural_data[start:end], color='k', lw=1.5, label='Observed LFP')
    ax.plot(full_pred, color='tab:red', lw=1.5, alpha=0.8, label='Full prediction')
    for g, lab in zip(group_ids, group_labels):
        cols_g = (regIdx == g)
        contrib = np.asarray(design_norm[start:end, cols_g] @ betas[cols_g]).ravel()
        ax.plot(contrib, lw=0.9, alpha=0.7, label=str(lab))
    ax.set_xlabel('Frame (window start = %d)' % START_TIME)
    ax.set_ylabel('LFP amplitude')
    ax.set_title(f'{designMatID}: per-family contributions (cv-R2={full_R2:.4f})')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    fig.tight_layout()
    fig.savefig(fig_dir / f'{designMatID}_traces.pdf')
    plt.close(fig)

    # ---- 3. Summary bars: unique dR2 and contribution-trace variance per family ----
    order = np.argsort(dR2)[::-1]
    labels_o = [str(group_labels[i]) for i in order]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, max(4, 0.35 * n)))
    ypos = np.arange(n)
    ax1.barh(ypos, dR2[order], color='tab:blue')
    ax1.set_yticks(ypos); ax1.set_yticklabels(labels_o, fontsize=8)
    ax1.invert_yaxis(); ax1.axvline(0, color='grey', lw=0.6)
    ax1.set_xlabel('unique dR2 (full - reduced)')
    ax1.set_title('Unique contribution')
    ax2.barh(ypos, trace_var[order], color='tab:green')
    ax2.set_yticks(ypos); ax2.set_yticklabels(labels_o, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('contribution-trace variance')
    ax2.set_title('Trace variance')
    fig.suptitle(f'{designMatID}: per-family contribution summary')
    fig.tight_layout()
    fig.savefig(fig_dir / f'{designMatID}_summary.pdf')
    plt.close(fig)

    print(f'  saved kernels / traces / summary -> {fig_dir}')


if __name__ == '__main__':
    sessions = SESSIONS if SESSIONS is not None else discover_sessions()
    if not sessions:
        raise SystemExit(f'No sessions with channel*_regressors found under {results_dir}')
    for session in sessions:
        channels = CHANNELS if CHANNELS is not None else discover_channels(session)
        for channel in channels:
            analyze_channel(session, channel)
