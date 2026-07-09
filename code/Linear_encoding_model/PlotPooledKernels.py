"""
=============================================================================================
Average regressor kernel across channels for a POOLED (multi-session) fit (step 6c).
=============================================================================================
For a pooled fit (one ridge fit per electrode over the concatenated sessions, written under a
`pooled_<...>/` folder by ArrayRun.py + AssemblePooled.py), this reads every channel's
contributions npz and, per regressor FAMILY, averages that family's per-lag KERNEL (its
betas) ACROSS channels -- the "average brain response shape" for each event in the pooled fit.

One figure is written: `<session>_mean_kernels.pdf` -- a grid with one panel per family, each
showing the mean kernel (thick line) +/- SEM (band) across channels, with the individual
per-channel kernels drawn faintly behind it. The x-axis is seconds relative to the event
(from `lag_pre_frames` + `frame_rate` in the design metadata); event onset is the dashed line.

Why mean +/- SEM and faint individuals: LFP polarity can differ across electrodes, so a raw
mean can partly cancel. The band + faint traces make it obvious whether the mean reflects a
consistent kernel or an average over disagreeing channels.

PERMUTATION TEST ON THE MEAN KERNEL (PERM_TEST, step-6c significance)
--------------------------------------------------------------------
RegressorContributions.py (step 6a) runs, per channel, N_PERM circular-shift permutations of
the target with a DETERMINISTIC, channel-shared shift schedule, and (when SAVE_COL_NULL=True)
stores the full per-lag SIGNED null betas `col_null` (n_columns x N_PERM). Because every channel
used the SAME shift for permutation p, we can build a null for the ACROSS-CHANNEL MEAN kernel:

  * real statistic per lag:   m_l   = mean_c beta_{c,l}                    (the plotted mean kernel)
  * null per lag, per perm:   m_l^p = mean_c col_null_{c,l,p}             (same channels, shift p)
  * family max-stat null:     M^p   = max_l |m_l^p|                       (FWER across the lags)
  * threshold:                theta = (1-ALPHA) percentile of {M^p}
  * a lag is significant if |m_l| > theta; family p = (1 + #{M^p >= max_l|m_l|}) / (N_PERM + 1).

Significant lags of the mean kernel are shaded; the family p-value is printed in each panel title.
This needs npz files written by a RegressorContributions.py run with SAVE_COL_NULL=True -- older
files (no `col_null`) fall back to the mean +/- SEM plot with a warning.

Options (edit below):
  * PERM_TEST -- compute + shade the pooled per-lag permutation significance (default True).
  * ONLY_SIG  -- average each family over ONLY the channels where it is significant per channel
                 (family_p < ALPHA). Gives the kernel of the channels that actually encode the
                 event. NB: selecting channels by their own significance and then re-testing the
                 mean is circular, so ONLY_SIG is ignored for the PERM_TEST (a warning is printed);
                 use ONLY_SIG only with PERM_TEST=False. Default False.
  * NORMALISE -- divide each channel's kernel (and its null block) by that channel's peak |beta|
                 before averaging, to compare SHAPE when amplitudes vary. Default False.

Light / login-node script (no acme):
    python AnalyzePooledKernels.py                       # default SESSION below
    python AnalyzePooledKernels.py pooled_all            # a specific pooled fit
"""

import os
import sys
import re
import glob
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# make glm_config (single source of truth for the output tree + sampling rate) importable
for _d in (os.path.dirname(os.path.abspath(__file__)),
           os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    if os.path.exists(os.path.join(_d, 'glm_config.py')):
        sys.path.insert(0, _d)
        break
from glm_config import RESULTS_DIR, PLOTS_DIR
results_dir = RESULTS_DIR
plots_dir = PLOTS_DIR

# -------------------------
# Config
# -------------------------
SESSION = 'pooled_all'      # default; override on the command line (a 'pooled_...' folder, but a
                            # single date works too -- it just averages that session's channels)
if len(sys.argv) > 1:
    SESSION = sys.argv[1]

PERM_TEST = True            # pool per-lag nulls across channels -> significance of the MEAN kernel
ONLY_SIG = False            # True -> average each family only over channels with family_p < ALPHA
NORMALISE = False           # True -> peak-normalise each channel's kernel (+ its null) before averaging
ALPHA = 0.05               # significance level for the circular-shift permutation test (family_p)
SHOW_INDIVIDUAL = True      # draw the per-channel kernels faintly behind the mean


def discover_channels(session):
    """Channel numbers with a channel<ch>_regressors folder under this (pooled) session."""
    chans = []
    for p in glob.glob(str(Path(results_dir) / session / 'channel*_regressors')):
        m = re.search(r'channel(\d+)_regressors$', os.path.basename(p))
        if m:
            chans.append(int(m.group(1)))
    return sorted(chans)


def contrib_path(session, channel):
    designMatID = f'{session}_channel{channel}'
    return Path(results_dir) / session / f'channel{channel}_regressors' / 'results' / \
        f'{designMatID}_contributions.npz'


def sig_for(label, pmap):
    """family_p for a raw label; state levels fall back to the merged 'state' p-value."""
    if label in pmap:
        return pmap[label]
    if label.startswith('state_'):
        return pmap.get('state', np.nan)
    return np.nan


def load_axis(session, channel):
    """(frame_rate, pre_by_label) so a kernel column c maps to (c - pre)/frame_rate seconds from
    the event. Falls back to (None, {}) -> plot against column index if the design predates it."""
    designMatID = f'{session}_channel{channel}'
    meta_path = Path(results_dir) / session / f'channel{channel}_regressors' / 'results' / \
        f'{designMatID}_dMatProcessed_metadata.npz'
    if not meta_path.exists():
        return None, {}
    dmeta = np.load(meta_path, allow_pickle=True)
    if 'lag_pre_frames' in dmeta.files:
        return int(dmeta['frame_rate']), dict(zip([str(x) for x in dmeta['regLabels'].ravel()],
                                                  dmeta['lag_pre_frames'].ravel()))
    return None, {}


def main():
    channels = discover_channels(SESSION)
    if not channels:
        raise SystemExit(f'No channel*_regressors folders under {results_dir}/{SESSION} '
                         f'(is {SESSION} a fitted pooled session?)')

    do_perm = PERM_TEST
    if PERM_TEST and ONLY_SIG:
        print('  [warn] ONLY_SIG selects channels by their own significance; re-testing that mean '
              'is circular, so ONLY_SIG is IGNORED while PERM_TEST is on.')
        only_sig = False
    else:
        only_sig = ONLY_SIG

    # ---- pass 1: small arrays (real kernels, per-channel sig, shift schedule) ----
    # chan_kernels[ch] = {label: kernel}; chan_sig[ch] = {label: family_p}; chan_shifts[ch] = array
    chan_kernels, chan_sig, chan_shifts, chan_hascn = {}, {}, {}, {}
    label_order = []
    ref_shifts = None
    for ch in channels:
        cp = contrib_path(SESSION, ch)
        if not cp.exists():
            continue
        c = np.load(cp, allow_pickle=True)
        betas = c['betas'].ravel()
        regIdx = c['regIdx'].ravel()
        regLabels = c['regLabels'].ravel()
        kern = {}
        for g in np.unique(regIdx):
            lab = str(regLabels[int(g) - 1])
            kern[lab] = betas[regIdx == g]
            if lab not in label_order:
                label_order.append(lab)
        chan_kernels[ch] = kern
        pmap = {}
        if 'family_p' in c.files and 'group_labels' in c.files:
            pmap = {str(l): float(p) for l, p in zip(c['group_labels'], c['family_p'])}
        chan_sig[ch] = pmap
        chan_shifts[ch] = c['perm_shifts'] if 'perm_shifts' in c.files else None
        has_cn = 'col_null' in c.files
        chan_hascn[ch] = has_cn
        if do_perm and has_cn and ref_shifts is None and chan_shifts[ch] is not None:
            ref_shifts = np.asarray(chan_shifts[ch])

    if not chan_kernels:
        raise SystemExit(f'Found {len(channels)} channel folders but none had a contributions.npz '
                         f'(run RegressorContributions.py for {SESSION} first).')

    if do_perm and ref_shifts is None:
        print('  [warn] no channel has `col_null` (rerun RegressorContributions.py with '
              'SAVE_COL_NULL=True). Falling back to mean +/- SEM without a permutation test.')
        do_perm = False

    frame_rate, pre_by_label = load_axis(SESSION, sorted(chan_kernels)[0])

    # ---- decide, per family, the included channels (consistent for the plotted mean AND the test) ----
    # modal lag length per label; a channel is included if its lag count matches, it passes ONLY_SIG,
    # and (for the test) it carries a col_null with the reference shift schedule.
    include = defaultdict(list)      # label -> [channels used for BOTH mean and null]
    modal_len = {}
    for lab in label_order:
        lens = [chan_kernels[ch][lab].size for ch in chan_kernels if lab in chan_kernels[ch]]
        if not lens:
            continue
        modal_len[lab] = Counter(lens).most_common(1)[0][0]
        for ch in sorted(chan_kernels):
            if lab not in chan_kernels[ch] or chan_kernels[ch][lab].size != modal_len[lab]:
                continue
            if only_sig and not (np.isfinite(sig_for(lab, chan_sig[ch])) and
                                 sig_for(lab, chan_sig[ch]) < ALPHA):
                continue
            if do_perm:
                ps = chan_shifts[ch]
                if not (chan_hascn[ch] and ps is not None and np.array_equal(np.asarray(ps), ref_shifts)):
                    continue                              # can't pool this channel's null -> drop it
            include[lab].append(ch)

    # ---- pass 2: accumulate the pooled null (mean over channels of the per-lag null, per perm) ----
    null_sum = {}      # label -> (modal_len x N_PERM) running sum over included channels
    n_perm = None
    if do_perm:
        for ch in sorted(chan_kernels):
            if not any(ch in include[lab] for lab in include):
                continue
            c = np.load(contrib_path(SESSION, ch), allow_pickle=True)
            cn = np.asarray(c['col_null'])                # (n_columns x N_PERM), signed
            if cn.ndim != 2 or cn.shape[1] == 0:
                continue
            n_perm = cn.shape[1] if n_perm is None else n_perm
            regIdx = c['regIdx'].ravel()
            regLabels = c['regLabels'].ravel()
            lab_by_g = {int(g): str(regLabels[int(g) - 1]) for g in np.unique(regIdx)}
            for g, lab in lab_by_g.items():
                if ch not in include.get(lab, []):
                    continue
                block = cn[regIdx == g].astype(np.float64)   # (n_lags x N_PERM)
                if NORMALISE:
                    peak = np.max(np.abs(chan_kernels[ch][lab])) or 1.0
                    block = block / peak
                if lab not in null_sum:
                    null_sum[lab] = np.zeros((modal_len[lab], cn.shape[1]))
                null_sum[lab] += block

    # ---- plot ----
    labels = [lab for lab in label_order if include.get(lab)]
    n = len(labels)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.7 * nrow), squeeze=False)

    n_sig_families = 0
    for j, lab in enumerate(labels):
        ax = axes[j // ncol][j % ncol]
        chs = include[lab]
        stack = np.vstack([
            (chan_kernels[ch][lab] / (np.max(np.abs(chan_kernels[ch][lab])) or 1.0)) if NORMALISE
            else chan_kernels[ch][lab]
            for ch in chs])
        n_ch = stack.shape[0]
        mean = stack.mean(axis=0)
        sem = stack.std(axis=0, ddof=1) / np.sqrt(n_ch) if n_ch > 1 else np.zeros_like(mean)

        if frame_rate:
            pre = int(pre_by_label.get(lab, 0))
            x = (np.arange(modal_len[lab]) - pre) / frame_rate
            ax.axvline(0, color='red', lw=0.6, ls='--')
            ax.set_xlabel('time from event (s)')
        else:
            x = np.arange(modal_len[lab])
            ax.set_xlabel('lag (column)')

        # pooled per-lag permutation significance of the MEAN kernel
        title_sig = ''
        if do_perm and lab in null_sum:
            mean_null = null_sum[lab] / n_ch                 # (n_lags x N_PERM) mean over channels
            Mnull = np.max(np.abs(mean_null), axis=0)        # (N_PERM,) max over lags -> FWER
            theta = float(np.percentile(Mnull, 100 * (1 - ALPHA)))
            real_stat = float(np.max(np.abs(mean)))
            fam_p = (1 + int(np.sum(Mnull >= real_stat))) / (Mnull.size + 1)   # (b+1)/(N+1)
            sig = np.abs(mean) > theta
            if sig.any():
                ylo, yhi = min(mean.min(), (mean - sem).min()), max(mean.max(), (mean + sem).max())
                pad = 0.05 * (yhi - ylo + 1e-12)
                ax.set_ylim(ylo - pad, yhi + pad)
                ax.fill_between(x, ylo - pad, yhi + pad, where=sig, color='#8dd3c7',
                                alpha=0.4, zorder=0)
            title_sig = f'  p={fam_p:.3g}' + ('*' if fam_p < ALPHA else '')
            n_sig_families += int(fam_p < ALPHA)

        if SHOW_INDIVIDUAL:
            for row in stack:
                ax.plot(x, row, color='grey', lw=0.4, alpha=0.25, zorder=1)
        ax.fill_between(x, mean - sem, mean + sem, color='tab:blue', alpha=0.3, zorder=2)
        ax.plot(x, mean, color='tab:blue', lw=1.8, zorder=3)
        ax.axhline(0, color='grey', lw=0.6)
        ax.set_title(f'{lab}  (n={n_ch}){title_sig}', fontsize=9)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')

    tag = 'significant channels only' if only_sig else 'all channels'
    norm = ', peak-normalised' if NORMALISE else ''
    test = (f'shaded = mean-kernel sig (pooled perm test, {n_perm} shifts, FWER/lag)'
            if do_perm else 'no permutation test')
    fig.suptitle(f'{SESSION}: mean regressor kernels across {tag} ({len(chan_kernels)} channels'
                 f'{norm})  --  band = +/- SEM, faint = per channel; {test}')
    fig.tight_layout()

    out_dir = Path(plots_dir) / SESSION / '_contribution_summaries'
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ('_sig' if only_sig else '') + ('_permtest' if do_perm else '')
    out = out_dir / f'{SESSION}_mean_kernels{suffix}.pdf'
    fig.savefig(out)
    plt.close(fig)
    print(f'Loaded {len(chan_kernels)} channels; {n} families plotted.')
    if do_perm:
        print(f'Pooled permutation test: {n_sig_families}/{n} families significant at p<{ALPHA:g}.')
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
