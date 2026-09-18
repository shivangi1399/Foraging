#!/usr/bin/env python
"""
=============================================================================================
export_figure_data.py -- the saved encoding-model results that figures.py draws, as .mat
=============================================================================================
Reads results the encoding-model pipeline already saved and writes them, or plain averages of
them, to writing/figures/data/. It reruns no analysis and no statistical test.

  fig07  kernels  Results/Linear_encoding_model/all_regressors/100Hz/pooled_all/
                  channel*_regressors/results/: saved kernels (betas), averaged across channels
                  (peak-normalized, as PlotPooledKernels.py)          -> fig07_encoding_model.mat
  fig08  dR2      .../pooled_all/_contribution_summaries/all_arrays_contributions.csv (per-channel
                  dR2 and p), averaged per array; plots/.../region_dR2_stats.csv (saved q values)
                                                                      -> fig08_contributions.mat
The step names are those of the files; figures.py draws them as Figures 7 and 8.

Every step only reads small files, so the login node is fine:
    /gs/home/patels/.conda/envs/warping/bin/python export_figure_data.py          # both
    /gs/home/patels/.conda/envs/warping/bin/python export_figure_data.py fig08    # one
"""

import os
import sys
import re
import glob
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

ROOT = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi'
OUT_DIR = os.path.join(ROOT, 'writing', 'figures', 'data')
GLM_DIR = os.path.join(ROOT, 'Results', 'Linear_encoding_model', 'all_regressors', '100Hz')
GLM_PLOTS = os.path.join(ROOT, 'plots', 'Linear_encoding_model', 'all_regressors', '100Hz')
POOLED = 'pooled_all'
REGION_NAMES = ['V1-periphery', 'V1-fovea', 'V4']


def region_of_array(a):
    """1 = V1-periphery (arrays 1-3), 2 = V1-fovea (array 4), 3 = V4 (arrays 5-6)."""
    a = np.asarray(a)
    return np.where(a <= 3, 1, np.where(a == 4, 2, 3))


def bh_fdr(p):
    """Benjamini-Hochberg q-values for a 1-D array of p-values; NaNs stay NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    n = int(ok.sum())
    if not n:
        return q
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]      # enforce monotonicity
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = out
    return q


def cells(seq):
    """A list of strings or of arrays -> a MATLAB cell array."""
    out = np.empty(len(seq), dtype=object)
    for i, s in enumerate(seq):
        out[i] = s
    return out


def save(name, d):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + '.mat')
    sio.savemat(path, d, do_compression=True, long_field_names=True)
    print(f'  wrote {path}')


# ============================================================================================
# fig07 -- encoding model: event raster, example fit, mean kernels
# ============================================================================================
RASTER_FAMS = ['stim_onset', 'reaction_onset', 'reward_onset', 'saccade_onset', 'saccade_offset',
               'target_in_RF_onset', 'distractor_in_RF_onset', 'grass_in_RF_onset']
START_SEC, WINDOW_SEC = 100.0, 15.0     # the window ExamineFit.py plots
NORMALISE = True                         # as PlotPooledKernels.py


def _results(ch):
    return os.path.join(GLM_DIR, POOLED, f'channel{ch}_regressors', 'results')


def fig07():
    csv = pd.read_csv(os.path.join(GLM_DIR, POOLED, '_contribution_summaries', 'all_arrays_contributions.csv'))
    arr_of = dict(zip(csv['channel'].astype(int), csv['array'].astype(int)))
    channels = sorted(int(re.search(r'channel(\d+)_regressors$', p).group(1))
                      for p in glob.glob(os.path.join(GLM_DIR, POOLED, 'channel*_regressors')))

    # ---- saved kernels, averaged across channels ----
    kern, label_order = {}, []
    for ch in channels:
        cp = os.path.join(_results(ch), f'{POOLED}_channel{ch}_contributions.npz')
        if not os.path.exists(cp):
            continue
        c = np.load(cp, allow_pickle=True)
        betas, regIdx, regLabels = c['betas'].ravel(), c['regIdx'].ravel(), c['regLabels'].ravel()
        kern[ch] = {}
        for g in np.unique(regIdx):
            lab = str(regLabels[int(g) - 1])
            kern[ch][lab] = betas[regIdx == g]
            if lab not in label_order:
                label_order.append(lab)

    first = sorted(kern)[0]
    meta = np.load(os.path.join(_results(first), f'{POOLED}_channel{first}_dMatProcessed_metadata.npz'),
                   allow_pickle=True)
    frame_rate = int(meta['frame_rate'])
    pre_by_label = dict(zip([str(v) for v in meta['regLabels'].ravel()], meta['lag_pre_frames'].ravel()))

    K = defaultdict(list)
    print('  mean kernels (peak = lag of the largest |mean|):')
    for lab in label_order:
        lens = [kern[ch][lab].size for ch in kern if lab in kern[ch]]
        modal = Counter(lens).most_common(1)[0][0]
        chs = [ch for ch in sorted(kern) if lab in kern[ch] and kern[ch][lab].size == modal]
        stack = np.vstack([kern[ch][lab] / ((np.max(np.abs(kern[ch][lab])) or 1.0) if NORMALISE else 1.0)
                           for ch in chs])
        mean = stack.mean(axis=0)
        sem = stack.std(axis=0, ddof=1) / np.sqrt(len(chs)) if len(chs) > 1 else np.zeros_like(mean)
        x = (np.arange(modal) - int(pre_by_label.get(lab, 0))) / frame_rate
        reg = region_of_array([arr_of.get(ch, 0) for ch in chs])
        by_region = np.vstack([stack[reg == r].mean(axis=0) if np.any(reg == r) else np.full(mean.shape, np.nan)
                               for r in (1, 2, 3)])
        for key, val in [('k_labels', lab), ('k_x', x), ('k_mean', mean), ('k_sem', sem),
                         ('k_nch', len(chs)), ('k_region_mean', by_region)]:
            K[key].append(val)
        peaks = [x[np.nanargmax(np.abs(by_region[r]))] if np.any(np.isfinite(by_region[r])) else np.nan
                 for r in range(3)]
        print(f'    {lab:24s} n={len(chs):2d} peak {x[np.argmax(np.abs(mean))]:+.2f} s | '
              + ' '.join(f'{REGION_NAMES[r]} {peaks[r]:+.2f}' for r in range(3)))

    # ---- saved fit of the channel whose full R2 is closest to the median ----
    ex_ch = int(csv.iloc[(csv['full_R2'] - csv['full_R2'].median()).abs().argmin()]['channel'])
    pk = sorted(glob.glob(os.path.join(_results(ex_ch), f'{POOLED}_channel{ex_ch}_*samples.pkl')),
                key=lambda p: int(re.search(r'_(\d+)samples\.pkl$', p).group(1)))[-1]
    with open(pk, 'rb') as f:
        res = pickle.load(f)
    preds = np.asarray(res['preds']).ravel()
    n_tp = preds.size
    z = np.load(os.path.join(_results(ex_ch), f'{POOLED}_channel{ex_ch}_neural_downsampled.npz'))
    obs = np.asarray(z['data']).ravel()[:n_tp]
    s0 = min(int(START_SEC * frame_rate), n_tp - 1)
    s1 = min(s0 + int(WINDOW_SEC * frame_rate), n_tp)

    # ---- event times in the same rows: the saved zero-lag column of each family ----
    X = sp.load_npz(os.path.join(_results(ex_ch), f'{POOLED}_channel{ex_ch}_dMatProcessed_sparse.npz')).tocsc()
    m = np.load(os.path.join(_results(ex_ch), f'{POOLED}_channel{ex_ch}_dMatProcessed_metadata.npz'), allow_pickle=True)
    regIdx = m['regIdx'].ravel()
    regLabels = [str(v) for v in m['regLabels'].ravel()]
    ras = []
    for lab in RASTER_FAMS:
        g = regLabels.index(lab) + 1
        col = np.where(regIdx == g)[0][int(m['lag_pre_frames'].ravel()[g - 1])]
        ras.append(np.nonzero(X[s0:s1, col].toarray().ravel())[0] / frame_rate)

    d = {k: (cells(v) if k in ('k_labels', 'k_x', 'k_mean', 'k_sem', 'k_region_mean') else np.asarray(v))
         for k, v in K.items()}
    d.update({'ex_channel': ex_ch, 'ex_array': arr_of.get(ex_ch, 0), 'ex_R2': float(res['scores']),
              'ex_t': np.arange(s1 - s0) / frame_rate, 'ex_obs': obs[s0:s1], 'ex_pred': preds[s0:s1],
              'ex_start_sec': START_SEC, 'ras_labels': cells(RASTER_FAMS), 'ras_times': cells(ras)})
    save('fig07_encoding_model', d)
    print(f'  example fit: channel {ex_ch} (array {arr_of.get(ex_ch)}), cross-validated R2 {float(res["scores"]):.4f}')


# ============================================================================================
# fig08 -- unique contributions: per channel, averaged per array, per region
# ============================================================================================
def fig08():
    df = pd.read_csv(os.path.join(GLM_DIR, POOLED, '_contribution_summaries', 'all_arrays_contributions.csv'))
    fams = [c.split('::')[1] for c in df.columns if c.startswith('dR2::')]
    fams = [f for f in fams if f not in ('session', 'dummy')]
    order = sorted(fams, key=lambda f: -df['dR2::' + f].mean()) + ['dummy']

    arr_mean = np.full((6, len(order)), np.nan)
    for a in range(1, 7):
        sub = df[df['array'] == a]
        for j, fam in enumerate(order):
            v = sub['dR2::' + fam].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                arr_mean[a - 1, j] = v.mean()

    st = pd.read_csv(os.path.join(GLM_PLOTS, POOLED, '_contribution_summaries', 'region_dR2_stats.csv')).set_index('family')

    def col(name):
        return np.array([st.loc[f, name] if f in st.index else np.nan for f in order], float)

    # Channel-level significance: every family is tested on every channel, so the per-channel
    # permutation p is corrected across the families of that channel with Benjamini-Hochberg,
    # as the array-level dots and the region contrasts already are. pval is kept beside qval
    # because the shuffled control is only a nominal-rate check when it is uncorrected.
    pval = df[['pval::' + f for f in order]].to_numpy(float)
    qval = np.vstack([bh_fdr(row) for row in pval])

    d = {'fam': cells(order),
         'dR2': df[['dR2::' + f for f in order]].to_numpy(float),
         'pval': pval, 'qval': qval,
         'full_R2': df['full_R2'].to_numpy(float),
         'array': df['array'].to_numpy(int), 'channel': df['channel'].to_numpy(int),
         'region': region_of_array(df['array'].to_numpy(int)),
         'arr_mean': arr_mean,
         'q_v4v1': col('q[V4 vs V1 (all)]'), 'q_rel_v4v1': col('q_rel[V4 vs V1 (all)]'),
         'q_fp': col('q[V1-fovea vs V1-periph]'), 'q_rel_fp': col('q_rel[V1-fovea vs V1-periph]')}
    save('fig08_contributions', d)
    for j, fam in enumerate(order):
        print(f'  {fam:24s} mean dR2 {d["dR2"][:, j].mean():+.2e}  '
              f'q<0.05 on {100 * np.mean(d["qval"][:, j] < 0.05):5.1f}% of channels '
              f'(uncorrected p<0.05 on {100 * np.mean(d["pval"][:, j] < 0.05):5.1f}%)')


STEPS = {'fig07': fig07, 'fig08': fig08}

if __name__ == '__main__':
    wanted = sys.argv[1:] or list(STEPS)
    for key in wanted:
        if key not in STEPS:
            raise SystemExit(f'unknown step {key}; choose from {", ".join(STEPS)}')
        print(f'\n===== {key} =====')
        STEPS[key]()
