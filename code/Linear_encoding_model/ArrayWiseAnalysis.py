"""
=============================================================================================
Test whether the GLM contributions show an ANATOMICAL gradient across the arrays.
=============================================================================================
PlotArraySummary.py shows the contributions array-by-array; this script goes one level up and
asks whether the arrays group by *cortical area*. Arrays are mapped to regions via the REGIONS
dict below (default: arrays 1-3 = peripheral V1, array 4 = foveal V1, arrays 5-6 = V4 -- edit it
if the probe layout changes), and for every regressor family we ask:

    1. Are one region's per-channel dR2 values systematically LARGER than another's? (Mann-Whitney
       per contrast + Kruskal-Wallis omnibus across all regions.) Note both are RANK tests: the
       null is that neither region's values tend to exceed the other's, NOT that the means are
       equal. The mean_* columns are reported for interpretation, but they are not what is tested,
       so describe a significant result as a shift in the distribution rather than in the mean.
       Every contrast also gets a rank-biserial correlation rb in [-1, +1]: the share of
       cross-region channel pairs favouring side A minus the share favouring side B. rb does not
       shrink as n grows, so it -- not p -- is the number to quote as the effect size.
    2. Does it differ once you divide out how well the model does on that channel at all?
       (dR2 / full_R2 -- the RELATIVE share, which matters because a region with a low full_R2
       is not necessarily a region where a given family is unimportant.) The same two rank tests
       are run on the relative values as well, giving the p_rel[...] / q_rel[...] columns.
    3. Is the effect really regional, or just a smooth gradient along the probe? Each family is
       also correlated with the raw array index (Spearman). A family that is genuinely regional
       shows a region difference WITHOUT a monotonic probe-position trend; a family that only
       shows the monotonic trend is more likely a depth/position artefact.
    4. Within the V1 arrays only, does the family in ECC_FAMILY vary continuously with channel
       number? Channels are numbered along the probe, so if eccentricity maps onto channel
       number the transition need not line up with the array boundaries.

MULTIPLE COMPARISONS: every test above is run once per regressor family, so a single p-column is
~20 tests asking the same question, and at ALPHA=0.05 roughly one of them comes up "significant"
by chance on every run. Correction is therefore applied DOWN each p-column -- across families,
within one test and one metric -- putting a q[...] beside every p[...]. The two contrasts are NOT
pooled with each other, and dR2 is NOT pooled with tracevar: those are separate questions (and the
two metrics are two views of the same data), so merging them would only raise the bar for no gain.
MC_METHOD picks the estimator:
    'fdr_bh'      (default) Benjamini-Hochberg. q = the expected share of false positives AMONG
                  the families you call significant -- q<0.05 over 8 flagged families means you
                  expect ~0.4 of them to be spurious. The right tool for a ~20-family screen, and
                  it stays valid under the positive correlation these families have (they share
                  the same channels and the same design matrix).
    'bonferroni'  controls the probability of ANY false positive. Much stricter; reach for it only
                  when making a confirmatory claim about one pre-specified family.
    'none'        raw p passed straight through (q == p), to reproduce pre-correction runs.
Figures star q by default (STAR_ON); raw p stays in the CSV and in the bar-panel titles, so nothing
is hidden. Correction fixes multiplicity ONLY -- it does nothing about the caveat below.

CAVEAT (printed at the end of every run, and it is not a small one): channels within an array
are NOT independent samples. The tests below treat each channel as a sample, so the effective n
is closer to the number of ARRAYS than the number of channels and the p-values are optimistic.
Read them as effect-size descriptors, and trust a result only when the direction is consistent
across the arrays making up a region (the per-array table is printed for exactly this reason).
Correction does not help with this: BH rescales p-values, it does not make the samples
independent, so q is optimistic in precisely the same way p is.

Works for a single session OR a POOLED (multi-session) fit -- it reads the same summary CSVs as
PlotArraySummary.py, so point SESSION at whichever folder ArrayRun.py wrote.

Light / login-node script (no acme, no SLURM -- it only reads the summary CSVs):
    python ArrayWiseAnalysis.py                   # default SESSION below
    python ArrayWiseAnalysis.py pooled_all        # the concatenated multi-session fit
    python ArrayWiseAnalysis.py 20230214          # one session
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
from scipy.stats import mannwhitneyu, kruskal, spearmanr

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
POOLED = SESSION.startswith('pooled_')
SUMMARY_DIR = os.path.join(results_dir, SESSION, '_contribution_summaries')
plots_dir = os.path.join(plots_base, SESSION, '_contribution_summaries')

# --- anatomy: which cortical area each array sits in (1-based array index, as in ArrayRun.py) ---
REGIONS = {1: 'V1-periph', 2: 'V1-periph', 3: 'V1-periph',
           4: 'V1-fovea',
           5: 'V4', 6: 'V4'}
REGION_ORDER = ['V1-periph', 'V1-fovea', 'V4']    # plotting / column order (low -> high in the hierarchy)

# Pairwise contrasts to test, as (label, [regions on one side], [regions on the other]).
# Regions are pooled within each side, so a side can be one region or several.
CONTRASTS = [
    ('V4 vs V1 (all)',   ['V4'],        ['V1-periph', 'V1-fovea']),
    ('V1-fovea vs V1-periph', ['V1-fovea'],  ['V1-periph']),
]

# --- within-V1 eccentricity check (question 4 above) ---
ECC_FAMILY = 'target_in_RF_onset'   # the stimulus-locked family expected to track eccentricity
ECC_REGIONS = ['V1-periph', 'V1-fovea']   # regions to run the channel-number correlation over

METRICS = ['dR2', 'tracevar']   # 'dR2' = unique (added-last) variance; 'tracevar' = gross, non-unique
ALPHA = 0.05
TOP_N_BARS = 8                  # how many families (strongest first) get a per-region bar panel

# --- multiple comparisons (see the MULTIPLE COMPARISONS block in the docstring) ---
MC_METHOD = 'fdr_bh'            # 'fdr_bh' (Benjamini-Hochberg) | 'bonferroni' | 'none'
STAR_ON = 'q'                   # draw the figure stars from 'q' (corrected) or 'p' (raw)


def load_summary():
    """Prefer the combined CSV; fall back to concatenating per-array CSVs. (Same loader as
    PlotArraySummary.py, so the two scripts always see identical data.)"""
    combined = os.path.join(SUMMARY_DIR, 'all_arrays_contributions.csv')
    if os.path.exists(combined):
        return pd.read_csv(combined)
    parts = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(SUMMARY_DIR, 'array*_contributions.csv')))]
    if not parts:
        raise SystemExit(f'No summary CSVs found in {SUMMARY_DIR} (run ArrayRun.py first)')
    return pd.concat(parts, ignore_index=True)


def attach_region(df):
    """Add a `region` column from REGIONS; drop channels whose array has no region mapping."""
    df = df.copy()
    df['region'] = df['array'].map(REGIONS)
    unmapped = sorted(set(df.loc[df['region'].isna(), 'array']))
    if unmapped:
        print(f'  (arrays {unmapped} have no REGIONS entry -- excluded)')
        df = df[df['region'].notna()]
    return df


def regions_present(df):
    """REGION_ORDER restricted to regions that actually have channels, order preserved."""
    have = set(df['region'])
    return [r for r in REGION_ORDER if r in have] + sorted(have - set(REGION_ORDER))


def family_order(df, cols):
    """Consistent family ordering: by global mean metric (strongest first). The shuffled `dummy`
    control is dropped -- it is a null-floor yardstick, not a result (cf. PlotArraySummary.py)."""
    means = df[cols].mean().sort_values(ascending=False)
    fams = [c.split('::')[1] for c in means.index]
    return [f for f in fams if f != 'dummy']


def metric_frame(df, fams, metric, relative=False):
    """Per-channel values for each family as a (channels x families) frame. `relative` divides each
    channel's value by that channel's full_R2, i.e. the SHARE of the model's explained variance the
    family accounts for -- comparable across regions with different overall fit quality."""
    cols = [f'{metric}::{f}' for f in fams if f'{metric}::{f}' in df.columns]
    M = df[cols].copy()
    M.columns = [c.split('::')[1] for c in cols]
    if relative:
        M = M.div(df['full_R2'].to_numpy(), axis=0)
    return M


def adjust_pvalues(p, method=MC_METHOD):
    """Correct ONE p-column, i.e. the same test repeated over the ~20 regressor families.
    NaNs (tests that could not run) are skipped and stay NaN, so they never inflate the count of
    tests being corrected for. See the docstring for which method to want when."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    n = int(ok.sum())
    if n == 0:
        return q
    if method == 'none':
        q[ok] = p[ok]
    elif method == 'bonferroni':
        q[ok] = np.clip(p[ok] * n, 0.0, 1.0)
    elif method == 'fdr_bh':
        order = np.argsort(p[ok])
        ranked = p[ok][order] * n / np.arange(1, n + 1)              # BH step-up ...
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]           # ... forced monotone, so a
        adj = np.empty(n)                                            #     family can never end up
        adj[order] = np.clip(ranked, 0.0, 1.0)                       #     with q < a smaller p's q
        q[ok] = adj
    else:
        raise ValueError(f'unknown MC_METHOD {method!r}')
    return q


def add_qvalues(S):
    """Insert a q[...] column immediately after every p[...] column of a stats frame. Correction
    runs down each column independently -- across families, never across contrasts or metrics."""
    out = {}
    for col in S.columns:
        out[col] = S[col]
        if col.startswith('p[') or col.startswith('p_rel['):
            out['q' + col[1:]] = adjust_pvalues(S[col].to_numpy())   # 'p[x]'->'q[x]', 'p_rel'->'q_rel'
    return pd.DataFrame(out)


def p_stars(v, alpha=ALPHA):
    """The conventional star ladder; '' for a test that could not run."""
    if v is None or not np.isfinite(v):
        return ''
    if v < 0.001:
        return '***'
    if v < 0.01:
        return '**'
    if v < alpha:
        return '*'
    return 'n.s.'


def star_value(srow, pcol):
    """The number the stars are drawn from for a given p-column: the corrected q by default."""
    col = ('q' + pcol[1:]) if STAR_ON == 'q' else pcol
    if srow is None or col not in srow.index:
        return np.nan
    return float(srow[col])


def star_source_label():
    """What the stars mean, for the figure captions -- so a reader never has to guess."""
    if STAR_ON != 'q' or MC_METHOD == 'none':
        return 'uncorrected p'
    return {'fdr_bh': 'BH-FDR q (across families)',
            'bonferroni': 'Bonferroni q (across families)'}[MC_METHOD]


def _mannwhitney(a, b):
    """Two-sided Mann-Whitney -> (p, rank-biserial). rb = +1 if every side-A channel outranks every
    side-B channel, -1 for the reverse, 0 for complete interleaving. NaN if a side is empty or the
    values are entirely tied (U is undefined there)."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not (a.size and b.size):
        return np.nan, np.nan
    try:
        res = mannwhitneyu(a, b, alternative='two-sided')
    except ValueError:
        return np.nan, np.nan
    return float(res.pvalue), float(2.0 * res.statistic / (a.size * b.size) - 1.0)


def _kruskal(groups):
    """Kruskal-Wallis omnibus over the regions; NaN unless at least 3 regions have data (with 2 it
    would just restate the Mann-Whitney)."""
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if g.size]
    if len(groups) < 3:
        return np.nan
    try:
        return float(kruskal(*groups).pvalue)
    except ValueError:
        return np.nan


def run_stats(df, fams, metric):
    """Per-family region means + contrast p-values + the probe-position (array-index) control.
    Returns a tidy DataFrame, one row per family, with a q[...] beside every p[...]."""
    regs = regions_present(df)
    abs_M = metric_frame(df, fams, metric)
    rel_M = metric_frame(df, fams, metric, relative=True)
    rows = []
    for fam in abs_M.columns:
        row = {'family': fam}
        for r in regs:
            sel = (df['region'] == r).to_numpy()
            row[f'mean_{r}'] = float(np.nanmean(abs_M[fam].to_numpy()[sel]))
            row[f'rel_{r}'] = float(np.nanmean(rel_M[fam].to_numpy()[sel]))
        # pairwise contrasts (Mann-Whitney: no normality assumption, small unbalanced groups).
        # Run on the absolute metric AND on the share of full_R2, since a region can lead on one
        # and not the other -- that divergence is question 2 in the docstring.
        for label, side_a, side_b in CONTRASTS:
            sa = df['region'].isin(side_a).to_numpy()
            sb = df['region'].isin(side_b).to_numpy()
            row[f'p[{label}]'], row[f'rb[{label}]'] = _mannwhitney(abs_M[fam].to_numpy()[sa],
                                                                   abs_M[fam].to_numpy()[sb])
            row[f'p_rel[{label}]'], row[f'rb_rel[{label}]'] = _mannwhitney(rel_M[fam].to_numpy()[sa],
                                                                           rel_M[fam].to_numpy()[sb])
        # omnibus across all regions
        row['p[Kruskal all regions]'] = _kruskal(
            [abs_M[fam].to_numpy()[(df['region'] == r).to_numpy()] for r in regs])
        row['p_rel[Kruskal all regions]'] = _kruskal(
            [rel_M[fam].to_numpy()[(df['region'] == r).to_numpy()] for r in regs])
        # probe-position control: a purely monotonic trend along the probe is suspicious
        v = abs_M[fam].to_numpy()
        ok = np.isfinite(v)
        if ok.sum() > 2:
            rho, p = spearmanr(df['array'].to_numpy()[ok], v[ok])
            row['rho[vs array idx]'], row['p[vs array idx]'] = float(rho), float(p)
        rows.append(row)
    # one correction pass per p-column, over the families just tested
    return add_qvalues(pd.DataFrame(rows))


def plot_region_heatmap(df, fams, metric, stats, relative=False):
    """regions x families heatmap of the mean metric (absolute, or as a share of full_R2), with a
    significance strip underneath: one row per contrast plus the Kruskal omnibus, shaded and starred
    per family. The strip reads the p-columns matching the panel above -- p_rel[...] for the relative
    view, p[...] for the absolute one -- so the stars always refer to the quantity being shown."""
    regs = regions_present(df)
    M = metric_frame(df, fams, metric, relative=relative)
    used = list(M.columns)
    A = np.vstack([np.nanmean(M.to_numpy()[(df['region'] == r).to_numpy()], axis=0) for r in regs])
    if relative:
        A = 100.0 * A   # percent of the channel's full_R2
    vlim = max(float(np.nanpercentile(np.abs(A), 98)), 1e-12)

    sig_rows = [lab for lab, _, _ in CONTRASTS] + ['Kruskal all regions']
    prefix = 'p_rel[' if relative else 'p['
    ST = stats.set_index('family')
    V = np.full((len(sig_rows), len(used)), np.nan)     # the value the stars come from
    for i, lab in enumerate(sig_rows):
        col = f'{prefix}{lab}]'
        for j, fam in enumerate(used):
            if fam in ST.index and col in ST.columns:
                V[i, j] = star_value(ST.loc[fam], col)
    # 0 = n.s., 1/2/3 = the three star levels, so significance is legible from the shading alone
    L = np.zeros_like(V)
    for lvl, thr in ((1, ALPHA), (2, 0.01), (3, 0.001)):
        L[np.isfinite(V) & (V < thr)] = lvl

    fig, (ax, sax) = plt.subplots(
        2, 1, sharex=True,
        figsize=(max(8, 0.45 * len(used) + 2), 0.6 * len(regs) + 0.4 * len(sig_rows) + 3.0),
        gridspec_kw={'height_ratios': [len(regs), len(sig_rows)], 'hspace': 0.08})
    im = ax.imshow(A, aspect='auto', cmap='RdBu_r',
                   norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim))
    ax.set_yticks(range(len(regs)))
    ax.set_yticklabels(regs)
    ax.tick_params(labelbottom=False)
    kind = f'{metric} as % of full R2' if relative else f'mean {metric}'
    ax.set_title(f'{SESSION}: {kind}, by region\n'
                 f'strip below: * {star_source_label()} < {ALPHA}, ** < 0.01, *** < 0.001',
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=kind)

    sax.imshow(L, aspect='auto', cmap='Greens', vmin=0, vmax=3)
    for i in range(len(sig_rows)):
        for j in range(len(used)):
            s = p_stars(V[i, j])
            if s and s != 'n.s.':
                sax.text(j, i, s, ha='center', va='center', fontsize=6,
                         color='w' if L[i, j] >= 2 else 'k')   # stars stay legible on the dark cells
    sax.set_yticks(range(len(sig_rows)))
    sax.set_yticklabels(sig_rows, fontsize=7)
    sax.set_xticks(range(len(used)))
    sax.set_xticklabels(used, rotation=90, fontsize=7)
    # keep the strip the same width as the heatmap despite the colorbar stealing space from ax
    fig.canvas.draw()
    box, sbox = ax.get_position(), sax.get_position()
    sax.set_position([box.x0, sbox.y0, box.width, sbox.height])
    tag = 'relative' if relative else 'absolute'
    out = os.path.join(plots_dir, f'region_{metric}_{tag}_heatmap.pdf')
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f'  saved {out}')


def _contrast_x(regs, side):
    """x position of one side of a contrast = the centre of the bars it pools (a side may cover
    several regions, e.g. 'V1 (all)' spans two bars). None if none of them are present."""
    idx = [j for j, r in enumerate(regs) if r in side]
    return float(np.mean(idx)) if idx else None


def plot_family_bars(df, fams, metric, stats, relative=False):
    """One panel per strong family: region means (bars) with every channel overlaid as a dot, so the
    within-region spread -- and any single array carrying a region -- is visible rather than hidden
    behind the mean. Each contrast gets a significance bracket spanning the bars it actually pools,
    and the Kruskal omnibus is starred in the panel title.

    `relative` plots the family as a % of each channel's full_R2 instead of the raw metric, and
    switches the annotations to the matching p_rel/rb_rel columns, so the stars and the effect sizes
    always describe the quantity drawn. The panel ORDER is the absolute one in both figures, so the
    two can be read side by side -- a family that leads the absolute figure but flattens out in the
    relative one is a region-fit-quality effect rather than a genuine reweighting."""
    regs = regions_present(df)
    ST = stats.set_index('family')
    pfx, rpfx = ('p_rel[', 'rb_rel[') if relative else ('p[', 'rb[')
    ylab = f'{metric} as % of full R2' if relative else metric
    M = metric_frame(df, fams, metric, relative=relative)
    if relative:
        M = 100.0 * M
    show = [f for f in fams if f in M.columns][:TOP_N_BARS]
    if not show:
        return
    ncol = min(4, len(show))
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), squeeze=False)
    colors = plt.get_cmap('tab10')
    for i, fam in enumerate(show):
        ax = axes[i // ncol][i % ncol]
        for j, r in enumerate(regs):
            v = M[fam].to_numpy()[(df['region'] == r).to_numpy()]
            v = v[np.isfinite(v)]
            ax.bar(j, np.mean(v) if v.size else 0.0, color=colors(j), alpha=0.45, width=0.65)
            # jitter the per-channel dots so overlapping values stay countable
            jit = (np.random.RandomState(0).rand(v.size) - 0.5) * 0.3
            ax.scatter(j + jit, v, s=9, color=colors(j), edgecolors='k', linewidths=0.3, zorder=3)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(range(len(regs)))
        ax.set_xticklabels(regs, rotation=20, fontsize=7)
        srow = ST.loc[fam] if fam in ST.index else None

        # one bracket per contrast, stacked upwards above the tallest bar
        ylo, yhi = ax.get_ylim()
        span = (yhi - ylo) or 1.0
        y = yhi + 0.04 * span
        for label, side_a, side_b in CONTRASTS:
            xa, xb = _contrast_x(regs, side_a), _contrast_x(regs, side_b)
            v = star_value(srow, f'{pfx}{label}]')
            if xa is None or xb is None or not np.isfinite(v):
                continue
            # stars = is it reliable, rb = how separated (and, by its sign, which way round).
            # The exact p/q stay in the CSV; on the figure they would only crowd out the effect size.
            rb = srow.get(f'{rpfx}{label}]', np.nan) if srow is not None else np.nan
            txt = p_stars(v) + (f'  rb={rb:+.2f}' if np.isfinite(rb) else '')
            ax.plot([xa, xa, xb, xb], [y, y + 0.02 * span, y + 0.02 * span, y],
                    lw=0.8, color='k', clip_on=False)
            ax.text((xa + xb) / 2.0, y + 0.025 * span, txt,
                    ha='center', va='bottom', fontsize=7, clip_on=False)
            y += 0.14 * span
        ax.set_ylim(ylo, max(yhi, y))

        # headline contrast spelled out in words, so the direction needs no decoding
        rbtxt = ''
        if CONTRASTS and srow is not None:
            lab0, sa0, sb0 = CONTRASTS[0]
            rb0 = srow.get(f'{rpfx}{lab0}]', np.nan)
            if np.isfinite(rb0):
                # name the winning side the way the contrast label already words it ('V4 vs V1 (all)'
                # -> 'V4' / 'V1 (all)'), falling back to the region lists for a label without ' vs '
                names = lab0.split(' vs ') if ' vs ' in lab0 else ['+'.join(sa0), '+'.join(sb0)]
                side = ('no separation' if abs(rb0) < 1e-12 else
                        f'{names[0] if rb0 > 0 else names[1]} higher')
                rbtxt = f'\n{lab0}: rb={rb0:+.2f} ({side})'
        kv = star_value(srow, f'{pfx}Kruskal all regions]')
        ktxt = f'   KW {p_stars(kv)}' if np.isfinite(kv) else ''
        ax.set_title(f'{fam}{ktxt}{rbtxt}', fontsize=8)
        if i % ncol == 0:
            ax.set_ylabel(ylab)
    for k in range(len(show), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle(f'{SESSION}: {ylab} by region (bar = region mean, dot = one channel)\n'
                 f'brackets = Mann-Whitney per contrast, KW = Kruskal-Wallis omnibus;  '
                 f'stars on {star_source_label()}: * < {ALPHA}, ** < 0.01, *** < 0.001\n'
                 f'rb = rank-biserial, signed as the contrast label reads: rb>0 means the '
                 f'FIRST-named side is higher (+1 = every one of its channels above every '
                 f'channel of the second side, 0 = no separation)', fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    tag = 'relative' if relative else 'absolute'
    out = os.path.join(plots_dir, f'region_{metric}_{tag}_bars.pdf')   # same naming as the heatmaps
    fig.savefig(out); plt.close(fig)
    print(f'  saved {out}')


def plot_eccentricity(df, metric='dR2'):
    """ECC_FAMILY vs channel number, to see whether the gradient is continuous ALONG the probe rather
    than stepping at the array boundaries. Channels in ECC_REGIONS get the correlation + fit; the
    remaining regions are plotted greyed-out for reference (they are a different area, so the
    eccentricity logic does not apply to them)."""
    col = f'{metric}::{ECC_FAMILY}'
    if col not in df.columns:
        print(f'  (no {col} in the summary -- skipping the eccentricity figure)')
        return
    d = df.sort_values('channel')
    inside = d['region'].isin(ECC_REGIONS).to_numpy()
    x, y = d['channel'].to_numpy(), d[col].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 4))
    for r in regions_present(d):
        sel = (d['region'] == r).to_numpy()
        ax.scatter(x[sel], y[sel], s=26, label=f'{r} (n={int(sel.sum())})',
                   alpha=0.9 if r in ECC_REGIONS else 0.35,
                   edgecolors='k', linewidths=0.3)
    # array boundaries: the channel numbering is contiguous, so these are where the probe groups split
    for a in sorted(d['array'].unique())[1:]:
        lo = d.loc[d['array'] == a, 'channel'].min()
        ax.axvline(lo - 0.5, color='grey', lw=0.6, ls=':')
        ax.text(lo, ax.get_ylim()[1], f'a{int(a)}', fontsize=6, color='grey',
                va='top', ha='left')

    ok = inside & np.isfinite(y)
    if ok.sum() > 2:
        rho, p = spearmanr(x[ok], y[ok])
        b = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color='k', lw=1.2, ls='--')
        # a single pre-specified test on one family, so no correction applies here (unlike the
        # family-wise screens above) -- these stars are the raw p
        ax.set_title(f'{SESSION}: {ECC_FAMILY} along the probe  --  within {"+".join(ECC_REGIONS)}: '
                     f'rho={rho:+.3f}, p={p:.2g} {p_stars(p)} (uncorrected: one planned test)',
                     fontsize=10)
    else:
        ax.set_title(f'{SESSION}: {ECC_FAMILY} along the probe', fontsize=10)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('channel (probe order)')
    ax.set_ylabel(f'{metric}::{ECC_FAMILY}')
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(plots_dir, f'eccentricity_{ECC_FAMILY}_{metric}.pdf')
    fig.savefig(out); plt.close(fig)
    print(f'  saved {out}')


if __name__ == '__main__':
    os.makedirs(plots_dir, exist_ok=True)
    df = attach_region(load_summary())
    regs = regions_present(df)
    print(f'Loaded {len(df)} channels; regions: '
          + ', '.join(f'{r} (n={int((df["region"] == r).sum())})' for r in regs))

    # full_R2 by region -- the denominator for every relative number below, so show it first
    print('\nfull_R2 by region:')
    print(df.groupby('region')['full_R2'].agg(['mean', 'median', 'std', 'count']).round(4).to_string())

    # per-array table: the honest check on every region-level claim (see the CAVEAT in the docstring)
    print('\nfull_R2 by array (does every array in a region agree?):')
    per_arr = df.groupby('array').agg(region=('region', 'first'), full_R2=('full_R2', 'mean'),
                                      n=('channel', 'size'))
    print(per_arr.round(4).to_string())

    for metric in METRICS:
        cols = [c for c in df.columns if c.startswith(f'{metric}::')]
        if not cols:
            print(f'\n-- {metric}: no {metric}:: columns in the summary CSV, skipping')
            continue
        fams = family_order(df, cols)
        stats = run_stats(df, fams, metric)
        print(f'\n-- {metric}: {len(fams)} families, {len(regs)} regions '
              f'(correction: {MC_METHOD}, down each p-column over the families)')
        with pd.option_context('display.width', 250, 'display.max_columns', 50):
            print(stats.head(TOP_N_BARS).round(6).to_string(index=False))

        # what survives correction -- the headline of the whole script, and the clearest way to see
        # how much of the raw-p tally was multiplicity
        for lab in [l for l, _, _ in CONTRASTS] + ['Kruskal all regions']:
            pcol, qcol = f'p[{lab}]', f'q[{lab}]'
            if qcol not in stats.columns:
                continue
            n_tested = int(stats[pcol].notna().sum())
            raw = stats.loc[stats[pcol] < ALPHA, 'family'].tolist()
            hit = stats.loc[stats[qcol] < ALPHA, 'family'].tolist()
            print(f'  {lab}: {len(raw)}/{n_tested} families at raw p<{ALPHA} -> '
                  f'{len(hit)} survive {MC_METHOD}' + (f': {", ".join(hit)}' if hit else ''))

        out_csv = os.path.join(plots_dir, f'region_{metric}_stats.csv')
        stats.to_csv(out_csv, index=False)
        print(f'  saved {out_csv}')

        plot_region_heatmap(df, fams, metric, stats, relative=False)
        plot_region_heatmap(df, fams, metric, stats, relative=True)
        plot_family_bars(df, fams, metric, stats, relative=False)
        plot_family_bars(df, fams, metric, stats, relative=True)

    plot_eccentricity(df, metric='dR2')

    print(f'\nDone -> {plots_dir}')
    print('NOTE: channels within an array are not independent, so the effective n is closer to the '
          'number of arrays than the number of channels -- read the p-values as effect-size '
          'descriptors and confirm each result against the per-array table above.')
    print(f'      Figure stars are {star_source_label()}; correction handles the many families '
          'tested per column, NOT the non-independence above, so q is optimistic in exactly the '
          'same way p is. Quote rb[...] for the size of an effect.')
