#!/usr/bin/env python
"""
=============================================================================================
figures.py -- every figure page of the V1/V4 naturalistic-foraging manuscript
=============================================================================================
Writes writing/figures/figNN_<name>.pdf, which writing/latex/build.py places in the manuscript.

House style (writing/resources/editing/FIGURE_STYLE.md): one column width, 16.2 cm, saved 1:1;
Calibri at 8 pt and 9 pt bold for panel letters only, live and embedded; data lines 1.5 pt,
guides and axes 0.6 pt; opaque tints; heatmaps as vector quads, significance as closed
dark outlines. Tracked colors, the same in every figure (REGION_COL, STATE_COL):
  V1 periphery sage #7D9988, V1 fovea dark red-brown #521E21, V4 light pink #EDB0D0; inattentive state (0)
  plum #83336E, state 1 orange #B35919, attentive state (2) teal #448A87, state 3 #262262. Significance is light teal
  (SIG_COL). The table heatmaps run white -> house teal at HEAT_ALPHA and carry a small color key per
  region or state.

Nothing is recomputed here. Sources (Methods: 1-4, Results: 5-8):
  fig01_task, fig02_recordings, fig03_rf_mapping, fig04_states_method
                             writing/figures/art/{setup, neural_rec_method, VR_RF_mapping_method,
                             states_method}.pdf (Illustrator), cropped                    step art
  fig05_states_behavior      plots/states_analysis/*.pdf (values read out of the PDFs) and
                             rt_pairwise_comparisons.csv                                  fig05
  fig06_state_erp            data/state_erp.npz (export_state_erp.py)                     fig06
  fig07_kernels              data/fig07_encoding_model.mat (export_figure_data.py); shading
                             and p from pooled_all_mean_kernels_permtest_norm.pdf         fig07
  fig08_contributions        data/fig08_contributions.mat; array-level dots from
                             overview_mean_dR2_by_array.pdf; q from region_dR2_stats.csv  fig08
Results figures group the encoding-model families into visual and non-visual (VISUAL below).
"Values read out of the PDF" means pdfplot.py: the curves, bands, cells and markers matplotlib
wrote, converted to data units with the tick labels.

Run (warping env, login node is fine), then build the manuscript:
    cd /mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/code/paper
    /gs/home/patels/.conda/envs/warping/bin/python export_figure_data.py      # only if the model changed
    /gs/home/patels/.conda/envs/warping/bin/python export_state_erp.py        # Figure 6
    /gs/home/patels/.conda/envs/warping/bin/python figures.py                 # or: figures.py fig06 fig07
    /gs/home/patels/.conda/envs/warping/bin/python ../../writing/latex/build.py      # -> writing/latex/V01.pdf
"""

import os
import re
import sys
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm, to_rgb
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import NullLocator
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

# ── Settings ──────────────────────────────────────────────────────────────────────────────
ROOT = '/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, 'writing', 'figures')          # figure pages
ART_DIR = os.path.join(OUT_DIR, 'art')                       # Illustrator .ai and their PDF exports
DATA_DIR = os.path.join(OUT_DIR, 'data')                     # exported results they draw
sys.path.insert(0, HERE)
from pdfplot import read_pdf                                 # noqa: E402

FONT_NAME = 'Calibri'
FONT_FILES = ['CALIBRI.TTF', 'CALIBRIB.TTF', 'CALIBRII.TTF', 'CALIBRIZ.TTF']
FONT_SIZE = 8           # everything a reader reads inside a panel
FONT_SIZE_BIG = 9       # panel letters ONLY
FIG_W = 16.2            # \textwidth of the manuscript, cm
LW_GUIDE = 0.6
AXIS_COLOR = (0.25, 0.25, 0.28)
LETTER_COLOR = (0.13, 0.13, 0.16)
CM = 1 / 2.54

MINUS, DEG, MU, TIMES, ARROW = '−', '°', 'µ', '×', '←'
DR2 = 'ΔR²'
R2LAB = 'R²'
SUP = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')

REGION_NAMES = ['V1 periphery', 'V1 fovea', 'V4']
REGION_SHORT = ['V1p', 'V1f', 'V4']
REG_RUNG = [2, 1, 0]            # ladder rung (0 = darkest) for regions 1, 2, 3: V4 darkest
LETTERS = 'abcdefghijklmnop'


def hx(h):
    return np.array(to_rgb(h))


PAL = {
    'lfp': hx('#008287'), 'eye': hx('#5C368C'), 'beh': hx('#E6731A'),
    'rf': hx('#B83D70'), 'task': hx('#248336'),
    'lad': {
        'lfp': [hx('#124E55'), hx('#017F84'), hx('#4EA8AC'), hx('#9ECFD1')],
        'eye': [hx('#583485'), hx('#8265A7'), hx('#A793C1'), hx('#CDC2DC')],
        'beh': [hx('#653D24'), hx('#AF5C1E'), hx('#E8802F'), hx('#F3BB90')],
        'rf': [hx('#72304F'), hx('#BC4979'), hx('#D284A4'), hx('#E6BBCD')],
        'task': [hx('#01531A'), hx('#248336'), hx('#54AE5D'), hx('#80DA86')],
    },
    'pair': {'lfp': [hx('#09696F'), hx('#6DB7BA')]},
    'ink': hx('#212129'), 'grey': hx('#7A7A80'), 'grey_mid': hx('#9E9EA1'), 'grey_lt': hx('#CCCCD1'),
}
PAL['lad']['null'] = [PAL['grey']] * 4


def setup_style():
    for name in FONT_FILES:
        path = os.path.join(os.path.expanduser('~/.fonts'), name)
        if os.path.exists(path):
            fm.fontManager.addfont(path)
    if FONT_NAME not in {f.name for f in fm.fontManager.ttflist}:
        raise SystemExit(f'{FONT_NAME} not found in ~/.fonts: matplotlib would silently use DejaVu Sans')
    plt.rcParams.update({
        'font.family': FONT_NAME, 'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE, 'axes.labelsize': FONT_SIZE, 'legend.fontsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE, 'ytick.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE,
        'axes.titleweight': 'normal', 'axes.titlepad': 4, 'axes.titlecolor': LETTER_COLOR,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.5, 'lines.solid_capstyle': 'butt',
        'axes.edgecolor': AXIS_COLOR, 'axes.labelcolor': AXIS_COLOR,
        'xtick.color': AXIS_COLOR, 'ytick.color': AXIS_COLOR,
        'xtick.major.size': 0, 'ytick.major.size': 0, 'xtick.minor.size': 0, 'ytick.minor.size': 0,
        'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5, 'axes.labelpad': 3,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.facecolor': 'none',
        'axes.unicode_minus': True, 'axes.formatter.useoffset': False,
        'axes.formatter.limits': (-9, 9), 'axes.formatter.use_mathtext': False,
        'legend.frameon': False, 'legend.handlelength': 1.75, 'legend.handletextpad': 0.4,
        'legend.columnspacing': 1.0, 'legend.borderpad': 0, 'legend.borderaxespad': 0,
        'legend.labelspacing': 0.25,
        'pdf.fonttype': 42, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })


# ── Page and layout, in centimetres ───────────────────────────────────────────────────────
class Page:
    def __init__(self, w_cm, h_cm):
        self.w, self.h = w_cm, h_cm
        self.fig = plt.figure(figsize=(w_cm * CM, h_cm * CM))

    def axes(self, x, y, w, h):
        return self.fig.add_axes([x / self.w, y / self.h, w / self.w, h / self.h])

    def grid(self, nrow, ncol, k, M):
        """Panel k (1-based, reading order) of a grid; M = [l b r t hgap vgap] in cm."""
        l, b, r, t, hg, vg = M
        w = (self.w - l - r - (ncol - 1) * hg) / ncol
        h = (self.h - b - t - (nrow - 1) * vg) / nrow
        row, col = (k - 1) // ncol, (k - 1) % ncol
        return l + col * (w + hg), self.h - t - (row + 1) * h - row * vg, w, h

    def grid_axes(self, nrow, ncol, k, M):
        return self.axes(*self.grid(nrow, ncol, k, M))

    def cm_of(self, ax):
        p = ax.get_position()
        return p.x0 * self.w, p.y0 * self.h, p.width * self.w, p.height * self.h

    def letter_at(self, x, y_top, txt, up=0.55, left=0.95):
        self.fig.text((x - left) / self.w, (y_top + up) / self.h, txt, fontsize=FONT_SIZE_BIG,
                      fontweight='bold', color=LETTER_COLOR, ha='left', va='center')

    def letter(self, ax, txt, up=0.55):
        x, y, w, h = self.cm_of(ax)
        self.letter_at(x, y + h, txt, up)

    def text_cm(self, x, y, s, **kw):
        kw.setdefault('color', AXIS_COLOR)
        self.fig.text(x / self.w, y / self.h, s, fontsize=FONT_SIZE, **kw)

    def save(self, name, subdir=''):
        align_ylabels(self.fig)
        os.makedirs(os.path.join(OUT_DIR, subdir), exist_ok=True)
        path = os.path.join(OUT_DIR, subdir, name + '.pdf')
        self.fig.savefig(path)
        plt.close(self.fig)
        print(f'  saved {path}')


def align_ylabels(fig, pad_pt=3.0):
    """Y labels of axes sharing a left edge sit at one distance from the plot: each label's right
    edge goes pad_pt left of the widest tick labels of its column."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    groups = {}
    for ax in fig.axes:
        if ax.get_ylabel() and ax.yaxis.get_label_position() == 'left':
            groups.setdefault(round(ax.get_position().x0, 4), []).append(ax)
    for axs in groups.values():
        lefts = []
        for ax in axs:
            tl = [t.get_window_extent(r) for t in ax.get_yticklabels() if t.get_visible() and t.get_text()]
            lefts.append(min(b.x0 for b in tl) if tl else ax.bbox.x0)
        target = min(lefts) - pad_pt * px
        for ax in axs:
            lab = ax.yaxis.label
            ax.yaxis.set_label_coords(-0.1, 0.5)
            fig.canvas.draw()
            b = lab.get_window_extent(r)
            x_disp = ax.transAxes.transform((-0.1, 0.5))[0] + (target - b.x1)
            ax.yaxis.set_label_coords((x_disp - ax.bbox.x0) / ax.bbox.width, 0.5)


def square(page, ax):
    x, y, w, h = page.cm_of(ax)
    s = min(w, h)
    ax.set_position([(x + (w - s) / 2) / page.w, (y + (h - s) / 2) / page.h, s / page.w, s / page.h])


def boxed(ax):
    for s in ax.spines.values():
        s.set_visible(True)


def legend_below(page, ax, handles, labels, ncol=1, down=1.25):
    x, y, w, h = page.cm_of(ax)
    page.fig.legend(handles, labels, loc='center', ncol=ncol,
                    bbox_to_anchor=((x + w / 2) / page.w, (y - down) / page.h),
                    bbox_transform=page.fig.transFigure)


def line_handle(col):
    return Line2D([], [], color=col, lw=1.5)


def dot_handle(col, ms=3):
    return Line2D([], [], color=col, marker='o', ls='none', ms=ms, mfc=col, mec='none')


def patch_handle(col):
    return Rectangle((0, 0), 1, 1, fc=col, ec='none')


# ── Values ────────────────────────────────────────────────────────────────────────────────
def tint(col, amount):
    """Blend toward white, opaque."""
    return np.asarray(col) * (1 - amount) + amount


def band_tint(col):
    return tint(col, 0.86)


def pad_lim(v, pad=0.08):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [0.0, 1.0]
    lo, hi = v.min(), v.max()
    if hi == lo:
        hi = lo + 1
    return [lo - pad * (hi - lo), hi + pad * (hi - lo)]


def headroom(yl, amount):
    return [yl[0], yl[1] + amount * (yl[1] - yl[0])]


def edges(x):
    x = np.asarray(x, float)
    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    d = np.diff(x)
    return np.concatenate([[x[0] - d[0] / 2], x[:-1] + d / 2, [x[-1] + d[-1] / 2]])


def runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def sci(v):
    if v == 0 or not np.isfinite(v):
        return '0'
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10 ** e
    if abs(m) >= 9.95:
        m, e = m / 10, e + 1
    return f'{m:.1f}{TIMES}10{str(e).translate(SUP)}'.replace('-', MINUS)


def num(v, fmt='{:.2g}'):
    return fmt.format(v).replace('-', MINUS)


def stars(q):
    if not np.isfinite(q) or q >= 0.05:
        return ''
    return '***' if q < 0.001 else '**' if q < 0.01 else '*'


# ── Drawing ───────────────────────────────────────────────────────────────────────────────
def guides(ax, xlim, ylim, zero_x=True):
    ax.plot(xlim, [0, 0], '-', color=PAL['grey_lt'], lw=LW_GUIDE, zorder=1)
    if zero_x:
        ax.plot([0, 0], ylim, '-', color=PAL['grey_lt'], lw=LW_GUIDE, zorder=1)


def sig_band(ax, x, mask, yl, col):
    e = edges(x)
    for s, t in runs(mask):
        ax.add_patch(Rectangle((e[s], yl[0]), e[t + 1] - e[s], yl[1] - yl[0], fc=col, ec='none', lw=0, zorder=0))


def sig_bar(ax, x, mask, yl, col):
    e = edges(x)
    h = 0.03 * (yl[1] - yl[0])
    for s, t in runs(mask):
        ax.add_patch(Rectangle((e[s], yl[1] - h), e[t + 1] - e[s], h, fc=col, ec='none', lw=0, zorder=3))


def sig_ticks(ax, x, mask, strip, row, nrow, col):
    """One bar per series, stacked in a strip reserved above the data (row 0 on top)."""
    e = edges(x)
    band = (strip[1] - strip[0]) / nrow
    h = 0.62 * band
    y0 = strip[1] - (row + 1) * band + (band - h) / 2
    for s, t in runs(mask):
        ax.add_patch(Rectangle((e[s], y0), e[t + 1] - e[s], h, fc=col, ec='none', lw=0, zorder=3))


def sig_outline(ax, x, y, mask, col='white', lw=1.2):
    """Closed outline around the True cells of mask [len(y) x len(x)], built from cell edges."""
    ex, ey = edges(x), edges(y)
    m = np.pad(np.asarray(mask, bool), 1)
    segs = []
    for i in range(1, m.shape[0] - 1):
        for j in range(1, m.shape[1] - 1):
            if not m[i, j]:
                continue
            x0, x1, y0, y1 = ex[j - 1], ex[j], ey[i - 1], ey[i]
            if not m[i - 1, j]:
                segs.append([(x0, y0), (x1, y0)])
            if not m[i + 1, j]:
                segs.append([(x0, y1), (x1, y1)])
            if not m[i, j - 1]:
                segs.append([(x0, y0), (x0, y1)])
            if not m[i, j + 1]:
                segs.append([(x1, y0), (x1, y1)])
    if segs:
        ax.add_collection(LineCollection(segs, colors=[col], linewidths=lw, capstyle='projecting', zorder=4))


def heatmap(ax, x, y, C, cmap, norm):
    """Vector quads; NaN cells are left blank."""
    return ax.pcolormesh(edges(x), edges(y), np.ma.masked_invalid(C), cmap=cmap, norm=norm,
                         shading='flat', linewidth=0.25, edgecolors='face', antialiased=True,
                         rasterized=False)


def colorbar_h(page, ax, cmap, ends, gap_cm, label):
    """A 0.20 cm strip under ax, full axes width; the two end values flush left and right,
    the measure on the line under them."""
    x, y, w, h = page.cm_of(ax)
    cax = page.axes(x, y - gap_cm - 0.20, w, 0.20)
    t = np.linspace(0, 1, 65)
    cax.pcolormesh(t, [0, 1], ((t[:-1] + t[1:]) / 2)[None, :], cmap=cmap, vmin=0, vmax=1,
                   shading='flat', linewidth=0.25, edgecolors='face', antialiased=True)
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([])
    cax.set_yticks([])
    boxed(cax)
    yb = y - gap_cm - 0.20
    page.text_cm(x, yb - 0.08, ends[0], ha='left', va='top')
    page.text_cm(x + w, yb - 0.08, ends[-1], ha='right', va='top')
    if label:
        page.text_cm(x + w / 2, yb - 0.42, label, ha='center', va='top')


def colorbar_v(page, ax, cmap, labels, label):
    """A vertical strip right of ax, 76% of its height, three tick labels."""
    x, y, w, h = page.cm_of(ax)
    cax = page.axes(x + w + 0.15, y + 0.12 * h, 0.18, 0.76 * h)
    t = np.linspace(0, 1, 65)
    cax.pcolormesh([0, 1], t, ((t[:-1] + t[1:]) / 2)[:, None], cmap=cmap, vmin=0, vmax=1,
                   shading='flat', linewidth=0.25, edgecolors='face', antialiased=True)
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks([0, 0.5, 1], labels)
    boxed(cax)
    if label:
        # matplotlib's labelpad clears the shortest tick label, not the widest, so a three-character
        # end value runs into the measure; measure the numbers and set the label clear of them.
        page.fig.canvas.draw()
        r = page.fig.canvas.get_renderer()
        right = max(t.get_window_extent(r).x1 for t in cax.get_yticklabels() if t.get_text())
        page.text_cm(right / page.fig.dpi * 2.54 + 0.08, y + h / 2, label,
                     rotation=270, ha='left', va='center')


def bar_rect(ax, x, w, y0, y1, col):
    ax.add_patch(Rectangle((x - w / 2, y0), w, y1 - y0, fc=col, ec='none', lw=0))


def bracket(ax, x1, x2, y, h, txt):
    if not txt:
        return
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], '-', color=PAL['ink'], lw=LW_GUIDE)
    ax.text((x1 + x2) / 2, y, txt, ha='center', va='baseline', color=PAL['ink'], fontsize=FONT_SIZE)


def log_ticks(axis, values):
    """Round ticks on a log axis, printed in full, no minor ticks, no mathtext."""
    axis.set_minor_locator(NullLocator())
    labels = []
    for v in values:
        e = np.log10(v)
        labels.append(f'10{str(int(e)).translate(SUP)}' if v >= 1000 else f'{v:g}')
    axis.set_ticks(values, labels)


# ── Names ─────────────────────────────────────────────────────────────────────────────────
def family_style(fam):
    """(manuscript name, hue, ladder) of a regressor family."""
    table = {
        'saccade_onset': ('Saccade onset', 'eye'), 'saccade_offset': ('Saccade offset', 'eye'),
        'reaction_onset': ('Reaction', 'beh'), 'state': ('Cognitive state', 'beh'),
        'trial_onset': ('Trial onset', 'task'), 'stim_onset': ('Stimulus onset', 'task'),
        'reward_onset': ('Reward', 'task'), 'block_onset': ('Block change', 'task'),
        'diff_easy': ('Difficulty', 'task'), 'correct': ('Correct', 'task'),
        'movement_left': ('Left choice', 'task'), 'dummy': ('Shuffled control', 'null'),
    }
    if fam in table:
        name, g = table[fam]
    else:
        cat, _, _, kind = fam.split('_')                    # e.g. target_in_RF_onset
        name, g = cat.capitalize() + (' enters RF' if kind == 'onset' else ' leaves RF'), 'rf'
    col = PAL['grey'] if g == 'null' else PAL[g]
    return name, col, PAL['lad'][g]


def region_of_channel(ch):
    a = np.ceil(np.asarray(ch) / 32)
    return np.where(a <= 3, 1, np.where(a == 4, 2, 3))


def load(name):
    return sio.loadmat(os.path.join(DATA_DIR, name + '.mat'), squeeze_me=True)


PLOTS = os.path.join(ROOT, 'plots')

SA = os.path.join(PLOTS, 'states_analysis')
LEM = os.path.join(PLOTS, 'Linear_encoding_model', 'all_regressors', '100Hz', 'pooled_all',
                   '_contribution_summaries')

TEAL, TEAL_LAD = PAL['lfp'], PAL['lad']['lfp']  # house hue: the table heatmaps
# Tracked colors: one color per region and per state, identical in every figure (chosen by the user).
# States follow the SfN 2025 figures (code/conferences/sfn2025/timelock_spectra.py) with states 1 and 2
# swapped; state 3 had no color there and is #262262 (user). Significance is light teal (user).
REGION_COL = [hx('#7D9988'), hx('#521E21'), hx('#EDB0D0')]                  # V1 periphery, V1 fovea, V4
STATE_COL = [hx('#83336E'), hx('#B35919'),                                  # 0 inattentive plum, 1 orange,
             hx('#448A87'), hx('#262262')]                                  # 2 attentive teal, 3 dark indigo
# (0-2 are matte versions of the SfN colors; 0 a darker plum-pink, 2 lightened just enough that
#  0 and 2, which share Figure 6, stay apart for red-green colorblind readers)
SIG_COL = TEAL                                # significance: light teal band (band_tint) and teal bar
STATE_NAME = {0: 'Inattentive', 2: 'Attentive'}   # named from their behavior (Figure 5)
REGION_NAMES = ['V1 periphery', 'V1 fovea', 'V4']

# Encoding-model families by kind. Visual: what sets the input to the receptive field -- the
# stimulus appearing, the eyes moving and what the receptive field holds. Non-visual: everything
# else, the attentional state, the animal's report and the task structure.
VISUAL = ['stim_onset', 'saccade_onset', 'saccade_offset',
          'target_in_RF_onset', 'target_in_RF_offset', 'distractor_in_RF_onset', 'distractor_in_RF_offset',
          'grass_in_RF_onset', 'grass_in_RF_offset', 'sky_in_RF_onset', 'sky_in_RF_offset',
          'mountain_in_RF_onset', 'mountain_in_RF_offset']
NONVISUAL = ['state', 'state_0', 'state_2', 'reaction_onset', 'reward_onset', 'block_onset', 'trial_onset',
             'diff_easy', 'correct', 'movement_left']
GROUPS = [('Visual', VISUAL), ('Non-visual', NONVISUAL), ('Control', ['dummy'])]


def state_label(s, short=False):
    """'State 0' -> 'Inattentive (0)'; rare states keep their number."""
    s = int(s)
    if s in STATE_NAME:
        return f'{STATE_NAME[s]} ({s})' if short else f'{STATE_NAME[s]} (state {s})'
    return f'{s}' if short else f'State {s}'
MAP = plt.get_cmap('magma')
HEAT_ALPHA = 0.45                             # table heatmaps: white -> teal at this opacity (opaque blend)
MAP_SOFT = ListedColormap([tint(TEAL, 1 - HEAT_ALPHA * t) for t in np.linspace(0, 1, 256)])
# Signed table heatmaps: light pink below zero, white at zero, teal above, at the same opacity.
MAP_DIV = ListedColormap([tint(PAL['rf'], 1 - HEAT_ALPHA * t) for t in np.linspace(1, 0, 128)] +
                         [tint(TEAL, 1 - HEAT_ALPHA * t) for t in np.linspace(0, 1, 128)])
LW_DENSE = 0.9                                # data lines in panels narrower than 2.5 cm


def region_of_array(a):
    return 1 if a <= 3 else 2 if a == 4 else 3


def array_col(a):
    return REGION_COL[region_of_array(a) - 1]


def swatches(ax, cols, axis='y', size_pt=4.0, gap_pt=2.0):
    """A small color key beside each row (axis='y') or under each column (axis='x') of a heatmap,
    so a region or state carries its tracked color into the table. None skips an entry."""
    fig = ax.figure
    pos = ax.get_position()
    w_in, h_in = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()
    off = (size_pt + gap_pt) / 72
    for k, c in enumerate(cols):
        if c is None:
            continue
        if axis == 'y':
            tr = blended_transform_factory(ax.transAxes, ax.transData) + ScaledTranslation(-off, 0, fig.dpi_scale_trans)
            r = Rectangle((0, k - 0.36), size_pt / 72 / w_in, 0.72, transform=tr, fc=c, ec='none', clip_on=False)
        else:
            tr = blended_transform_factory(ax.transData, ax.transAxes) + ScaledTranslation(0, -off, fig.dpi_scale_trans)
            r = Rectangle((k - 0.36, 0), 0.72, size_pt / 72 / h_in, transform=tr, fc=c, ec='none', clip_on=False)
        ax.add_patch(r)
    ax.tick_params(axis=axis, pad=2.5 + size_pt + gap_pt)


def letter_near(page, ax, txt, up=0.35, left=0.3):
    x, y, w, h = page.cm_of(ax)
    page.letter_at(x, y + h, txt, up=up, left=left)


def save(page, name, subdir=''):
    page.save(name, subdir)


def xtick_minus(ax, axis='both'):
    """Tick labels with a true minus and no trailing zeros."""
    fmt = plt.FuncFormatter(lambda v, _: f'{v:g}'.replace('-', MINUS))
    if axis in ('x', 'both'):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ('y', 'both'):
        ax.yaxis.set_major_formatter(fmt)


def spans(ax, fills, yl, col, bar=True):
    """Significance shading read from a plot: each filled span -> band (and bar) in col."""
    h = 0.03 * (yl[1] - yl[0])
    for f in fills:
        x0, x1 = f.xy[:, 0].min(), f.xy[:, 0].max()
        ax.add_patch(Rectangle((x0, yl[0]), x1 - x0, yl[1] - yl[0], fc=band_tint(col), ec='none', lw=0, zorder=0))
        if bar:
            ax.add_patch(Rectangle((x0, yl[1] - h), x1 - x0, h, fc=col, ec='none', lw=0, zorder=3))


def mask_spans(ax, x, mask, yl, col):
    e = edges(x)
    h = 0.03 * (yl[1] - yl[0])
    for s, t in runs(mask):
        ax.add_patch(Rectangle((e[s], yl[0]), e[t + 1] - e[s], yl[1] - yl[0], fc=band_tint(col), ec='none', lw=0, zorder=0))
        ax.add_patch(Rectangle((e[s], yl[1] - h), e[t + 1] - e[s], h, fc=col, ec='none', lw=0, zorder=3))


# ==========================================================================================
# heatmaps read from a plot: row labels, column labels, values, significance
# ==========================================================================================
def read_heat(pdf):
    ax = read_pdf(pdf)[0]
    rows = sorted(ax.ycats, key=lambda t: -t[0])
    cols = sorted(ax.xcats or [(px, f'{v:g}') for px, v in ax.xticks], key=lambda t: t[0])
    Z = np.full((len(rows), len(cols)), np.nan)
    SIG = np.zeros(Z.shape, bool)
    rpx = np.array([r[0] for r in rows])
    cpx = np.array([c[0] for c in cols])
    for x, y, s in ax.page_texts:
        m = re.match(r'^\s*([-−]?\d+\.?\d*)(\*?)\s*$', s)
        if not m:
            continue
        i, j = np.argmin(np.abs(rpx - y)), np.argmin(np.abs(cpx - x))
        Z[i, j] = float(m.group(1).replace('−', '-'))
        SIG[i, j] = m.group(2) == '*'
    assert np.isfinite(Z).all(), f'{pdf}: not every cell has a value'
    return [r[1] for r in rows], [c[1] for c in cols], Z, SIG


def draw_heat(ax, Z, SIG, norm, row_labels, col_labels, show_x=True):
    ny, nx = Z.shape
    x, y = np.arange(nx), np.arange(ny)
    heatmap(ax, x, y, Z, MAP_DIV, norm)
    sig_outline(ax, x, y, SIG, col=PAL['ink'], lw=1.0)
    for i in range(ny):
        for j in range(nx):
            ax.text(j, i, f'{Z[i, j]:.1f}'.replace('-', MINUS), ha='center', va='center',
                    color=PAL['ink'],
                    fontweight='bold' if SIG[i, j] else 'normal')
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(ny - 0.5, -0.5)
    ax.set_yticks(y, row_labels)
    ax.set_xticks(x, col_labels if show_x else [''] * nx)
    boxed(ax)


# ==========================================================================================
# Figure 5 -- behavior of the states
# ==========================================================================================
def fig05():
    rt = read_pdf(os.path.join(SA, 'rt_by_state_selected.pdf'))[0]
    bodies = sorted([f for f in rt.fills if len(f.xy) > 10], key=lambda f: f.xy[:, 0].mean())
    meds = sorted([f for f in rt.fills if len(f.xy) <= 2], key=lambda f: f.xy[:, 0].mean())
    whisk = sorted([l for l in rt.lines if abs(l.width - 1.875) < 0.01], key=lambda l: l.xy[0, 0])
    iqr = sorted([l for l in rt.lines if abs(l.width - 5.625) < 0.01], key=lambda l: l.xy[0, 0])
    pw = pd.read_csv(os.path.join(SA, 'rt_pairwise_comparisons.csv'))

    heats = {k: read_heat(os.path.join(SA, f + '.pdf')) for k, f in [
        ('b', 'trial_outcome_vs_states_zscores'), ('c', 'trial_outcome_difficulty_vs_states_zscores'),
        ('d', 'state_prob_block_position_zscore'), ('e', 'state_prob_block_transitions_zscore')]}
    # each table keeps its own scale, centred on zero, so a panel with small residuals is not flattened
    zmax = {k: np.ceil(np.abs(h[2]).max()) for k, h in heats.items()}
    norms = {k: Normalize(-z, z) for k, z in zmax.items()}

    def heat_bar(ax, k):
        colorbar_v(page, ax, MAP_DIV, [f'{MINUS}{zmax[k]:g}', '0', f'{zmax[k]:g}'], 'Adjusted residual')

    H = 15.4
    page = Page(FIG_W, H)

    # a: reaction time
    ax = page.axes(1.3, 8.45, 5.6, 6.2)
    for s, body in enumerate(bodies):
        ax.add_patch(Polygon(body.xy, closed=True, fc=STATE_COL[s], ec='none', zorder=2))
    for l in whisk:
        ax.plot(l.xy[:, 0], l.xy[:, 1], '-', color=PAL['ink'], lw=LW_GUIDE, zorder=3)
    for l in iqr:
        ax.plot(l.xy[:, 0], l.xy[:, 1], '-', color=PAL['ink'], lw=3.0, solid_capstyle='butt', zorder=3)
    for m in meds:
        ax.plot(m.xy[:, 0].mean(), m.xy[:, 1].mean(), 'o', ms=2.2, mfc='white', mec='none', zorder=4)
    top = max(b.xy[:, 1].max() for b in bodies)
    level = top + 0.25
    for _, r in pw.sort_values(['State1', 'State2'], key=lambda c: c).iterrows():
        a_, b_ = int(r['State1']), int(r['State2'])
        bracket(ax, a_, b_, level, 0.08, stars(float(r['p-corrected'])))
        level += 0.34
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(min(b.xy[:, 1].min() for b in bodies) - 0.1, level + 0.1)
    ax.set_xticks(range(4), [f'{s}\n{STATE_NAME[s].lower()}' if s in STATE_NAME else f'{s}\n' for s in range(4)])
    ax.set_xlabel('State')
    ax.set_ylabel('Reaction time (s)')
    xtick_minus(ax, 'y')
    page.letter(ax, 'a', up=0.2)

    # b, c: outcome; d, e: position in the block and around a block change
    rl, cl, Z, SIG = heats['b']
    ax_b = page.axes(10.0, 12.35, 3.8, 2.2)
    draw_heat(ax_b, Z, SIG, norms['b'], rl, cl, show_x=False)
    heat_bar(ax_b, 'b')
    page.letter(ax_b, 'b', up=0.2)

    rl, cl, Z, SIG = heats['c']
    rl = [re.sub(r'^(\w+) \| (\w+)$', lambda m: f'{m.group(1).capitalize()}, {m.group(2)}', r) for r in rl]
    rl = [r.replace('Wrong', 'Incorrect').replace('Exit', 'Exit') for r in rl]
    ax_c = page.axes(10.0, 8.45, 3.8, 3.3)
    draw_heat(ax_c, Z, SIG, norms['c'], rl, [c.split('.')[0] for c in cl])
    swatches(ax_c, [STATE_COL[int(float(c))] for c in cl], axis='x')
    ax_c.set_xlabel('State')
    heat_bar(ax_c, 'c')
    page.letter(ax_c, 'c', up=0.2)

    rl, cl, Z, SIG = heats['d']
    ax = page.axes(2.75, 4.65, 11.05, 2.3)
    swatches(ax, [STATE_COL[int(r.split()[-1])] for r in rl])
    draw_heat(ax, Z, SIG, norms['d'], [state_label(r.split()[-1], short=True) for r in rl],
              [c.replace('-', '–').replace('%', '') for c in cl])
    ax.set_xlabel('Position within the block (%)')
    heat_bar(ax, 'd')
    page.letter(ax, 'd', up=0.2)

    rl, cl, Z, SIG = heats['e']
    ax = page.axes(2.75, 1.0, 11.05, 2.3)
    swatches(ax, [STATE_COL[int(r.split()[-1])] for r in rl])
    draw_heat(ax, Z, SIG, norms['e'], [state_label(r.split()[-1], short=True) for r in rl], [c.replace('-', MINUS) for c in cl])
    ax.set_xlabel('Trial relative to the block change')
    heat_bar(ax, 'e')
    page.letter(ax, 'e', up=0.2)

    for k, (rl, cl, Z, SIG) in heats.items():
        print(f'  {k}: {Z.shape[0]}x{Z.shape[1]} cells, |z| max {np.abs(Z).max():.1f}, {SIG.sum()} significant')
    save(page, 'fig05_states_behavior')


# ==========================================================================================
# Figure 6 -- event-locked LFP in the inattentive and attentive states, per array
# ==========================================================================================
ERP_ROWS = [('stimulus', 'Stimulus onset', (-0.2, 0.9), [0, 0.4, 0.8]),
            ('reaction', 'Reaction', (-0.45, 0.45), [-0.4, 0, 0.4]),
            ('saccade_onset', 'Saccade onset', (-0.2, 0.3), [-0.2, 0, 0.2]),
            ('saccade_offset', 'Saccade offset', (-0.2, 0.3), [-0.2, 0, 0.2])]

def state_erp_data():
    """{event: [one dict per array: x, m0 (state 0), m2 (state 2), sig]}, from export_state_erp.py."""
    path = os.path.join(DATA_DIR, 'state_erp.npz')
    if not os.path.exists(path):
        raise SystemExit(f'{path} missing: run code/paper/export_state_erp.py first')
    d = np.load(path, allow_pickle=True)
    data = {}
    for ev, *_ in ERP_ROWS:
        tr, x = d[f'{ev}_traces'].astype(float), d[f'{ev}_x'].astype(float)     # session x state x array x time
        data[ev] = [dict(x=x, m0=np.nanmean(tr[:, 0, a], axis=0), m2=np.nanmean(tr[:, 1, a], axis=0),
                         sig=d[f'{ev}_sig'][a].astype(bool)) for a in range(6)]
    return data


def fig06():
    data = state_erp_data()
    c0, c2 = STATE_COL[0], STATE_COL[2]
    L, R, HG, VG = 1.75, 0.1, 0.22, 1.2
    w = (FIG_W - L - R - 5 * HG) / 6
    h, bottom, top = 2.0, 0.95, 1.0
    H = bottom + 4 * h + 3 * VG + top
    page = Page(FIG_W, H)
    for r, (ev, name, xl, xticks) in enumerate(ERP_ROWS):
        y0 = bottom + (3 - r) * (h + VG)
        rows = data.get(ev)
        if rows is not None:
            yl = headroom(pad_lim(np.concatenate([np.r_[d['m0'], d['m2']] for d in rows])), 0.06)
        for a in range(6):
            ax = page.axes(L + a * (w + HG), y0, w, h)
            if r == 0:
                ax.set_title(f'Array {a + 1} · {REGION_SHORT[region_of_array(a + 1) - 1]}')
            if a == 0:
                page.letter(ax, LETTERS[r], up=0.2 if r else 0.55)
            if rows is None:
                ax.set_axis_off()
                if a == 0:
                    ax.set_ylabel(f'{name}\nLFP ({MU}V)')
                continue
            d = rows[a]
            mask_spans(ax, d['x'], d['sig'], yl, SIG_COL)
            ax.plot(xl, [0, 0], '-', color=PAL['grey_lt'], lw=LW_GUIDE, zorder=1)
            ax.plot([0, 0], yl, '-', color=PAL['grey_lt'], lw=LW_GUIDE, zorder=1)
            ax.plot(d['x'], d['m2'], '-', color=c2, lw=LW_DENSE, zorder=2)
            ax.plot(d['x'], d['m0'], '-', color=c0, lw=LW_DENSE, zorder=3)
            ax.set_xlim(xl)
            ax.set_ylim(yl)
            ax.set_xticks(xticks)
            xtick_minus(ax)
            if a == 0:
                ax.set_ylabel(f'{name}\nLFP ({MU}V)')
            else:
                ax.set_yticklabels([])
        if rows is None:
            page.text_cm(FIG_W / 2, y0 + h / 2, 'Not exported yet: run code/paper/export_state_erp.py',
                         ha='center', va='center', color=PAL['grey'])
        else:
            xlab = 'Time from reaction time (s)' if ev == 'reaction' else f'Time from {name.lower()} (s)'
            page.text_cm(L + (FIG_W - L - R) / 2, y0 - 0.72, xlab, ha='center', va='top')
        # what the text quotes: significant spans and the difference inside them
        if rows is not None:
            out = []
            for a, d in enumerate(rows):
                e = edges(d['x'])
                spans_s = ', '.join(f'{e[s]:+.3f}..{e[t + 1]:+.3f} (0−2 {np.mean((d["m0"] - d["m2"])[s:t + 1]):+.1f})'
                                    for s, t in runs(d['sig']))
                out.append(f'    array {a + 1}: {spans_s or "-"}')
            print(f'  {ev}:\n' + '\n'.join(out))
        else:
            print(f'  {ev}: not exported yet')
    page.fig.legend([line_handle(c0), line_handle(c2)], [state_label(0), state_label(2)], ncol=2, loc='center',
                    bbox_to_anchor=(0.5, (H - 0.25) / H), bbox_transform=page.fig.transFigure)
    save(page, 'fig06_state_erp')


# ==========================================================================================
# Figure 7 -- mean kernels by region, visual and non-visual families
# ==========================================================================================
def fam_style(fam):
    extra = {'state_0': ('Inattentive (state 0)', 'beh'), 'state_2': ('Attentive (state 2)', 'beh'),
             'correct': ('Correct', 'beh'), 'movement_left': ('Left choice', 'beh'),
             'reaction_onset': ('Reaction time', 'beh'), 'state': ('Attentional state', 'beh'),
             'session': ('Session offsets', 'null')}
    name, g = extra[fam] if fam in extra else family_style(fam)[:1] + (None,)
    if fam in ('dummy', 'session'):
        return name, PAL['grey'], PAL['lad']['null']
    return name, TEAL, TEAL_LAD          # one hue for every family; the group header names its kind


def group_header(page, x0, x1, y, text):
    """Group name over a block of panels, with a hairline to its right."""
    page.text_cm(x0, y, text, ha='left', va='center', color=LETTER_COLOR)
    tw = 0.14 * len(text) + 0.3
    page.fig.add_artist(Line2D([(x0 + tw) / page.w, x1 / page.w], [y / page.h] * 2,
                               color=PAL['grey_lt'], lw=LW_GUIDE))




def fig07():
    K = load('fig07_encoding_model')
    labels = [str(v) for v in np.atleast_1d(K['k_labels'])]
    src = {}
    for a in read_pdf(os.path.join(LEM, 'pooled_all_mean_kernels_permtest_norm.pdf')):
        m = re.search(r'(\w+)\s+\(n=(\d+)\)\s+p=([\d.]+)', a.title)
        if m:
            src[m.group(1)] = (a, int(m.group(2)), float(m.group(3)))
    order = ['stim_onset', 'saccade_onset', 'saccade_offset', 'target_in_RF_onset', 'target_in_RF_offset',
             'distractor_in_RF_onset', 'distractor_in_RF_offset',
             'state_0', 'state_2', 'reaction_onset', 'reward_onset', 'block_onset', 'trial_onset',
             'diff_easy', 'correct', 'movement_left']
    groups = [(g, [f for f in order if f in members and f in src and f in labels]) for g, members in GROUPS[:2]]
    left_out = [f for f in labels if not any(f in fs for _, fs in groups)]
    print(f'  kernels drawn: {sum(len(fs) for _, fs in groups)}; not drawn (no pooled test): {", ".join(left_out)}')

    nc = 5
    L, R, HG, VG = 1.45, 0.15, 0.4, 1.35
    w = (FIG_W - L - R - (nc - 1) * HG) / nc
    h, head, bottom, top = 1.85, 1.35, 0.95, 0.85
    nrows = [int(np.ceil(len(fs) / nc)) for _, fs in groups]
    H = bottom + top + sum(n * h + (n - 1) * VG + head for n in nrows) + (len(groups) - 1) * VG
    page = Page(FIG_W, H)
    y_top = H - top
    k = 0
    for (gname, fams), nr in zip(groups, nrows):
        group_header(page, L - 0.95, FIG_W - R, y_top - 0.2, f'{gname} factors')
        y_top -= head
        for j, fam in enumerate(fams):
            i = labels.index(fam)
            x = np.asarray(K['k_x'][i], float)
            mu = np.asarray(K['k_mean'][i], float)          # all 66 channels: the line the test is about
            a_src, n, p = src[fam]
            name, col, lad = fam_style(fam)
            yl = [-1.15, 1.3]
            r, c = divmod(j, nc)
            ax = page.axes(L + c * (w + HG), y_top - (r + 1) * h - r * VG, w, h)
            sig = [f for f in a_src.fills if f.color[0] > 0.5 and f.color[1] > 0.8]
            spans(ax, sig, yl, SIG_COL)
            ax.plot([x[0], x[-1]], [0, 0], '-', color=PAL['grey_lt'], lw=LW_GUIDE, zorder=1)
            ax.plot([0, 0], yl, '-', color=PAL['grey_lt'], lw=LW_GUIDE, zorder=1)
            ax.plot(x, mu, '-', color=PAL['ink'], lw=1.2, zorder=3)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(yl)
            xtick_minus(ax)
            ax.set_title(name)
            ax.set_yticks([-1, 0, 1])
            if c == 0:
                ax.set_ylabel('Kernel (normalized)')
            else:
                ax.set_yticklabels([])
            if r == nr - 1 or j + nc >= len(fams):
                ax.set_xlabel('Time from event (s)')
            letter_near(page, ax, LETTERS[k], up=0.62, left=0.95 if c == 0 else 0.3)
            k += 1
        y_top -= nr * h + (nr - 1) * VG + VG
    save(page, 'fig07_kernels')


# ==========================================================================================
# Figure 8 -- unique contributions: per array, and by region, visual and non-visual families
# ==========================================================================================
def contributions():
    S = load('fig08_contributions')
    fam = [str(v) for v in np.atleast_1d(S['fam'])]
    st = pd.read_csv(os.path.join(LEM, 'region_dR2_stats.csv')).set_index('family')
    return S, fam, st


def grouped_order(fam):
    """Family indices, visual first, then non-visual, then the control; overall rank kept within a group."""
    return [(g, [fam.index(f) for f in fam if f in members]) for g, members in GROUPS]


def fig08():
    S, fam, st = contributions()
    groups = grouped_order(fam)
    missing = set(fam) - {fam[j] for _, js in groups for j in js}
    assert not missing, f'families without a group: {missing}'
    order = [j for _, js in groups for j in js]

    A = S['arr_mean'].astype(float)
    ov = read_pdf(os.path.join(LEM, 'overview_mean_dR2_by_array.pdf'))[0]
    cpx = {lab.split(' (')[0]: px for px, lab in ov.xcats}
    rpx = {int(lab.split()[-1]): px for px, lab in ov.ycats}
    dots = np.zeros(A.shape, bool)
    for x, y, nm, _ in ov.page_markers:
        if not nm.startswith('M'):
            continue
        cfam = min(cpx, key=lambda f: abs(cpx[f] - x))
        arr = min(rpx, key=lambda a: abs(rpx[a] - y))
        if cfam in fam:
            dots[arr - 1, fam.index(cfam)] = True

    dR = S['dR2'].astype(float)
    reg = S['region'].astype(int)
    rel = 100 * dR / S['full_R2'].astype(float)[:, None]
    nc = 6
    L, R, HG, VG = 1.45, 0.1, 0.42, 1.0
    w = (FIG_W - L - R - (nc - 1) * HG) / nc
    h, head = 1.4, 0.7
    bar_groups = [('Visual factors', groups[0][1]), ('Non-visual factors and control', groups[1][1] + groups[2][1])]
    nrows = [int(np.ceil(len(js) / nc)) for _, js in bar_groups]
    heat_block = 6.2
    H = 0.45 + heat_block + sum(n * h + n * VG + head for n in nrows) + 0.2
    page = Page(FIG_W, H)

    # a: mean contribution per array, columns grouped
    ax = page.axes(2.2, H - 3.25, FIG_W - 2.2 - 1.85, 2.35)
    Ao = A[:, order]
    lo, hi = np.nanmin(A), np.nanquantile(A, 0.99)
    norm = TwoSlopeNorm(0, lo, hi) if lo < 0 else Normalize(0, hi, clip=True)
    xs, ys = np.arange(len(order)), np.arange(6)
    heatmap(ax, xs, ys, Ao, MAP_SOFT, norm)
    for i, jj in zip(*np.where(dots[:, order])):
        ax.plot(jj, i, 'o', ms=2.2, mfc=PAL['ink'], mec='none')
    edge = 0
    for g, js in groups:
        if edge:
            ax.plot([edge - 0.5] * 2, [-0.5, 5.5], '-', color=AXIS_COLOR, lw=1.0, zorder=5)
        ax.text(edge + len(js) / 2 - 0.5, -0.75, g, ha='center', va='bottom', color=LETTER_COLOR)
        edge += len(js)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(5.5, -0.5)
    ax.set_yticks(ys, [f'Array {a} · {REGION_SHORT[region_of_array(a) - 1]}' for a in range(1, 7)])
    swatches(ax, [array_col(a) for a in range(1, 7)])
    ax.set_xticks(xs, [fam_style(fam[j])[0] for j in order], rotation=90)
    for t, j in zip(ax.get_xticklabels(), order):
        t.set_color(PAL['grey'] if fam[j] == 'dummy' else LETTER_COLOR)
    boxed(ax)
    colorbar_v(page, ax, MAP_SOFT, [sci(lo), '0', sci(hi)], f'Mean {DR2}')
    page.letter(ax, 'a', up=0.55)

    # b: relative contribution by region
    rng = np.random.default_rng(2)
    y_top = H - 0.45 - heat_block
    first = True
    for (gname, js), nr in zip(bar_groups, nrows):
        group_header(page, L - 0.95, FIG_W - R, y_top - 0.15, gname)
        if first:
            page.letter_at(L, y_top + 0.35, 'b', up=0.0, left=0.2)   # same x as a (2.2 - 0.95)
            first = False
        y_top -= head
        for k, j in enumerate(js):
            f = fam[j]
            name, col, lad = fam_style(f)
            v = rel[:, j]
            r, c = divmod(k, nc)
            ax = page.axes(L + c * (w + HG), y_top - (r + 1) * h - r * VG - 0.35, w, h)
            yl0 = pad_lim(v)
            yl = headroom(yl0, 0.42)
            ax.plot([0.4, 3.6], [0, 0], '-', color=PAL['grey_lt'], lw=LW_GUIDE, zorder=1)
            for g in (1, 2, 3):
                vr = v[reg == g]
                ax.add_patch(Rectangle((g - 0.32, 0), 0.64, vr.mean(), fc=tint(REGION_COL[g - 1], 0.35),
                                       ec='none', zorder=1.5))
                ax.plot(g + (rng.random(vr.size) - 0.5) * 0.42, vr, 'o', ls='none', ms=1.3,
                        mfc=REGION_COL[g - 1], mec='none', zorder=2)
            span = yl0[1] - yl0[0]
            q = lambda key: float(st.loc[f, key]) if f in st.index else np.nan
            bracket(ax, 1, 2, yl0[1] + 0.06 * span, 0.03 * span, stars(q('q_rel[V1-fovea vs V1-periph]')))
            bracket(ax, 1.5, 3, yl0[1] + 0.28 * span, 0.03 * span, stars(q('q_rel[V4 vs V1 (all)]')))
            ax.set_xlim(0.4, 3.6)
            ax.set_ylim(yl)
            last = r == nr - 1 or k + nc >= len(js)
            ax.set_xticks([1, 2, 3], REGION_SHORT if last else ['', '', ''])
            xtick_minus(ax, 'y')
            kw = stars(q('q_rel[Kruskal all regions]'))
            ax.set_title(name + (f'\nKW {kw}' if kw else '\n'), color=LETTER_COLOR if f != 'dummy' else PAL['grey'],
                         linespacing=1.0)
            if c == 0:
                ax.set_ylabel(f'{DR2} / {R2LAB} (%)')
        y_top -= nr * h + nr * VG
    print(f'  {len(fam)} families: ' + '; '.join(f'{g} {len(js)}' for g, js in groups) +
          f'; array-level dots {int(dots.sum())} (state on {int(dots[:, fam.index("state")].sum())} of 6 arrays)')
    save(page, 'fig08_contributions')


# ==========================================================================================
# Figures 1 to 4 -- the Illustrator exports in writing/figures/art, cropped to their ink
# ==========================================================================================
ART = {'fig01_task': 'setup', 'fig02_recordings': 'neural_rec_method', 'fig03_rf_mapping': 'VR_RF_mapping_method',
       'fig04_states_method': 'states_method'}
ART_TEX = r"""\documentclass{article}
\usepackage[paperwidth=%(W).2fbp,paperheight=%(H).2fbp,margin=0pt]{geometry}
\usepackage{graphicx}
\pagestyle{empty}
\setlength{\topskip}{0pt}\setlength{\parindent}{0pt}\setlength{\parskip}{0pt}
\begin{document}
\vbox to 0pt{\includegraphics[trim=%(trim)s,clip,width=%(W).2fbp,height=%(H).2fbp]{art.pdf}\vss}
\end{document}
"""


def art():
    """Crop each art page to its ink (+2 bp) and shrink it to the text width if wider. The text
    in the art shrinks with it: draw the art at 16.2 cm to keep 8 pt labels."""
    tw = FIG_W / 2.54 * 72
    for name, src in ART.items():
        pdf = os.path.join(ART_DIR, src + '.pdf')
        info = subprocess.run(['pdfinfo', pdf], capture_output=True, text=True).stdout
        pw, ph = map(float, re.search(r'Page size:\s+([\d.]+) x ([\d.]+)', info).groups())
        gs = subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-sDEVICE=bbox', pdf], capture_output=True, text=True)
        x0, y0, x1, y1 = map(float, re.search(r'HiResBoundingBox:\s+(\S+) (\S+) (\S+) (\S+)', gs.stderr).groups())
        x0, y0, x1, y1 = max(x0 - 2, 0), max(y0 - 2, 0), min(x1 + 2, pw), min(y1 + 2, ph)
        s = min(1.0, tw / (x1 - x0))
        W, H = (x1 - x0) * s, (y1 - y0) * s
        trim = f'{x0:.2f}bp {y0:.2f}bp {pw - x1:.2f}bp {ph - y1:.2f}bp'
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copyfile(pdf, os.path.join(tmp, 'art.pdf'))
            open(os.path.join(tmp, 'p.tex'), 'w').write(ART_TEX % dict(W=W, H=H, trim=trim))
            r = subprocess.run(['lualatex', '-interaction=batchmode', 'p.tex'], cwd=tmp, capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f'lualatex failed for {name}')
            pages = int(re.search(r'Pages:\s+(\d+)', subprocess.run(['pdfinfo', os.path.join(tmp, 'p.pdf')],
                                                                    capture_output=True, text=True).stdout).group(1))
            if pages != 1:      # the art spilled onto a second page and the first would be blank
                raise SystemExit(f'{name}: cropped page has {pages} pages')
            shutil.copyfile(os.path.join(tmp, 'p.pdf'), os.path.join(OUT_DIR, name + '.pdf'))
        print(f'  {name}: {src}.pdf cropped to {W / 72 * 2.54:.1f} x {H / 72 * 2.54:.1f} cm (scale {s:.2f})')


STEPS = {'art': art, 'fig05': fig05, 'fig06': fig06, 'fig07': fig07, 'fig08': fig08}

if __name__ == '__main__':
    setup_style()
    for key in sys.argv[1:] or list(STEPS):
        if key not in STEPS:
            raise SystemExit(f'unknown figure {key}; choose from {", ".join(STEPS)}')
        print(f'\n===== {key} =====')
        STEPS[key]()
