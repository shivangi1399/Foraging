#!/usr/bin/env python
"""
pdfplot.py -- read the plotted values back out of a matplotlib PDF in plots/

Several analysis plots were saved only as PDFs, with no data file next to them. A matplotlib PDF
keeps every curve, filled area, marker and tick label as vector drawing commands in page
coordinates, so the values that were plotted can be recovered exactly (to the precision of
matplotlib's path writer) and redrawn in the house style. Nothing is recomputed.

    from pdfplot import read_pdf
    axes = read_pdf('plots/.../some_plot.pdf')
    ax = axes[0]
    for line in ax.lines:          # stroked paths inside the axes, in data coordinates
        line.color, line.xy
    for patch in ax.fills:         # filled paths inside the axes (bands, spans, violins, cells)
        patch.color, patch.xy
    ax.texts                       # every text drawn over the axes box: (x, y, string) in data units

Only the stdlib and numpy: the PDF objects are found with a regex, content streams inflated
with zlib (matplotlib writes PDF 1.4 without object streams). Axes are the clip rectangles
matplotlib sets around its data artists; tick marks on an axes edge, paired with the label
drawn right after them, calibrate pixel -> data for each axis.
"""
import re
import zlib
from dataclasses import dataclass, field

import numpy as np

TOKEN = re.compile(rb'\((?:\\.|[^\\)])*\)|\[|\]|/[^\s/\[\]()<>]+|[^\s\[\]()<>/]+')
NUM = re.compile(rb'^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$')
EPS = 0.05


@dataclass
class Path:
    color: tuple
    xy: np.ndarray            # page units first, data units after calibration
    width: float = 1.0
    alpha: float = 1.0
    closed: bool = False


@dataclass
class Axes:
    box: tuple                # x0, y0, x1, y1 in page units
    lines: list = field(default_factory=list)
    fills: list = field(default_factory=list)
    markers: list = field(default_factory=list)   # (x, y, name) in data units
    texts: list = field(default_factory=list)     # (x, y, string) in data units
    page_texts: list = field(default_factory=list)  # the same texts in page units
    xticks: list = field(default_factory=list)    # (px, value) of numeric tick labels
    yticks: list = field(default_factory=list)
    xcats: list = field(default_factory=list)     # (px, label) of text tick labels
    ycats: list = field(default_factory=list)
    xmap: tuple = None        # data = a * px + b
    ymap: tuple = None
    title: str = ''
    page_markers: list = field(default_factory=list)

    def to_data(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        out = xy.copy()
        if self.xmap:
            out[:, 0] = self.xmap[0] * xy[:, 0] + self.xmap[1]
        if self.ymap:
            out[:, 1] = self.ymap[0] * xy[:, 1] + self.ymap[1]
        return out

    @property
    def xlim(self):
        return tuple(self.to_data([[self.box[0], self.box[1]], [self.box[2], self.box[3]]])[:, 0])

    @property
    def ylim(self):
        return tuple(self.to_data([[self.box[0], self.box[1]], [self.box[2], self.box[3]]])[:, 1])


def _objects(raw):
    return {int(m.group(1)): m.group(2) for m in re.finditer(rb'(\d+) 0 obj(.*?)endobj', raw, re.S)}


def _stream(body):
    head, rest = body.split(b'stream', 1)
    rest = rest[1:] if rest[:1] == b'\n' else rest[2:] if rest[:2] == b'\r\n' else rest
    data = rest[:rest.rindex(b'endstream')]
    return zlib.decompress(data) if b'FlateDecode' in head else data


def _alphas(objs):
    """ExtGState name -> fill alpha."""
    out = {}
    for body in objs.values():
        for name, d in re.findall(rb'/(A\d+)\s*<<(.*?)>>', body, re.S):
            m = re.search(rb'/ca\s+([\d.]+)', d)
            out[name.decode()] = float(m.group(1)) if m else 1.0
    return out


def _page_content(raw):
    objs = _objects(raw)
    page = next(b for b in objs.values() if re.search(rb'/Type\s*/Page\b', b))
    ref = int(re.search(rb'/Contents\s+(\d+)\s+0\s+R', page).group(1))
    return _stream(objs[ref]), _alphas(objs)


def _string(tok):
    s = tok[1:-1]
    s = re.sub(rb'\\([()\\])', rb'\1', s)
    return s.decode('latin1').replace('\x00', '')


# glyphs that matplotlib draws as XObjects instead of characters of the Type 3 font
GLYPH = {'minus': '−', 'Delta': 'Δ', 'plusminus': '±', 'uni0394': 'Δ', 'twosuperior': '²',
         'mu': 'µ', 'uni00B2': '²', 'degree': '°'}


def _events(content, alphas):
    """Walk the content stream; yield drawing events in page units."""
    toks = TOKEN.findall(content)
    stack = []
    st = dict(RG=(0, 0, 0), rg=(0, 0, 0), w=1.0, ctm=np.eye(3), clip=None, alpha=1.0)
    ops, path, sub = [], [], []
    text = None
    for t in toks:
        if NUM.match(t):
            ops.append(float(t))
            continue
        if t[:1] in (b'(', b'/', b'[', b']'):
            ops.append(t)
            continue
        op = t.decode('latin1')
        nums = [o for o in ops if isinstance(o, float)]
        if op == 'q':
            stack.append(dict(st, ctm=st['ctm'].copy()))
        elif op == 'Q':
            if text is not None and text['depth'] == len(stack):
                yield ('text', text)
                text = None
            st = stack.pop() if stack else st
        elif op == 'cm':
            a, b, c, d, e, f = nums[-6:]
            st['ctm'] = np.array([[a, b, 0], [c, d, 0], [e, f, 1]]) @ st['ctm']
        elif op == 'w':
            st['w'] = nums[-1]
        elif op == 'RG':
            st['RG'] = tuple(nums[-3:])
        elif op == 'rg':
            st['rg'] = tuple(nums[-3:])
        elif op == 'G':
            st['RG'] = (nums[-1],) * 3
        elif op == 'g':
            st['rg'] = (nums[-1],) * 3
        elif op == 'gs':
            st['alpha'] = alphas.get(ops[-1][1:].decode(), 1.0)
        elif op == 'm':
            if sub:
                path.append(sub)
            sub = [nums[-2:]]
        elif op == 'l':
            sub.append(nums[-2:])
        elif op == 'c':
            sub.append(nums[-2:])
        elif op == 're':
            x, y, w, h = nums[-4:]
            if sub:
                path.append(sub)
            sub = []
            path.append([[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]])
        elif op == 'h':
            if sub:
                sub.append(sub[0])
        elif op == 'W':
            if sub:
                path.append(sub)
            pts = np.array([p for s in path for p in s], float)
            pts = _apply(pts, st['ctm'])
            st['clip'] = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
            path, sub = [], []
        elif op in ('S', 'f', 'f*', 'B', 'B*', 'n', 's', 'b'):
            if sub:
                path.append(sub)
            if op != 'n' and path:
                for s in path:
                    pts = _apply(np.array(s, float), st['ctm'])
                    yield ('path', dict(op=op, xy=pts, stroke=st['RG'], fill=st['rg'], w=st['w'],
                                        clip=st['clip'], alpha=st['alpha']))
            path, sub = [], []
        elif op == 'BT':
            if text is None:
                org = _apply(np.zeros((1, 2)), st['ctm'])[0]
                text = dict(x=org[0], y=org[1], s='', depth=len(stack), minus=False, size=None)
        elif op == 'Tf':
            if text is not None:
                text['size'] = nums[-1]
        elif op in ('TJ', 'Tj'):
            if text is not None:
                text['s'] += ''.join(_string(o) for o in ops if isinstance(o, bytes) and o[:1] == b'(')
        elif op == 'Do':
            name = ops[-1][1:].decode()
            if text is not None:
                g = name.split('-')[-1]
                text['s'] = text['s'] + GLYPH.get(g, '') if g != 'minus' else text['s']
                text['minus'] |= g == 'minus'
            else:
                org = _apply(np.zeros((1, 2)), st['ctm'])[0]
                yield ('marker', dict(x=org[0], y=org[1], name=name, clip=st['clip'], fill=st['rg']))
        ops = []
    if text is not None:
        yield ('text', text)


def _apply(pts, ctm):
    pts = np.asarray(pts, float).reshape(-1, 2)
    h = np.c_[pts, np.ones(len(pts))] @ ctm
    return h[:, :2]


def _num(s, minus):
    s = s.strip().replace('−', '-')
    try:
        v = float(s)
    except ValueError:
        return None
    return -abs(v) if minus else v


def read_pdf(pdf):
    raw = open(pdf, 'rb').read()
    content, alphas = _page_content(raw)
    events = list(_events(content, alphas))

    # axes = clip rectangles that data artists were drawn in
    boxes = []
    for kind, e in events:
        if kind in ('path', 'marker') and e['clip'] is not None:
            c = tuple(round(v, 2) for v in e['clip'])
            if c not in boxes:
                boxes.append(c)
    axes = [Axes(box=b) for b in boxes]

    def owner(clip):
        c = tuple(round(v, 2) for v in clip)
        return axes[boxes.index(c)]

    def edge_axes(x, y, vertical):
        for ax in axes:
            x0, y0, x1, y1 = ax.box
            if vertical and abs(y - y0) < 0.6 and x0 - EPS <= x <= x1 + EPS:
                return ax
            if not vertical and abs(x - x0) < 0.6 and y0 - EPS <= y <= y1 + EPS:
                return ax
        return None

    free_texts = []
    pending = None
    for kind, e in events:
        if kind == 'path':
            xy = e['xy']
            if e['clip'] is None:
                # tick mark: a two-point stroke leaving an axes edge
                if len(xy) == 2 and e['op'] in ('B', 'S'):
                    dx, dy = abs(xy[1] - xy[0])
                    if dx < 1e-6 and 0 < dy < 12:
                        ax = edge_axes(xy[0, 0], xy[:, 1].max(), True)
                        if ax is not None:
                            pending = (ax, 'x', xy[0, 0])
                            continue
                    if dy < 1e-6 and 0 < dx < 12:
                        ax = edge_axes(xy[:, 0].max(), xy[0, 1], False)
                        if ax is not None:
                            pending = (ax, 'y', xy[0, 1])
                            continue
                pending = None
                continue
            ax = owner(e['clip'])
            closed = len(xy) > 2 and np.allclose(xy[0], xy[-1])
            if e['op'] == 'S':
                ax.lines.append(Path(e['stroke'], xy, e['w'], 1.0, closed))
            else:
                ax.fills.append(Path(e['fill'], xy, e['w'], e['alpha'], closed))
            pending = None
        elif kind == 'marker':
            if e['clip'] is not None:
                owner(e['clip']).markers.append((e['x'], e['y'], e['name'], e['fill']))
            pending = None
        elif kind == 'text':
            if pending is not None:
                ax, which, px = pending
                v = _num(e['s'], e['minus'])
                if v is not None:
                    (ax.xticks if which == 'x' else ax.yticks).append((px, v))
                else:
                    (ax.xcats if which == 'x' else ax.ycats).append((px, e['s']))
                pending = None
                continue
            free_texts.append(e)

    for ax in axes:
        for which, ticks in (('x', ax.xticks), ('y', ax.yticks)):
            if len(ticks) >= 2:
                px, val = np.array(ticks).T
                a, b = np.polyfit(px, val, 1)
                setattr(ax, which + 'map', (a, b))
    # shared axes: an axis without labels takes the map of an axes with the same extent on it
    for ax in axes:
        for other in axes:
            if ax.xmap is None and other.xmap and abs(other.box[0] - ax.box[0]) < 0.6 \
                    and abs(other.box[2] - ax.box[2]) < 0.6:
                ax.xmap = other.xmap
            if ax.ymap is None and other.ymap and abs(other.box[1] - ax.box[1]) < 0.6 \
                    and abs(other.box[3] - ax.box[3]) < 0.6:
                ax.ymap = other.ymap
    for ax in axes:
        for p in ax.lines + ax.fills:
            p.xy = ax.to_data(p.xy)
        ax.page_markers = list(ax.markers)
        ax.markers = [(*ax.to_data([[x, y]])[0], n, c) for x, y, n, c in ax.markers]

    # texts: over an axes box -> that axes (data units); just above -> its title
    for t in free_texts:
        s = ('−' if t['minus'] and not t['s'].startswith('-') else '') + t['s']
        for ax in axes:
            x0, y0, x1, y1 = ax.box
            if x0 - 1 <= t['x'] <= x1 + 1 and y0 - 1 <= t['y'] <= y1 + 1:
                ax.texts.append((*ax.to_data([[t['x'], t['y']]])[0], s))
                ax.page_texts.append((t['x'], t['y'], s))
                break
            if x0 - 20 <= t['x'] <= x1 + 20 and y1 < t['y'] <= y1 + 3 * (t['size'] or 10):
                ax.title = (ax.title + ' ' + s).strip() if ax.title else s
                break
    # rows top to bottom, then left to right
    axes.sort(key=lambda a: (-round(a.box[3]), a.box[0]))
    return axes


def grid(axes, ncols):
    """Axes list (read_pdf order) -> rows of ncols."""
    return [axes[i:i + ncols] for i in range(0, len(axes), ncols)]


if __name__ == '__main__':
    import sys
    for ax in read_pdf(sys.argv[1]):
        print(f'box {tuple(round(v) for v in ax.box)}  title {ax.title!r}  xticks {len(ax.xticks)} '
              f'yticks {len(ax.yticks)}  lines {len(ax.lines)}  fills {len(ax.fills)} '
              f'markers {len(ax.markers)}  texts {len(ax.texts)}')
