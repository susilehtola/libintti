#!/usr/bin/env python3
"""Build the libintti logo as a self-contained SVG (all text as outlines)."""
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen

DEJAVU = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
QTMIL  = "/usr/share/fonts/qualitype/QTMilitary.otf"
STIX   = "/usr/share/fonts/stix-fonts/STIXTwoMath-Regular.otf"

CREAM = "#F2EFE2"
FIELD = "#2F3B21"


def glyphs(fontfile, text, size, tracking=0.0):
    f = TTFont(fontfile)
    upem = f["head"].unitsPerEm
    cmap = f.getBestCmap()
    gs = f.getGlyphSet()
    hmtx = f["hmtx"]
    scale = size / upem
    out, pen_x = [], 0.0
    for ch in text:
        gname = cmap[ord(ch)]
        pen = SVGPathPen(gs)
        gs[gname].draw(pen)
        adv = hmtx[gname][0] * scale
        out.append(dict(ch=ch, d=pen.getCommands(), x=pen_x, adv=adv, scale=scale))
        pen_x += adv + tracking
    total = pen_x - (tracking if text else 0)
    return out, total


def emit(gl, x0, baseline, fill=CREAM, gid=None):
    s = f'  <g{f" id=\"{gid}\"" if gid else ""} fill="{fill}">\n'
    for g in gl:
        tf = (f'translate({x0 + g["x"]:.2f},{baseline:.2f}) '
              f'scale({g["scale"]:.6f},{-g["scale"]:.6f})')
        s += f'    <path transform="{tf}" d="{g["d"]}"/>\n'
    s += "  </g>\n"
    return s


# ---------------- lettering ----------------
WM_SIZE, WM_TRACK, WM_BASE = 72, 2.0, 372
wm, wm_w = glyphs(DEJAVU, "libintti", WM_SIZE, WM_TRACK)
wm_x0 = 256 - wm_w / 2

S1_SIZE, S1_TRACK = 20, 1.6
s1, s1_w = glyphs(QTMIL, "QUANTUM CHEMISTRY", S1_SIZE, S1_TRACK)
s2, s2_w = glyphs(QTMIL, "INTEGRALS LIBRARY", S1_SIZE, S1_TRACK)

INT_SIZE = 196
ig, ig_w = glyphs(STIX, "∫", INT_SIZE)

# ---------------- stencil notches ----------------
# Cut the strokes where a stencil bridge would sit: at the bowl/shoulder
# junctions of b and n, and where the crossbar of t meets its stem.
em = WM_SIZE / 1000.0 * 1000  # = size; fractions below are in em
xh = 0.55 * WM_SIZE           # x-height
notches = []
for g in wm:
    gx, adv, ch = wm_x0 + g["x"], g["adv"], g["ch"]
    if ch == "b":
        stem_r = gx + 0.20 * WM_SIZE       # right edge of the stem
        notches += [(stem_r - 3, WM_BASE - xh + 2, 7, 4.5),      # top junction
                    (stem_r - 3, WM_BASE - 7.5, 7, 4.5)]         # bottom junction
    elif ch == "n":
        # bridges where the shoulder springs from the stems
        stem_r = gx + 0.20 * WM_SIZE
        right_stem = gx + adv - 0.185 * WM_SIZE
        notches += [(stem_r - 3.5, WM_BASE - xh + 12, 7, 4.5),
                    (right_stem - 3.5, WM_BASE - xh + 12, 7, 4.5)]
    elif ch == "t":
        stem_l = gx + 0.10 * WM_SIZE
        stem_r = gx + 0.29 * WM_SIZE
        ybar = WM_BASE - 0.44 * WM_SIZE
        notches += [(stem_l - 4.0, ybar, 4.0, 8),                # left of stem
                    (stem_r, ybar, 4.0, 8)]                      # right of stem
notch_svg = '  <g id="stencil-notches" fill="%s">\n' % FIELD
for (nx, ny, nw, nh) in notches:
    notch_svg += f'    <rect x="{nx:.2f}" y="{ny:.2f}" width="{nw:.2f}" height="{nh:.2f}"/>\n'
notch_svg += "  </g>\n"

# ---------------- assemble ----------------
head = open("logo_body.svg").read()   # everything except the lettering
svg = head.replace("<!--LETTERING-->",
                   emit(wm, wm_x0, WM_BASE, gid="wordmark")
                   + notch_svg
                   + emit(s1, 256 - s1_w / 2, 400, gid="sub1")
                   + emit(s2, 256 - s2_w / 2, 424, gid="sub2"))
svg = svg.replace("<!--INTEGRAL-->", emit(ig, 176, 280, gid="integral"))
open("logo.svg", "w").write(svg)
print(f"wordmark w={wm_w:.1f} x0={wm_x0:.1f} | sub1 w={s1_w:.1f} | sub2 w={s2_w:.1f} "
      f"| integral w={ig_w:.1f} | notches={len(notches)}")
