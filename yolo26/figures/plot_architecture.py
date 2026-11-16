"""Flowchart-style figures of the proposed YOLO26n (ADown + ECA + HPConv) for the paper.

Writes, next to this script:
  architecture_proposed_yolo26n.pdf / .png  - whole network (backbone, neck, head), IEEE full width (figure*)
  modules_adown_eca_hpconv.pdf / .png       - inside of the three new modules
Shapes are for a 640 x 640 input and the n scale (as printed by Ultralytics 8.4.137).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7})

# one fill + edge per role; the module name is always written in the box, so colour is never the only cue
ROLE = {
    "plain":  ("#f2f2f1", "#8a8a8a"),
    "small":  ("#ffffff", "#9a9a9a"),
    "adown":  ("#d6e6fa", "#2a78d6"),
    "eca":    ("#fde3d3", "#d4561f"),
    "hpconv": ("#d2f1e4", "#128a5f"),
    "detect": ("#ebe8fb", "#6250d6"),
    "io":     ("#ffffff", "#222222"),
}
INK, MUTED, LINE = "#1a1a1a", "#555555", "#444444"


def box(ax, xc, yc, w, h, left, right="", role="plain", fs=6.6, bold=False):
    fc, ec = ROLE[role]
    lw = 1.1 if role in ("adown", "eca", "hpconv", "detect", "io") else 0.7
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    if right:
        ax.text(xc - w / 2 + 0.06, yc, left, ha="left", va="center", fontsize=fs, color=INK, zorder=4,
                fontweight="bold" if bold else "normal")
        ax.text(xc + w / 2 - 0.06, yc, right, ha="right", va="center", fontsize=fs - 0.6, color=MUTED, zorder=4)
    else:
        ax.text(xc, yc, left, ha="center", va="center", fontsize=fs, color=INK, zorder=4,
                fontweight="bold" if bold else "normal")


def path(ax, pts, head=True, color=LINE, lw=0.8):
    """Polyline through pts; arrowhead on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] if head else xs, ys[:-1] if head else ys, color=color, lw=lw, zorder=2,
            solid_capstyle="butt", solid_joinstyle="miter")
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=6, shrinkA=0, shrinkB=0))


def dot(ax, x, y):
    ax.plot([x], [y], "o", ms=2.2, color=LINE, zorder=5)


def section(ax, x0, y0, x1, y1, title, tint):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=tint, ec="#c9c9c9", lw=0.6, ls=(0, (3, 2)), zorder=0))
    ax.text((x0 + x1) / 2, y1 - 0.13, title, ha="center", va="center", fontsize=7.6, fontweight="bold", color=INK)


# --------------------------------------------------------------------------------------------------
def architecture():
    W, H = 7.16, 4.2
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    h = 0.19

    # sections
    section(ax, 0.06, 0.48, 1.84, 4.12, "Backbone", "#fafafa")
    section(ax, 2.10, 0.48, 5.40, 4.12, "Neck (FPN + PAN)", "#fafafa")
    section(ax, 5.56, 0.48, 7.10, 4.12, "Head (YOLO26n, unchanged)", "#fafafa")
    ax.text(2.85, 3.80, "top-down path", ha="center", fontsize=6.6, style="italic", color=MUTED)
    ax.text(4.65, 3.80, "bottom-up path", ha="center", fontsize=6.6, style="italic", color=MUTED)

    # backbone column
    xb, wb, top, dy = 0.95, 1.62, 3.72, 0.27
    rows = [
        ("Input", "640×640×3", "io"),
        ("0  Conv 3×3, s2", "320×320×16", "plain"),
        ("1  Conv 3×3, s2", "160×160×32", "plain"),
        ("2  C3k2", "160×160×64", "plain"),
        ("3  ADown ★", "80×80×64", "adown"),
        ("4  C3k2", "80×80×128", "plain"),
        ("5  ADown ★", "40×40×128", "adown"),
        ("6  C3k2 + ECA (k=3) ★", "40×40×128", "eca"),
        ("7  ADown ★", "20×20×256", "adown"),
        ("8  C3k2 + ECA (k=5) ★", "20×20×256", "eca"),
        ("9  SPPF", "20×20×256", "plain"),
        ("10  C2PSA", "20×20×256", "plain"),
    ]
    yb = [top - i * dy for i in range(len(rows))]
    for (l, r, role), y in zip(rows, yb):
        box(ax, xb, y, wb, h, l, r, role, bold=(role == "io"))
    for y0, y1 in zip(yb[:-1], yb[1:]):
        path(ax, [(xb, y0 - h / 2), (xb, y1 + h / 2)])
    ax.text(xb + wb / 2 + 0.02, yb[5] + 0.11, "P3", fontsize=6, color=MUTED, ha="left")
    ax.text(xb + wb / 2 + 0.02, yb[7] + 0.11, "P4", fontsize=6, color=MUTED, ha="left")
    ax.text(xb + wb / 2 + 0.02, yb[11] + 0.11, "P5", fontsize=6, color=MUTED, ha="left")

    # neck columns
    xf, xp, wn = 2.85, 4.65, 1.36
    yn = [0.95 + 0.44 * i for i in range(6)]          # FPN goes up, PAN goes down on the same rows
    fpn = [("11  Upsample ×2", "40×40", "small"), ("12  Concat", "40×40×384", "small"),
           ("13  C3k2", "40×40×128", "plain"), ("14  Upsample ×2", "80×80", "small"),
           ("15  Concat", "80×80×256", "small"), ("16  C3k2", "80×80×64", "plain")]
    pan = [("17  HPConv ★", "40×40×64", "hpconv"), ("18  Concat", "40×40×192", "small"),
           ("19  C3k2", "40×40×128", "plain"), ("20  HPConv ★", "20×20×128", "hpconv"),
           ("21  Concat", "20×20×384", "small"), ("22  C3k2", "20×20×256", "plain")]
    for (l, r, role), y in zip(fpn, yn):
        box(ax, xf, y, wn, h, l, r, role)
    for (l, r, role), y in zip(pan, yn[::-1]):
        box(ax, xp, y, wn, h, l, r, role)
    for y0, y1 in zip(yn[:-1], yn[1:]):
        path(ax, [(xf, y0 + h / 2), (xf, y1 - h / 2)])        # FPN upwards
        path(ax, [(xp, y1 - h / 2), (xp, y0 + h / 2)])        # PAN downwards

    # cross connections
    yP5 = yb[11]
    path(ax, [(xb + wb / 2, yP5), (xf, yP5), (xf, yn[0] - h / 2)])                 # 10 -> 11
    dot(ax, xf, yP5)
    path(ax, [(xf, yP5), (3.80, yP5), (3.80, yn[1]), (xp - wn / 2, yn[1])])        # 10 -> 21
    lane1a, lane1b = 1.93, 2.02
    path(ax, [(xb + wb / 2, yb[7]), (lane1a, yb[7]), (lane1a, yn[1]), (xf - wn / 2, yn[1])])   # 6 -> 12
    path(ax, [(xb + wb / 2, yb[5]), (lane1b, yb[5]), (lane1b, yn[4]), (xf - wn / 2, yn[4])])   # 4 -> 15
    path(ax, [(xf + wn / 2, yn[2]), (3.72, yn[2]), (3.72, yn[4]), (xp - wn / 2, yn[4])])       # 13 -> 18
    path(ax, [(xf + wn / 2, yn[5]), (xp - wn / 2, yn[5])])                                     # 16 -> 17

    # head
    xh, wh = 6.33, 1.40
    heads = [(yn[5], "Detect P3", "80×80, stride 8"), (yn[3], "Detect P4", "40×40, stride 16"),
             (yn[0], "Detect P5", "20×20, stride 32")]
    for y, l, r in heads:
        box(ax, xh, y, wh, h, l, r, "detect")
    path(ax, [(xf, yn[5] + h / 2), (xf, 3.48), (xh, 3.48), (xh, yn[5] + h / 2)])                # 16 -> Detect P3
    path(ax, [(xp + wn / 2, yn[3]), (xh - wh / 2, yn[3])])                                     # 19 -> Detect P4
    path(ax, [(xp + wn / 2, yn[0]), (xh - wh / 2, yn[0])])                                     # 22 -> Detect P5
    ax.text(xh, 3.72, "one-to-one (NMS-free)\nreg_max = 1 (no DFL)\nclasses: 27 identities / 1 face",
            ha="center", va="center", fontsize=6.2, color=MUTED, linespacing=1.35)
    box(ax, xh, 0.63, wh, h, "Boxes + scores (no NMS)", role="io")
    path(ax, [(xh, yn[0] - h / 2), (xh, 0.63 + h / 2)])

    # legend
    items = [("adown", "ADown (replaces Conv s2)"), ("eca", "C3k2 + ECA"),
             ("hpconv", "HPConv (replaces Conv s2)"), ("plain", "Unchanged layer"),
             ("detect", "Detect"), (None, "★ = changed layer")]
    renderer = fig.canvas.get_renderer()
    x = 0.12
    for role, text in items:  # place each entry after the measured width of the previous one (1 data unit = 1 inch)
        if role:
            fc, ec = ROLE[role]
            ax.add_patch(FancyBboxPatch((x, 0.17), 0.22, 0.13, boxstyle="round,pad=0,rounding_size=0.02",
                                        fc=fc, ec=ec, lw=1.0))
            x += 0.28
        t = ax.text(x, 0.235, text, va="center", fontsize=6.6, color=INK)
        x += t.get_window_extent(renderer).width / fig.dpi + 0.16
    assert x < W, f"legend too wide: {x:.2f} in"

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"architecture_proposed_yolo26n.{ext}", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
def modules():
    W, H = 7.16, 2.75
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    h = 0.19
    tints = {"adown": "#f4f8fe", "eca": "#fef6f1", "hpconv": "#f2fbf7"}

    # ---- (a) ADown ----
    section(ax, 0.06, 0.10, 2.30, 2.68, "(a) ADown", tints["adown"])
    cx = 1.18
    box(ax, cx, 2.28, 1.5, h, "Input  H×W×$C_1$", role="io")
    box(ax, cx, 1.94, 1.5, h, "AvgPool 2×2, s1", role="plain")
    box(ax, cx, 1.60, 1.5, h, "Split channels (½ | ½)", role="small")
    xl, xr = 0.62, 1.74
    box(ax, xl, 1.18, 0.98, h, "Conv 3×3, s2", role="adown", fs=6.3)
    box(ax, xr, 1.30, 0.98, h, "MaxPool 3×3, s2", role="plain", fs=6.3)
    box(ax, xr, 1.00, 0.98, h, "Conv 1×1", role="adown", fs=6.3)
    box(ax, cx, 0.62, 1.5, h, "Concat", role="small")
    box(ax, cx, 0.30, 1.5, h, "Output  H/2×W/2×$C_2$", role="io")
    path(ax, [(cx, 2.28 - h / 2), (cx, 1.94 + h / 2)]); path(ax, [(cx, 1.94 - h / 2), (cx, 1.60 + h / 2)])
    path(ax, [(xl, 1.60 - h / 2), (xl, 1.18 + h / 2)]); path(ax, [(xr, 1.60 - h / 2), (xr, 1.30 + h / 2)])
    path(ax, [(xr, 1.30 - h / 2), (xr, 1.00 + h / 2)])
    path(ax, [(xl, 1.18 - h / 2), (xl, 0.62 + h / 2)]); path(ax, [(xr, 1.00 - h / 2), (xr, 0.62 + h / 2)])
    path(ax, [(cx, 0.62 - h / 2), (cx, 0.30 + h / 2)])

    # ---- (b) ECA ----
    section(ax, 2.46, 0.10, 4.70, 2.68, "(b) C3k2 + ECA", tints["eca"])
    cx = 3.48
    box(ax, cx, 2.28, 1.5, h, "C3k2 output  H×W×C", role="io")
    box(ax, cx, 1.94, 1.5, h, "Global AvgPool → 1×1×C", role="plain")
    box(ax, cx, 1.60, 1.5, h, "Conv1D over channels, k = 3 / 5", role="eca")
    box(ax, cx, 1.18, 1.5, h, "Sigmoid → weights 1×1×C", role="plain")
    ax.add_patch(plt.Circle((cx, 0.80), 0.085, fc="#ffffff", ec=LINE, lw=0.8, zorder=3))
    ax.text(cx, 0.80, "×", ha="center", va="center", fontsize=8, zorder=4)
    box(ax, cx, 0.30, 1.5, h, "Output  H×W×C", role="io")
    for a, b in [(2.28, 1.94), (1.94, 1.60), (1.60, 1.18)]:
        path(ax, [(cx, a - h / 2), (cx, b + h / 2)])
    path(ax, [(cx, 1.18 - h / 2), (cx, 0.80 + 0.085)])
    xs = cx + 0.90
    path(ax, [(cx + 0.75, 2.28), (xs, 2.28), (xs, 0.80), (cx + 0.085, 0.80)])                 # identity branch
    path(ax, [(cx, 0.80 - 0.085), (cx, 0.30 + h / 2)])
    ax.text(xs + 0.05, 1.55, "x", fontsize=6.4, color=MUTED, style="italic", ha="left", va="center")

    # ---- (c) HPConv ----
    section(ax, 4.86, 0.10, 7.10, 2.68, "(c) HPConv", tints["hpconv"])
    cx = 5.98
    box(ax, cx, 2.28, 1.5, h, "Input  H×W×$C_1$", role="io")
    box(ax, cx, 1.94, 1.5, h, "Split channels (½ | ½)", role="small")
    xl, xr = 5.47, 6.60
    box(ax, xl, 1.62, 0.94, h, "AvgPool 3×3, s1", role="plain", fs=6.2)
    box(ax, xl, 1.32, 0.94, h, "Split (¼ | ¼)", role="small", fs=6.2)
    xl1, xl2 = 5.19, 5.77
    box(ax, xl1, 0.98, 0.55, h, "Conv 7×7, s2", role="hpconv", fs=5.6)
    box(ax, xl2, 0.98, 0.55, h, "Conv 5×5, s2", role="hpconv", fs=5.6)
    box(ax, xr, 1.62, 0.86, h, "MaxPool 3×3, s2", role="plain", fs=6.2)
    box(ax, xr, 1.32, 0.86, h, "Conv 1×1", role="hpconv", fs=6.2)
    box(ax, cx, 0.62, 1.76, h, "Concat (¼ + ¼ + ½)", role="small")
    box(ax, cx, 0.30, 1.5, h, "Output  H/2×W/2×$C_2$", role="io")
    path(ax, [(cx, 2.28 - h / 2), (cx, 1.94 + h / 2)])
    path(ax, [(xl, 1.94 - h / 2), (xl, 1.62 + h / 2)]); path(ax, [(xr, 1.94 - h / 2), (xr, 1.62 + h / 2)])
    path(ax, [(xl, 1.62 - h / 2), (xl, 1.32 + h / 2)]); path(ax, [(xr, 1.62 - h / 2), (xr, 1.32 + h / 2)])
    path(ax, [(xl1, 1.32 - h / 2), (xl1, 0.98 + h / 2)]); path(ax, [(xl2, 1.32 - h / 2), (xl2, 0.98 + h / 2)])
    path(ax, [(xl1, 0.98 - h / 2), (xl1, 0.62 + h / 2)]); path(ax, [(xl2, 0.98 - h / 2), (xl2, 0.62 + h / 2)])
    path(ax, [(xr, 1.32 - h / 2), (xr, 0.62 + h / 2)])
    path(ax, [(cx, 0.62 - h / 2), (cx, 0.30 + h / 2)])

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"modules_adown_eca_hpconv.{ext}", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
def architecture_flow():
    """Same network in a black-and-white, left-to-right block style (containers > stages > blocks, skip lines on top)."""
    W, H = 7.16, 2.55
    sans = {"family": "Nimbus Sans"}
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    K, LW = "#111111", 0.8
    fs, bh, gap = 5.4, 0.17, 0.07

    def rbox(x0, y0, x1, y1, r, lw=LW):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc="white", ec=K, lw=lw, zorder=1))

    def container(x0, x1, title):
        rbox(x0, 0.08, x1, 2.45, 0.16)
        ax.text(x0 + 0.10, 2.34, title, fontsize=7, ha="left", va="center", color=K, fontdict=sans)

    def line(pts, head=True, label=None, lpos=None, ha="center"):
        path(ax, pts, head=head, color=K, lw=LW)
        if label:
            ax.text(*lpos, label, fontsize=5.4, ha=ha, va="bottom", color=K, fontdict=sans)

    def jdot(x, y):
        ax.plot([x], [y], "o", ms=1.8, color=K, zorder=5)

    def stage(xc, w, ys, items, title, flow):
        """A stage box: title on top, one block per item at the given y centres, small arrows in the flow direction."""
        top = max(ys) + bh / 2 + 0.30
        bot = min(ys) - bh / 2 - 0.07
        rbox(xc - w / 2, bot, xc + w / 2, top, 0.07)
        ax.text(xc, top - 0.15, title, fontsize=fs, ha="center", va="center", color=K, fontdict=sans, linespacing=1.15)
        for y, (text, changed) in zip(ys, items):
            ax.add_patch(FancyBboxPatch((xc - (w - 0.08) / 2, y - bh / 2), w - 0.08, bh,
                                        boxstyle="round,pad=0,rounding_size=0.015", fc="white", ec=K,
                                        lw=1.25 if changed else 0.6, zorder=3))
            ax.text(xc, y, text + (r" $\bigstar$" if changed else ""), fontsize=fs, ha="center", va="center", color=K,
                    fontdict=sans, fontweight="bold" if changed else "normal", zorder=4)
        seq = sorted(ys, reverse=(flow == "down")) if flow != "none" else []
        for a, b in zip(seq[:-1], seq[1:]):
            s = 1 if b > a else -1
            path(ax, [(xc, a + s * bh / 2), (xc, b - s * bh / 2)], color=K, lw=0.6)
        return bot, top

    ym = 1.15  # main line of the backbone
    # ---- input + backbone ----
    ax.text(0.22, ym + 0.05, "Input\n640×640×3", fontsize=fs, ha="center", va="bottom", color=K, fontdict=sans)
    container(0.46, 3.70, "Backbone")
    w = 0.62
    xs = [0.85, 1.67, 2.49, 3.31]
    def col(n):  # y centres of n blocks, centred on the main line, first block on top
        span = (n - 1) * (bh + gap)
        return [ym + span / 2 - i * (bh + gap) for i in range(n)]
    stage(xs[0], w, col(3), [("Conv 3×3, s2", False), ("Conv 3×3, s2", False), ("C3k2", False)],
          "Stem\n160×160×64", "down")
    stage(xs[1], w, col(2), [("ADown", True), ("C3k2", False)], "Stage 1\n80×80×128", "down")
    stage(xs[2], w, col(2), [("ADown", True), ("C3k2 + ECA", True)], "Stage 2\n40×40×128", "down")
    stage(xs[3], w, col(4), [("ADown", True), ("C3k2 + ECA", True), ("SPPF", False), ("C2PSA", False)],
          "Stage 3\n20×20×256", "down")
    line([(0.36, ym), (xs[0] - w / 2, ym)])
    for i, lab in enumerate(["P2", "P3", "P4"]):
        x0, x1 = xs[i] + w / 2, xs[i + 1] - w / 2
        line([(x0, ym), (x1, ym)], label=lab, lpos=(x0 + 0.03, ym + 0.03), ha="left")

    # ---- neck ----
    container(3.78, 5.94, "Neck")
    xf, xp, wn = 4.46, 5.52, 0.68
    yr = [0.47 + i * (bh + gap) for i in range(6)]       # rows, bottom to top
    stage(xf, wn, yr, [("Upsample ×2", False), ("Concat (+P4)", False), ("C3k2", False),
                       ("Upsample ×2", False), ("Concat (+P3)", False), ("C3k2", False)],
          "Top-down path\n(FPN)", "up")
    stage(xp, wn, yr[::-1], [("HPConv", True), ("Concat", False), ("C3k2", False),
                             ("HPConv", True), ("Concat (+P5)", False), ("C3k2", False)],
          "Bottom-up path\n(PAN)", "down")
    xP, xB, xA = 3.86, 3.96, 4.06                        # vertical lanes in front of the FPN stage
    xQ, xR, xS = 4.90, 5.00, 5.10                        # lanes between FPN and PAN
    # P5: main line into the FPN, and along the bottom into the PAN
    line([(xs[3] + w / 2, ym), (xP, ym), (xP, yr[0]), (xf - wn / 2, yr[0])],
         label="P5", lpos=(xs[3] + w / 2 + 0.03, ym + 0.03), ha="left")
    jdot(xP, yr[0])
    line([(xP, yr[0]), (xP, 0.20), (xQ, 0.20), (xQ, yr[1]), (xp - wn / 2, yr[1])],
         label="P5 (20×20×256)", lpos=(4.42, 0.22))
    # skip connections over the top (P3 and P4 into the FPN)
    xb3 = xs[2] - w / 2 - 0.06   # branch points just in front of the next stage
    xb4 = xs[3] - w / 2 - 0.06
    jdot(xb3, ym); jdot(xb4, ym)
    line([(xb3, ym), (xb3, 2.20), (xA, 2.20), (xA, yr[4]), (xf - wn / 2, yr[4])],
         label="Skip connection: P3 (80×80×128)", lpos=((xb3 + 3.70) / 2, 2.22))
    line([(xb4, ym), (xb4, 2.06), (xB, 2.06), (xB, yr[1]), (xf - wn / 2, yr[1])],
         label="Skip: P4 (40×40×128)", lpos=(3.28, 2.08))
    # FPN -> PAN
    line([(xf + wn / 2, yr[5]), (xp - wn / 2, yr[5])])                                 # C3k2 (80×80) -> HPConv
    line([(xf + wn / 2, yr[2]), (xR, yr[2]), (xR, yr[4]), (xp - wn / 2, yr[4])])         # C3k2 (40×40) -> Concat
    jdot(xS, yr[5])

    # ---- head ----
    container(6.02, 6.74, "Head")
    xh, wh = 6.38, 0.56
    yh = [yr[5], yr[3], yr[0]]
    hb, ht = stage(xh, wh, yh, [("P3: 80×80", False), ("P4: 40×40", False), ("P5: 20×20", False)],
                   "Detect\n(YOLO26n)", "none")
    line([(xS, yr[5]), (xS, 2.14), (5.98, 2.14), (5.98, yr[5]), (xh - wh / 2, yr[5])],
         label="80×80×64", lpos=(5.52, 2.16))
    line([(xp + wn / 2, yr[3]), (xh - wh / 2, yr[3])])
    line([(xp + wn / 2, yr[0]), (xh - wh / 2, yr[0])])
    line([(xh + wh / 2, ym), (7.13, ym)])
    ax.text(6.94, ym + 0.04, "Output\n(no NMS)", fontsize=fs, ha="center", va="bottom", color=K,
            fontdict=sans, linespacing=1.15)
    ax.text(6.38, 0.20, r"$\bigstar$ = changed layer", fontsize=5.4, ha="center", va="center", color=K, fontdict=sans)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"architecture_proposed_yolo26n_flow.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    modules()
    architecture_flow()
    print("saved to", OUT)
