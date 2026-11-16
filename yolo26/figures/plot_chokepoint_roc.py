"""ROC curves of the ChokePoint two-stage pipelines, drawn from the real FAR/FRR sweeps of the Kaggle runs.

Source: det_points_accN.csv inside the four result zips (one row per threshold tau, 2,001 thresholds per pipeline).
TAR = 1 - FRR (an enrolled frame counts only if it is accepted with the right identity), FAR = share of unauthorized
frames accepted; both on the test session S4. Nothing is smoothed or interpolated beyond straight lines between points.
Writes next to this script:
  chokepoint_roc.pdf/.png         one IEEE column: full ROC with a zoomed inset of the top-left corner
  chokepoint_roc_2panel.pdf/.png  full page width: (a) full ROC, (b) zoom
"""
import csv
import io
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
KAGGLE = OUT.parent / "paper_code_chokepoint" / "three_accounts" / "KAGGLE"

# (account zip, pipeline name in det_points, legend name, colour, line style, width)
PIPES = [
    (1, "Proposed YOLO26n (ADown + ECA + HPConv) + ArcFace", "This work + ArcFace", "#2a78d6", "-", 1.7),
    (1, "YOLO26n + ArcFace", "YOLO26n + ArcFace", "#eb6834", "--", 1.1),
    (2, "YOLOv8n + ArcFace", "YOLOv8n + ArcFace", "#1baf7a", "-.", 1.1),
    (2, "YOLOv11n + ArcFace", "YOLOv11n + ArcFace", "#c98500", ":", 1.3),
    (3, "YOLO12n + ArcFace", "YOLO12n + ArcFace", "#d55181", (0, (5, 1, 1, 1)), 1.1),
    (3, "SSDLite320-MobileNetV3-Large (official torchvision) + ArcFace", "SSDLite-MobileNetV3 + ArcFace",
     "#008300", (0, (3, 1)), 1.1),
    (4, "YuNet (OpenCV) + ArcFace", "YuNet + ArcFace", "#6250d6", (0, (1, 1)), 1.3),
    (1, "Haar cascade (Viola-Jones) + LBPH", "Haar + LBPH", "#e34948", "-", 1.1),
    (1, "True eyes + ArcFace (upper bound)", "Manual eye labels + ArcFace", "#555555", (0, (6, 2)), 0.9),
]
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8})


def roc(acc, pipeline):
    """Sorted (FAR, TAR) points of one pipeline, starting at (0, 0)."""
    z = zipfile.ZipFile(KAGGLE / str(acc) / f"chokepoint_two_stage_all_results_acc{acc}.zip")
    pts = {(float(r["FAR"]), 1 - float(r["FRR"]))
           for r in csv.DictReader(io.TextIOWrapper(z.open(f"det_points_acc{acc}.csv"))) if r["pipeline"] == pipeline}
    assert pts, pipeline
    return [(0.0, 0.0)] + sorted(pts)


def auc(pts):
    return sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]))


DATA = [(name, color, ls, lw, roc(acc, p)) for acc, p, name, color, ls, lw in PIPES]


def draw(ax, xlim, ylim, legend):
    for name, color, ls, lw, pts in DATA:
        x, y = zip(*pts)
        ax.plot(x, y, color=color, ls=ls, lw=lw, zorder=4 if name.startswith("This work") else 3,
                label=f"{name} (AUC {auc(pts):.4f})")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, color="#d9d9d9", lw=0.5); ax.set_axisbelow(True)
    if legend:
        ax.legend(loc="lower right", fontsize=5.8, frameon=True, framealpha=0.95, edgecolor="#b0b0b0")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


FULL, ZOOM = ((0, 1), (0, 1.005)), ((-0.0005, 0.02), (0.90, 1.002))
XL, YL = "FAR (unauthorized frames accepted)", "TAR = 1 $-$ FRR"

if __name__ == "__main__":
    # one column: full ROC + inset zoom of the top-left corner
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    draw(ax, *FULL, legend=True)
    ax.set_xlabel(XL, fontweight="bold"); ax.set_ylabel(YL, fontweight="bold")
    ins = ax.inset_axes([0.44, 0.545, 0.40, 0.215])   # below the Haar curve, above the legend
    ins.set_facecolor("white"); ins.set_zorder(5)
    draw(ins, *ZOOM, legend=False)
    ins.set_xticks([0, 0.01, 0.02], ["0", "1%", "2%"]); ins.set_yticks([0.90, 0.95, 1.00])
    ins.tick_params(labelsize=5.5, pad=1)
    ax.indicate_inset_zoom(ins, edgecolor="#777777", lw=0.6)
    fig.tight_layout(); save(fig, "chokepoint_roc")

    # full width: (a) full, (b) zoom
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.9))
    draw(a, *FULL, legend=True); draw(b, *ZOOM, legend=False)
    b.set_xticks([0, 0.005, 0.01, 0.015, 0.02], ["0", "0.5%", "1%", "1.5%", "2%"])
    for axx, t in ((a, "(a) ROC"), (b, "(b) Zoom: FAR $\\leq$ 2%")):
        axx.set_xlabel(XL, fontweight="bold"); axx.set_ylabel(YL, fontweight="bold"); axx.set_title(t, fontsize=8.5)
    fig.tight_layout(); save(fig, "chokepoint_roc_2panel")

    # the numbers behind the figure (a table view for readers who cannot tell the lines apart)
    print(f"{'pipeline':32s} {'AUC':>7s} {'TAR@FAR=0':>10s} {'max TAR':>8s}")
    for name, *_, pts in DATA:
        print(f"{name:32s} {auc(pts):7.4f} {max(t for f, t in pts if f == 0):10.4f} {max(t for _, t in pts):8.4f}")
    print("saved to", OUT)
