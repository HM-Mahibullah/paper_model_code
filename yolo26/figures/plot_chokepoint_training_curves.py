"""Training curves (validation mAP per epoch) of all ChokePoint face detectors, drawn from the real training logs.

Reads runs/<model>/results.csv straight from the three Kaggle result zips (nothing is invented or smoothed):
  acc1: proposed, YOLO26n   acc2: YOLOv8n, YOLOv11n   acc3: YOLO12n, SSDLite-MobileNetV3 (mAP@0.5 every 5 epochs)
Writes next to this script:
  chokepoint_map50_all_models.pdf/.png        mAP@0.5 per epoch (one IEEE column)
  chokepoint_map50-95_all_models.pdf/.png     mAP@0.5:0.95 per epoch (YOLO models; SSDLite logs mAP@0.5 only)
  chokepoint_training_curves_2panel.pdf/.png  both side by side (full page width)
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

# (zip account, run, legend name, colour, line style); colour follows the model in every figure of the paper
MODELS = [
    (1, "face_proposed",     "Proposed YOLO26n",     "#2a78d6", "-"),
    (1, "face_yolo26n",      "YOLO26n",              "#eb6834", "--"),
    (2, "face_yolov8n",      "YOLOv8n",              "#1baf7a", "-."),
    (2, "face_yolo11n",      "YOLOv11n",             "#c98500", ":"),
    (3, "face_yolo12n",      "YOLO12n",              "#d55181", (0, (5, 1, 1, 1))),
    (3, "face_ssdlite_mbv3", "SSDLite-MobileNetV3",  "#008300", "-"),
]
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8})


def load(acc, run):
    """Rows of results.csv inside the account's zip, with stripped column names."""
    z = zipfile.ZipFile(KAGGLE / str(acc) / f"chokepoint_two_stage_all_results_acc{acc}.zip")
    rows = list(csv.DictReader(io.TextIOWrapper(z.open(f"runs/{run}/results.csv"))))
    return [{k.strip(): v for k, v in r.items()} for r in rows]


def series(acc, run, metric):
    """(epochs, values) of metric 'map50' or 'map5095'; None if the model does not log it."""
    rows = load(acc, run)
    if "val_mAP50" in rows[0]:                       # SSDLite (torchvision): mAP@0.5 every 5 epochs only
        if metric != "map50":
            return None
        pts = [(int(r["epoch"]), float(r["val_mAP50"])) for r in rows if r["val_mAP50"]]
    else:                                            # Ultralytics
        key = "metrics/mAP50(B)" if metric == "map50" else "metrics/mAP50-95(B)"
        pts = [(int(float(r["epoch"])), float(r[key])) for r in rows]
    return [p[0] for p in pts], [p[1] for p in pts]


def draw(ax, metric, ylim, ylabel):
    for acc, run, name, color, ls in MODELS:
        s = series(acc, run, metric)
        if s is None:
            continue
        x, y = s
        main = run == "face_proposed"
        ax.plot(x, y, color=color, ls=ls, lw=1.6 if main else 1.1, label=name, zorder=4 if main else 3,
                marker="o" if run == "face_ssdlite_mbv3" else None, ms=2.6)
    ax.set_xlim(0, 101)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(True, color="#d9d9d9", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=6.6, frameon=True, framealpha=0.95, edgecolor="#b0b0b0")


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    # y-axes start above the first-epoch values so that the converged part is readable (epoch 1 is clipped)
    LIM50, LIM95 = (0.90, 1.0), (0.60, 0.90)
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    draw(ax, "map50", LIM50, "mAP@0.5")
    save(fig, "chokepoint_map50_all_models")

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    draw(ax, "map5095", LIM95, "mAP@0.5:0.95")
    save(fig, "chokepoint_map50-95_all_models")

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.7))
    draw(a, "map50", LIM50, "mAP@0.5")
    draw(b, "map5095", LIM95, "mAP@0.5:0.95")
    a.set_title("(a) mAP@0.5", fontsize=8.5)
    b.set_title("(b) mAP@0.5:0.95", fontsize=8.5)
    save(fig, "chokepoint_training_curves_2panel")

    for acc, run, name, *_ in MODELS:                 # the numbers behind the curves, for the text
        for metric in ("map50", "map5095"):
            s = series(acc, run, metric)
            if s:
                x, y = s
                best = max(range(len(y)), key=y.__getitem__)
                print(f"{name:22s} {metric:8s} best {y[best]:.3f} at epoch {x[best]:3d}, last {y[-1]:.3f} "
                      f"(epoch {x[-1]})")
    print("saved to", OUT)
