import sys, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S, OUT = sys.argv[1], sys.argv[2]
runs = [("Proposed YOLO26n (ADown + ECA + HPConv)", f"{S}/runs/nk_hpconv_s0/results.csv", "#1f6fd1", "-"),
        ("YOLO26n (baseline)", f"{S}/runs/0_base_s0/results.csv", "#c75f08", "--")]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9})
fig, ax = plt.subplots(figsize=(3.5, 2.9))  # one IEEE column
for label, path, col, ls in runs:
    d = pd.read_csv(path); d.columns = d.columns.str.strip()
    y = d["metrics/mAP50(B)"]
    ax.plot(d["epoch"], y, color=col, ls=ls, lw=1.4, label=label)
    b = y.idxmax()
    ax.plot(d["epoch"][b], y[b], "o", ms=4, color=col, mec="white", mew=0.8)
    print(label, "best", round(y[b], 3), "epoch", int(d["epoch"][b]), "epochs", len(d))
ax.set_xlabel("Epoch"); ax.set_ylabel("mAP@0.5")
ax.grid(True, color="#d9d9d9", lw=0.5); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.set_xlim(0, 101); ax.set_ylim(0.85, 0.975); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, frameon=False, ncol=1)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/map50_curve_proposed_vs_yolo26n.{ext}", dpi=300)
