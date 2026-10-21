# Jetson benchmark (PyTorch, CUDA FP16). Run inside the unzipped folder:  python jetson_bench.py
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("WARNING: CUDA GPU not available. Install the NVIDIA Jetson PyTorch wheel for your JetPack, then run again.")
print("GPU:", torch.cuda.get_device_name(0))

sys.path.insert(0, ".")
import simam_head  # needed to load the SimAM checkpoint
from ultralytics import YOLO

import glob, statistics, sys, time
import cv2

IMGSZ, WARMUP, RUNS = 640, 100, 1000
imgs = [cv2.imread(p) for p in sorted(glob.glob("bench_images/*"))]
imgs = [im for im in imgs if im is not None]
if not imgs:
    sys.exit("No images found in bench_images/. Run this script inside the unzipped folder.")
print(f"{len(imgs)} real test images, imgsz {IMGSZ}, batch 1, {WARMUP} warm-up + {RUNS} timed runs (median)")


def bench(model, **kw):
    for i in range(WARMUP):
        model.predict(imgs[i % len(imgs)], imgsz=IMGSZ, verbose=False, **kw)
    times = []
    for i in range(RUNS):
        t = time.perf_counter()
        model.predict(imgs[i % len(imgs)], imgsz=IMGSZ, verbose=False, **kw)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)

models = sorted(glob.glob("*.pt"))
if not models:
    sys.exit("No .pt files found.")
for path in models:
    ms = bench(YOLO(path, task="detect"), device=0, half=True)
    print(f"{path.replace('.pt', ''):12s}  median {ms:8.2f} ms   {1000 / ms:7.2f} FPS")
