# Raspberry Pi 5 benchmark (NCNN). Run inside the unzipped folder:  python pi5_bench.py
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

models = sorted(glob.glob("*_ncnn_model"))
if not models:
    sys.exit("No *_ncnn_model folders found.")
for path in models:
    ms = bench(YOLO(path, task="detect"))
    print(f"{path.replace('_ncnn_model', ''):12s}  median {ms:8.2f} ms   {1000 / ms:7.2f} FPS")
