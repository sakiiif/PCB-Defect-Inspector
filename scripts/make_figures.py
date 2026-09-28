# Generate figures: detection example + SPC c-chart.  python scripts/make_figures.py --data ../DeepPCB

import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pcbinspect import detect_defects
from pcbinspect.detect import draw
from pcbinspect.dataset import load_pairs
from pcbinspect.spc import c_chart

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
out = pathlib.Path(__file__).resolve().parents[1] / "assets"
pairs = list(load_pairs(a.data, "test", 60))

# 1) pipeline figure on one sample
name, ref, test, gt = pairs[3]
dets, diff = detect_defects(ref, test, align=False)
gtimg = test.copy()
for x1, y1, x2, y2, _ in gt:
    cv2.rectangle(gtimg, (x1, y1), (x2, y2), (0, 200, 0), 2)
fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
for a_, im, t in zip(ax, [ref, test, diff, draw(test, dets)],
                     ["Reference (template)", "Test board", "Cleaned XOR difference",
                      f"Detections ({len(dets)})  vs GT ({len(gt)})"]):
    a_.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im, cmap="gray")
    a_.set_title(t, fontsize=10); a_.axis("off")
plt.tight_layout(); plt.savefig(out / "pipeline.png", dpi=110); plt.close()

# 2) SPC c-chart across boards
counts = [len(detect_defects(r, t, align=False)[0]) for _, r, t, _ in pairs]
cbar, ucl, lcl, ooc = c_chart(counts)
plt.figure(figsize=(9, 3.6))
plt.plot(counts, "o-", ms=4, lw=1, color="#1f4e79")
plt.axhline(cbar, color="gray"); plt.axhline(ucl, color="red", ls="--"); plt.axhline(lcl, color="red", ls="--")
plt.scatter(ooc, [counts[i] for i in ooc], color="red", zorder=5, label="out of control")
plt.xlabel("Board #"); plt.ylabel("Defects detected"); plt.title(f"c-chart  (c̄={cbar:.1f}, UCL={ucl:.1f})")
if ooc: plt.legend()
plt.tight_layout(); plt.savefig(out / "c_chart.png", dpi=110); plt.close()
print("figures written; OOC boards:", ooc)
