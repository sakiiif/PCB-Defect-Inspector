# Evaluate the pipeline on DeepPCB.  python scripts/run_eval.py --data ../DeepPCB

import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pcbinspect import detect_defects, DEFECT_NAMES
from pcbinspect.dataset import load_pairs
from pcbinspect.evaluate import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--min-area", type=int, default=25)
ap.add_argument("--open-ksize", type=int, default=5)
ap.add_argument("--align", action="store_true", help="ECC-align test to reference (slow, for unaligned captures)")
a = ap.parse_args()

det = lambda r, t: detect_defects(r, t, align=a.align, min_area=a.min_area,
                                  open_ksize=a.open_ksize)[0]
m = evaluate(load_pairs(a.data, a.split, a.limit), det)
print(f"precision {m['precision']:.3f}  recall {m['recall']:.3f}  F1 {m['f1']:.3f}  "
      f"({m['tp']} TP / {m['fp']} FP / {m['fn']} FN)  {m['ms_per_image']:.1f} ms/img")
for c, (r, n) in m["per_class"].items():
    print(f"  {DEFECT_NAMES.get(c, c):10s} recall {r:.3f}  (n={n})")
