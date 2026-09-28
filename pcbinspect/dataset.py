# DeepPCB loader: pairs of (template, test) images plus ground truth boxes.

from __future__ import annotations
from pathlib import Path
import cv2


def load_pairs(root: str | Path, split: str = "test", limit: int | None = None):
    # Yield (name, ref, test, gt_boxes). gt_boxes: list of (x1,y1,x2,y2,type).

    root = Path(root) / "PCBData"
    lines = [l.split() for l in (root / f"{split}.txt").read_text().splitlines() if l.strip()]
    for img_rel, lbl_rel in lines[:limit]:
        stem = (root / img_rel).with_suffix("")
        ref = cv2.imread(str(stem) + "_temp.jpg")
        test = cv2.imread(str(stem) + "_test.jpg")
        if ref is None or test is None:
            continue
        gt = []
        lp = root / lbl_rel
        if lp.exists():
            for row in lp.read_text().splitlines():
                p = row.split()
                if len(p) == 5:
                    gt.append(tuple(int(v) for v in p))
        yield stem.name, ref, test, gt
