# Box-level evaluation: greedy IoU matching, precision / recall / F1, timing.

from __future__ import annotations
import time
import numpy as np


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua else 0.0


def match(pred_boxes, gt_boxes, thr: float = 0.3):
    # Greedy match. Returns (tp, fp, fn, matched_gt_indices).
    used = set()
    for pb in pred_boxes:
        best, bj = 0.0, -1
        for j, g in enumerate(gt_boxes):
            if j in used:
                continue
            v = iou(pb, g[:4])
            if v > best:
                best, bj = v, j
        if best >= thr:
            used.add(bj)
    tp = len(used)
    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp, used


def evaluate(pairs, detector, thr: float = 0.3):
    # Run `detector(ref, test)->list[Detection]` over pairs; aggregate metrics.
    TP = FP = FN = 0
    per_class, times, counts = {}, [], []
    for name, ref, test, gt in pairs:
        t0 = time.perf_counter()
        dets = detector(ref, test)
        times.append(time.perf_counter() - t0)
        tp, fp, fn, hit = match([d.box for d in dets], gt, thr)
        TP += tp; FP += fp; FN += fn
        counts.append(len(dets))
        for j, g in enumerate(gt):
            c = per_class.setdefault(g[4], [0, 0])
            c[1] += 1
            c[0] += j in hit
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=prec, recall=rec, f1=f1, tp=TP, fp=FP, fn=FN,
                ms_per_image=1000 * float(np.mean(times)) if times else 0.0,
                per_class={k: (v[0] / v[1] if v[1] else 0.0, v[1]) for k, v in sorted(per_class.items())},
                counts=counts)
