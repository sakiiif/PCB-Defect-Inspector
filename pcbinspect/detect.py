# Reference vs test defect detection: align -> binarize -> difference -> filter -> classify.

from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

DEFECT_NAMES = {0: "background", 1: "open", 2: "short", 3: "mousebite",
                4: "spur", 5: "copper", 6: "pin-hole"}


@dataclass
class Detection:
    box: tuple          # x1, y1, x2, y2
    area: int
    kind: str           # heuristic label: "missing_copper" / "extra_copper"
    score: float        # mean absolute difference inside the box (0-1)


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def align_ecc(ref: np.ndarray, test: np.ndarray, max_iter: int = 100, eps: float = 1e-5):
    # Refine translation/rotation of `test` onto `ref` with ECC. Falls back to identity.

    r, t = to_gray(ref), to_gray(test)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    try:
        _, warp = cv2.findTransformECC(r, t, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
    except cv2.error:
        return test, np.eye(2, 3, dtype=np.float32)
    h, w = r.shape
    aligned = cv2.warpAffine(test, warp, (w, h),
                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
    return aligned, warp


def binarize(gray: np.ndarray) -> np.ndarray:
    # Copper = 255. Blur then Otsu; robust to per-image brightness shifts.

    g = cv2.GaussianBlur(gray, (5, 5), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def detect_defects(ref: np.ndarray, test: np.ndarray, *, align: bool = True,
                   min_area: int = 25, open_ksize: int = 5, pad: int = 6,
                   merge_ksize: int = 15) -> tuple[list[Detection], np.ndarray]:
    # Return (detections, difference mask).

    # Misregistration produces thin 1-2 px edge slivers in the XOR image;
    # morphological opening removes them while real defects (blobs) survive.

    if align:
        test, _ = align_ecc(ref, test)
    rg, tg = to_gray(ref), to_gray(test)
    rb, tb = binarize(rg), binarize(tg)

    diff = cv2.bitwise_xor(rb, tb)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k)

    # Merge fragments of one defect (e.g. a short spanning two traces)
    km = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_ksize, merge_ksize))
    merged = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, km)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    h, w = rg.shape
    absdiff = cv2.absdiff(rg, tg).astype(np.float32) / 255.0
    dets: list[Detection] = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        comp = labels[y:y + bh, x:x + bw] == i
        area = int((diff[y:y + bh, x:x + bw][comp] > 0).sum())
        if area < min_area:
            continue

        # missing copper: reference copper where test has none
        miss = int(((rb[y:y + bh, x:x + bw] > 0) & (tb[y:y + bh, x:x + bw] == 0) & comp).sum())
        kind = "missing_copper" if miss >= area / 2 else "extra_copper"
        box = (max(x - pad, 0), max(y - pad, 0), min(x + bw + pad, w), min(y + bh + pad, h))
        score = float(absdiff[y:y + bh, x:x + bw][comp].mean())
        dets.append(Detection(box, area, kind, score))
    return dets, diff


def draw(img: np.ndarray, dets: list[Detection]) -> np.ndarray:
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for d in dets:
        x1, y1, x2, y2 = d.box
        color = (0, 0, 255) if d.kind == "missing_copper" else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out
