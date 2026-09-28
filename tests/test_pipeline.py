import numpy as np, cv2, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pcbinspect import detect_defects
from pcbinspect.evaluate import iou, match
from pcbinspect.spc import c_chart


def _board():
    img = np.full((200, 200, 3), 40, np.uint8)
    cv2.line(img, (20, 60), (180, 60), (220, 220, 220), 8)
    cv2.line(img, (20, 140), (180, 140), (220, 220, 220), 8)
    return img


def test_identical_images_have_no_detections():
    b = _board()
    assert detect_defects(b, b, align=False)[0] == []


def test_short_between_traces_is_found():
    ref, test = _board(), _board()
    cv2.rectangle(test, (95, 60), (105, 140), (220, 220, 220), -1)   # copper bridge
    dets, _ = detect_defects(ref, test, align=False)
    assert len(dets) >= 1 and dets[0].kind == "extra_copper"


def test_iou_and_match():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    tp, fp, fn, _ = match([(0, 0, 10, 10), (50, 50, 60, 60)], [(1, 1, 10, 10, 2)])
    assert (tp, fp, fn) == (1, 1, 0)


def test_c_chart_flags_spike():
    counts = [5, 4, 6, 5, 5, 4, 6, 5, 30]
    _, ucl, _, ooc = c_chart(counts, baseline=8)
    assert 8 in ooc and ucl > 5
