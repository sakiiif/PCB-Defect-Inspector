# Streamlit demo:  streamlit run app.py

import cv2, numpy as np, streamlit as st
from pcbinspect import detect_defects
from pcbinspect.detect import draw

st.set_page_config(page_title="PCB Defect Inspector", layout="wide")
st.title("PCB Defect Inspector")
st.caption("Classical reference-vs-test inspection: binarize, XOR, morphology, connected components.")

c1, c2 = st.columns(2)
ref_f = c1.file_uploader("Reference (golden) image", type=["jpg", "jpeg", "png"])
test_f = c2.file_uploader("Test image", type=["jpg", "jpeg", "png"])
align = st.sidebar.checkbox("ECC alignment (slow; for unaligned captures)", False)
min_area = st.sidebar.slider("Min defect area (px)", 5, 200, 25)
open_k = st.sidebar.slider("Noise-removal kernel", 1, 9, 5, step=2)

def read(f):
    return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

if ref_f and test_f:
    ref, test = read(ref_f), read(test_f)
    if ref.shape != test.shape:
        test = cv2.resize(test, (ref.shape[1], ref.shape[0]))
    dets, diff = detect_defects(ref, test, align=align, min_area=min_area, open_ksize=open_k)
    a, b = st.columns(2)
    a.image(cv2.cvtColor(draw(test, dets), cv2.COLOR_BGR2RGB), caption=f"{len(dets)} defect(s) found")
    b.image(diff, caption="Cleaned difference mask")
    st.dataframe([{"box": d.box, "area_px": d.area, "type": d.kind, "score": round(d.score, 2)} for d in dets])
else:
    st.info("Upload a reference and a test image (e.g. a *_temp.jpg / *_test.jpg pair from DeepPCB).")
