import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from gradcam import make_gradcam_heatmap, overlay_heatmap

IMG = 224
CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]
THRESHOLD = 0.80

st.set_page_config(page_title="Kidney CT Classifier", page_icon="🩻", layout="centered")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("kidney_ct_model.keras", compile=False)


model = load_model()

st.title("Kidney CT Scan Classifier")
st.caption(
    "Classifies kidney CT slices as Normal, Cyst, Tumor or Stone, "
    "and shows a Grad-CAM heatmap of where the model looked."
)
st.warning("Educational project only. Not a medical device and not a substitute for a radiologist.")

uploaded = st.file_uploader("Upload a kidney CT image", type=["jpg", "jpeg", "png"])

if uploaded:
    original = np.array(Image.open(uploaded).convert("RGB"), dtype=np.float32)
    arr = tf.image.resize(original, (IMG, IMG)).numpy()  # same preprocessing as training

    probs = model.predict(arr[None], verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])

    if conf >= THRESHOLD:
        st.success(f"Prediction: **{CLASS_NAMES[idx]}** ({conf:.1%} confidence)")
    else:
        st.error(
            f"Uncertain: top guess {CLASS_NAMES[idx]} at only {conf:.1%} confidence. "
            "Refer to a radiologist."
        )

    heatmap = make_gradcam_heatmap(arr[None], model)
    col1, col2 = st.columns(2)
    col1.image(np.uint8(arr), caption="Uploaded scan")
    col2.image(overlay_heatmap(arr, heatmap), caption="Grad-CAM: where the model looked")

    st.subheader("Class probabilities")
    st.bar_chart({name: float(p) for name, p in zip(CLASS_NAMES, probs)})

    with st.expander("How to read this"):
        st.markdown(
            f"- The model is **96.8% accurate when confident** (≥ {THRESHOLD:.0%}) on the test set, "
            "and only 65.3% when uncertain, so uncertain scans are referred to a human.\n"
            "- **Red/yellow** areas in the heatmap influenced the prediction most. "
            "Ideally they sit on the kidneys."
        )
