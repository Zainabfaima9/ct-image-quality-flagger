import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import pydicom
import cv2
import pandas as pd
import io
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="CT Image Quality Flagger", page_icon="🩻", layout="wide")

# ============ MODEL LOADING ============
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="zainabfatima9/ct-image-quality-flagger",
        filename="ct_quality_model_21patients_v2.h5"
    )
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_resource
def build_gradcam_models(_model):
    base_model = _model.layers[0]
    last_conv_layer_model = tf.keras.Model(base_model.input, base_model.get_layer('block4_conv3').output)
    classifier_input = tf.keras.Input(shape=last_conv_layer_model.output.shape[1:])
    x = classifier_input
    found = False
    for layer in base_model.layers:
        if found:
            x = layer(x)
        if layer.name == 'block4_conv3':
            found = True
    for layer in _model.layers[1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    return last_conv_layer_model, classifier_model

model = load_model()
last_conv_layer_model, classifier_model = build_gradcam_models(model)

THRESHOLD = 0.25

# ============ HELPER FUNCTIONS ============
def dicom_to_array(dicom_bytes, window_center=40, window_width=400):
    ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
    hu = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    hu = np.clip(hu, lower, upper)
    hu = ((hu - lower) / (upper - lower) * 255).astype(np.uint8)
    img = Image.fromarray(hu).convert('RGB').resize((224, 224))

    metadata = {}
    for tag, name in [('KVP', 'KVP'), ('XRayTubeCurrent', 'Tube Current (mA)'),
                       ('Exposure', 'Exposure (mAs)'), ('SliceThickness', 'Slice Thickness (mm)'),
                       ('BodyPartExamined', 'Body Part')]:
        if hasattr(ds, tag):
            metadata[name] = getattr(ds, tag)
    return img, metadata

def make_gradcam(img_array_pp):
    with tf.GradientTape() as tape:
        conv_output = last_conv_layer_model(img_array_pp)
        tape.watch(conv_output)
        preds = classifier_model(conv_output)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_output[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
    return np.clip(heatmap_resized, 0, 1), preds.numpy()[0][0]

def get_confidence_tier(score):
    if score >= 0.5:
        return "🔴 High-confidence flag — review recommended", "high"
    elif score >= THRESHOLD:
        return "🟠 Flagged — borderline, use clinical judgment", "medium"
    elif score >= 0.15:
        return "🟡 Likely acceptable — minor uncertainty", "low"
    else:
        return "🟢 Acceptable — low degradation detected", "clear"

def process_image(img_pil):
    img_array = np.array(img_pil.resize((224, 224)))
    img_input = np.expand_dims(img_array.astype(float), axis=0)
    img_input_pp = preprocess_input(img_input.copy())
    heatmap, score = make_gradcam(img_input_pp)
    return img_array, heatmap, score

# ============ UI ============
st.title("🩻 CT Image Quality Flagger")
st.markdown("**AI-Assisted CT Dose Optimization: An Image-Quality–Aware Decision-Support Prototype**")
st.markdown("*Medical Imaging Technology × Artificial Intelligence*")

with st.expander("📋 Why I built this"):
    st.write("""
    Lowering CT radiation dose protects patients — but push it too far, and image quality
    can drop below what's diagnostically reliable. Pakistani research (Yaseen et al., 2024)
    introduced the "Acceptable Quality Dose" concept, showing that dose optimization can't
    ignore image quality. This prototype explores whether AI can provide a consistent,
    early flag for scans where dose reduction may have compromised diagnostic usefulness —
    not replacing a radiologist's or technologist's judgment, but supporting it.
    """)

with st.expander("⚙️ How it works"):
    st.write("""
    A VGG16-based model (transfer learning) was trained on 21 patients' paired full-dose
    and low-dose chest CT scans from the LDCT-and-Projection-data (TCIA) dataset. Quality
    labels were derived from per-patient-normalized noise measurements, not a single global
    threshold. The model was validated on 3 fully held-out patients it never saw during
    training, and cross-checked against 1,058 full-dose (undegraded) images to confirm it
    isn't over-flagging.
    """)

with st.expander("⚠️ Limitations"):
    st.write("""
    - Trained on 21 patients from one public dataset — external validation on other scanners
      and institutions would be needed before any clinical use.
    - Quality labels are derived from a noise-based proxy, not radiologist-confirmed ground truth.
    - This tool assesses image quality only — it does not diagnose disease or interpret findings.
    - Not intended for clinical use. Educational/research prototype only.
    """)

st.sidebar.header("📊 Model Performance")
st.sidebar.metric("Recall (Review_Flag)", "85%")
st.sidebar.metric("Precision (Review_Flag)", "54%")
st.sidebar.metric("ROC-AUC", "0.839")
st.sidebar.metric("Full-dose false-positive rate", "3.3%")
st.sidebar.caption("Evaluated on 3 fully held-out test patients (1,058 CT slices)")

st.divider()

# Sample images
st.subheader("🖼️ Try a sample image")
sample_cols = st.columns(4)
sample_files = {
    "Sample 1 (Acceptable)": "sample_images/sample_acceptable_1.png",
    "Sample 2 (Acceptable)": "sample_images/sample_acceptable_2.png",
    "Sample 3 (Flagged)": "sample_images/sample_flagged_1.png",
    "Sample 4 (Flagged)": "sample_images/sample_flagged_2.png",
}
selected_sample = None
for i, (label, path) in enumerate(sample_files.items()):
    if sample_cols[i].button(label):
        selected_sample = path

st.divider()

# Upload
st.subheader("📤 Upload CT image(s)")
st.caption("Supports DICOM (.dcm) or standard image formats (PNG/JPG). Upload multiple files for series-level analysis.")
uploaded_files = st.file_uploader("Upload one or more CT slices", type=['dcm', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)

results = []

def render_result(name, img_array, heatmap, score, metadata=None):
    tier_text, tier_level = get_confidence_tier(score)
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_array.astype('uint8'), caption=f"{name} — Original", use_column_width=True)
    with col2:
        st.image(img_array.astype('uint8'), caption="Grad-CAM", use_column_width=True)
    st.markdown(f"**Score:** {score:.3f} — {tier_text}")
    st.progress(min(float(score), 1.0))
    if metadata:
        with st.expander("DICOM metadata"):
            for k, v in metadata.items():
                st.write(f"**{k}:** {v}")
    st.divider()
    results.append({"File": name, "Score": round(float(score), 3), "Assessment": tier_text})

if selected_sample:
    img_pil = Image.open(selected_sample).convert('RGB')
    img_array, heatmap, score = process_image(img_pil)
    render_result("Sample", img_array, heatmap, score)

if uploaded_files:
    for f in uploaded_files:
        if f.name.lower().endswith('.dcm'):
            img_pil, metadata = dicom_to_array(f.read())
            img_array, heatmap, score = process_image(img_pil)
            render_result(f.name, img_array, heatmap, score, metadata)
        else:
            img_pil = Image.open(f).convert('RGB')
            img_array, heatmap, score = process_image(img_pil)
            render_result(f.name, img_array, heatmap, score)

    if len(results) > 1:
        st.subheader("📈 Series Summary")
        flagged_count = sum(1 for r in results if r['Score'] >= THRESHOLD)
        st.metric("Slices flagged for review", f"{flagged_count} / {len(results)}",
                   f"{flagged_count/len(results)*100:.1f}%")

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download report as CSV", csv, "ct_quality_report.csv", "text/csv")

st.divider()
st.caption("Built by Zainab Fatima | Medical Imaging Technology Student | Educational/research prototype only — not for clinical use.")
