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

# ============ CUSTOM STYLING ============
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f2540 0%, #1a4d7a 50%, #2a7ba8 100%);
        padding: 3rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem; color: white;
    }
    .main-header h1 { margin: 0; font-size: 2.4rem; }
    .main-header p { margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 1.08rem; }
    .kpi-row { display: flex; gap: 0.8rem; margin: 1rem 0 1.5rem 0; flex-wrap: wrap; }
    .kpi-card {
        flex: 1; min-width: 150px; background: #ffffff; border: 1px solid #e2e8f0;
        border-left: 4px solid #2a7ba8; border-radius: 10px; padding: 1rem 1.2rem;
    }
    .kpi-card .val { font-size: 1.6rem; font-weight: 700; color: #0f2540; }
    .kpi-card .lbl { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.03em; }
    .result-card {
        border-radius: 14px; padding: 1.3rem 1.5rem; margin: 1rem 0;
        border: 1px solid #e2e8f0; border-left: 6px solid var(--accent, #94a3b8); background: #ffffff;
    }
    .tier-badge { display: inline-block; padding: 0.35rem 1rem; border-radius: 20px; font-weight: 600; font-size: 0.92rem; }
    .feature-card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.3rem; height: 100%;
    }
    .feature-card h4 { margin-top: 0; color: #0f2540; }
    .stButton>button { border-radius: 10px; border: 1px solid #2a7ba8; color: #2a7ba8; font-weight: 500; }
    .stButton>button:hover { background-color: #2a7ba8; color: white; }
    div[data-testid="stFileUploader"] { border: 2px dashed #2a7ba8; border-radius: 12px; padding: 1rem; }
    section[data-testid="stSidebar"] { background-color: #0f2540; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ============ MODEL LOADING ============
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="zainabfatima9/ct-image-quality-flagger",
        filename="ct_quality_model_21patients_v2.h5"
    )
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def build_gradcam_extractor(_model):
    base_model = _model.layers[0]
    conv_layer_model = tf.keras.Model(base_model.input, base_model.get_layer('block4_conv3').output)
    return conv_layer_model, base_model

model = load_model()
conv_layer_model, base_model = build_gradcam_extractor(model)
THRESHOLD = 0.25

# ============ HELPERS ============
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

def get_score(img_array_pp):
    """TTA: averages 5 views — matches the reported 85% recall / 3.3% FP-rate metrics."""
    preds = [model.predict(img_array_pp, verbose=0)[0][0]]
    flipped = np.flip(img_array_pp, axis=2)
    preds.append(model.predict(flipped, verbose=0)[0][0])
    img_uint = img_array_pp[0]
    center = (112, 112)
    for angle in [-5, 5]:
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_uint, rot_mat, (224, 224), borderMode=cv2.BORDER_REFLECT)
        preds.append(model.predict(np.expand_dims(rotated, axis=0), verbose=0)[0][0])
    crop = img_uint[11:213, 11:213]
    zoomed = cv2.resize(crop, (224, 224))
    preds.append(model.predict(np.expand_dims(zoomed, axis=0), verbose=0)[0][0])
    return float(np.mean(preds))

def make_gradcam_overlay(img_array, img_array_pp):
    with tf.GradientTape() as tape:
        conv_output = conv_layer_model(img_array_pp)
        tape.watch(conv_output)
        x = conv_output
        found = False
        for layer in base_model.layers:
            if found:
                x = layer(x)
            if layer.name == 'block4_conv3':
                found = True
        for layer in model.layers[1:]:
            x = layer(x)
        loss = x[:, 0]
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_output[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224), interpolation=cv2.INTER_CUBIC)
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_array.astype('uint8'), 0.6, heatmap_color, 0.4, 0)

def get_confidence_tier(score):
    if score >= 0.5:
        return "High-confidence flag — review recommended", "#dc2626", "🔴"
    elif score >= THRESHOLD:
        return "Flagged — borderline, use clinical judgment", "#ea580c", "🟠"
    elif score >= 0.15:
        return "Likely acceptable — minor uncertainty", "#ca8a04", "🟡"
    else:
        return "Acceptable — low degradation detected", "#16a34a", "🟢"

def process_image(img_pil):
    img_array = np.array(img_pil.resize((224, 224)))
    img_input = np.expand_dims(img_array.astype(float), axis=0)
    img_input_pp = preprocess_input(img_input.copy())
    score = get_score(img_input_pp)
    overlaid = make_gradcam_overlay(img_array, img_input_pp)
    return img_array, overlaid, score

# ============ FAQ ASSISTANT (lightweight, no external API) ============
FAQ = {
    "dataset": "The model was trained on 21 patients' paired full-dose and low-dose chest CT scans from the LDCT-and-Projection-data collection (The Cancer Imaging Archive).",
    "accuracy": "On 3 fully held-out test patients (never seen during training), the model achieves 85% recall and 54% precision for flagged scans, with a ROC-AUC of 0.839.",
    "how it works": "A VGG16-based deep learning model (transfer learning) analyzes each CT slice and outputs a quality-risk score. Scores are averaged across 5 augmented views (Test-Time Augmentation) for stability.",
    "label": "Quality labels come from per-patient-normalized noise measurements comparing low-dose to full-dose versions of the same scan — not from radiologist review.",
    "clinical use": "No — this is an educational/research prototype. It has not been clinically validated and should not be used for real patient-care decisions.",
    "dicom": "Yes — upload a .dcm file directly and the app will extract dose-related metadata (KVP, exposure, slice thickness) alongside the quality assessment.",
    "grad-cam": "Grad-CAM highlights the image regions that most influenced the model's decision, shown as a heatmap overlay next to each result.",
    "who built": "This was built independently by Zainab Fatima, a Medical Imaging Technology student, as a research prototype exploring AI-assisted CT dose optimization.",
}

def answer_faq(question):
    q = question.lower()
    for key, ans in FAQ.items():
        if key in q:
            return ans
    if any(w in q for w in ["hello", "hi", "hey"]):
        return "Hi! Ask me about the dataset, accuracy, how the model works, or whether this is validated for clinical use."
    return "I don't have a specific answer for that yet — try asking about the dataset, accuracy, how it works, DICOM support, or clinical use. You can also check the 'How It Works' page."

# ============ SESSION STATE ============
if "results" not in st.session_state:
    st.session_state.results = []

# ============ SIDEBAR NAVIGATION ============
st.sidebar.markdown("## 🩻 CT Quality Flagger")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Home", "Try It Yourself", "How It Works", "Model Performance", "Ask a Question"],
                         label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
st.sidebar.metric("Recall", "85%")
st.sidebar.metric("ROC-AUC", "0.839")
st.sidebar.caption("Built by Zainab Fatima\nMedical Imaging Technology Student")

# ============ HOME PAGE ============
if page == "Home":
    st.markdown("""
    <div class="main-header">
        <h1>🩻 CT Image Quality Flagger</h1>
        <p>AI-Assisted CT Dose Optimization — An Image-Quality–Aware Decision-Support Prototype</p>
        <p style="font-size:0.9rem;">Medical Imaging Technology × Artificial Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="kpi-row">
        <div class="kpi-card"><div class="val">85%</div><div class="lbl">Recall (Review_Flag)</div></div>
        <div class="kpi-card"><div class="val">54%</div><div class="lbl">Precision</div></div>
        <div class="kpi-card"><div class="val">0.839</div><div class="lbl">ROC-AUC</div></div>
        <div class="kpi-card"><div class="val">3.3%</div><div class="lbl">Full-dose false-positive rate</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="feature-card"><h4>🎯 What it does</h4>
        Flags CT slices where dose reduction may have degraded diagnostic quality —
        a second-check layer, not a replacement for professional judgment.</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="feature-card"><h4>📊 Why it's different</h4>
        Trained on real paired clinical data with patient-level held-out testing,
        Grad-CAM interpretability, and honest sanity-checks against full-dose scans.</div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="feature-card"><h4>🇵🇰 Grounded in local research</h4>
        Directly connects to Pakistani CT-dose literature (Yaseen et al., 2024) on
        Acceptable Quality Dose — not an imported, disconnected idea.</div>""", unsafe_allow_html=True)

    st.write("")
    st.info("👈 Use the sidebar to try the model, read how it works, see full performance metrics, or ask a question.")

# ============ TRY IT PAGE ============
elif page == "Try It Yourself":
    st.title("🔬 Try the Model")

    st.subheader("🖼️ Try a sample image")
    sample_cols = st.columns(4)
    sample_files = {
        "Sample 1 (Acceptable)": "sample_images/sample_acceptable_1.png",
        "Sample 2 (Acceptable)": "sample_images/sample_acceptable_2.png",
        "Sample 3 (Flagged)": "sample_images/sample_flagged_1.png",
        "Sample 4 (Flagged)": "sample_images/sample_flagged_2.png",
    }
    for i, (label, path) in enumerate(sample_files.items()):
        if sample_cols[i].button(label, key=f"sample_btn_{i}"):
            img_pil = Image.open(path).convert('RGB')
            img_array, overlaid, score = process_image(img_pil)
            st.session_state.results.append({"name": f"Sample: {label}", "img": img_array,
                                              "overlay": overlaid, "score": score, "metadata": None})

    st.divider()
    st.subheader("📤 Upload CT image(s)")
    st.caption("Supports DICOM (.dcm) or standard image formats (PNG/JPG). Upload multiple files for series-level analysis.")
    uploaded_files = st.file_uploader("Upload one or more CT slices", type=['dcm', 'png', 'jpg', 'jpeg'],
                                       accept_multiple_files=True, key="uploader")

    if uploaded_files:
        for f in uploaded_files:
            already_added = any(r["name"] == f.name for r in st.session_state.results)
            if not already_added:
                if f.name.lower().endswith('.dcm'):
                    img_pil, metadata = dicom_to_array(f.read())
                    img_array, overlaid, score = process_image(img_pil)
                else:
                    img_pil = Image.open(f).convert('RGB')
                    img_array, overlaid, score = process_image(img_pil)
                    metadata = None
                st.session_state.results.append({"name": f.name, "img": img_array,
                                                  "overlay": overlaid, "score": score, "metadata": metadata})

    if st.session_state.results:
        st.divider()
        st.subheader("Results")
        if st.button("🗑️ Clear all results"):
            st.session_state.results = []
            st.rerun()

        for r in st.session_state.results:
            tier_text, color, emoji = get_confidence_tier(r["score"])
            st.markdown(f'<div class="result-card" style="--accent:{color};">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.image(r["img"].astype('uint8'), caption=f"{r['name']} — Original", use_container_width=True)
            with col2:
                st.image(r["overlay"], caption="Grad-CAM — model's focus area", use_container_width=True)
            st.markdown(f'<span class="tier-badge" style="background:{color}22;color:{color};">{emoji} {tier_text}</span>',
                        unsafe_allow_html=True)
            st.write("")
            st.progress(min(r["score"], 1.0), text=f"Quality-risk score: {r['score']:.3f}")
            if r["metadata"]:
                with st.expander("DICOM metadata"):
                    for k, v in r["metadata"].items():
                        st.write(f"**{k}:** {v}")
            st.markdown('</div>', unsafe_allow_html=True)

        if len(st.session_state.results) > 1:
            st.subheader("📈 Series Summary")
            flagged_count = sum(1 for r in st.session_state.results if r['score'] >= THRESHOLD)
            total = len(st.session_state.results)
            st.metric("Slices flagged for review", f"{flagged_count} / {total}", f"{flagged_count/total*100:.1f}%")
            df = pd.DataFrame([{"File": r["name"], "Score": round(r["score"], 3),
                                 "Assessment": get_confidence_tier(r["score"])[0]} for r in st.session_state.results])
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download report as CSV", csv, "ct_quality_report.csv", "text/csv")

# ============ HOW IT WORKS PAGE ============
elif page == "How It Works":
    st.title("⚙️ How It Works")

    with st.expander("📋 Why I built this", expanded=True):
        st.write("""
        Lowering CT radiation dose protects patients — but push it too far, and image quality
        can drop below what's diagnostically reliable. Pakistani research (Yaseen et al., 2024)
        introduced the "Acceptable Quality Dose" concept, showing that dose optimization can't
        ignore image quality. This prototype explores whether AI can provide a consistent,
        early flag for scans where dose reduction may have compromised diagnostic usefulness —
        not replacing a radiologist's or technologist's judgment, but supporting it.
        """)

    with st.expander("🧠 Model & training", expanded=True):
        st.write("""
        A VGG16-based model (transfer learning) was trained on 21 patients' paired full-dose
        and low-dose chest CT scans from the LDCT-and-Projection-data (TCIA) dataset. Quality
        labels were derived from per-patient-normalized noise measurements — a raw global
        threshold was tried first but mostly reflected which patient a slice came from, not
        genuine dose-related degradation, so it was corrected. The final model was validated
        on 3 fully held-out patients never seen during training, cross-checked against 1,058
        full-dose (undegraded) images to confirm it isn't over-flagging, and uses Test-Time
        Augmentation (5-view averaging) for more stable predictions.
        """)

    with st.expander("⚠️ Limitations", expanded=True):
        st.write("""
        - Trained on 21 patients from one public dataset — external validation on other scanners
          and institutions would be needed before any clinical use.
        - Quality labels are derived from a noise-based proxy, not radiologist-confirmed ground truth.
        - This tool assesses image quality only — it does not diagnose disease or interpret findings.
        - Not intended for clinical use. Educational/research prototype only.
        """)

# ============ MODEL PERFORMANCE PAGE ============
elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.markdown("""
    <div class="kpi-row">
        <div class="kpi-card"><div class="val">85%</div><div class="lbl">Recall (Review_Flag)</div></div>
        <div class="kpi-card"><div class="val">54%</div><div class="lbl">Precision</div></div>
        <div class="kpi-card"><div class="val">0.66</div><div class="lbl">F1-score</div></div>
        <div class="kpi-card"><div class="val">0.839</div><div class="lbl">ROC-AUC</div></div>
        <div class="kpi-card"><div class="val">3.3%</div><div class="lbl">Full-dose false-positive rate</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("All metrics evaluated on 3 fully held-out test patients never seen during training (1,058 CT slices).")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve")
        try:
            st.image("assets/roc_curve.png", use_container_width=True)
        except Exception:
            st.caption("ROC curve image not found in assets/ — add roc_curve.png to display it here.")
    with col2:
        st.subheader("t-SNE Feature Separation")
        try:
            st.image("assets/tsne_features.png", use_container_width=True)
        except Exception:
            st.caption("t-SNE image not found in assets/ — add tsne_features.png to display it here.")

    st.divider()
    st.subheader("Development journey")
    st.write("""
    Several improvement techniques were systematically tested against a fixed held-out test
    set: data augmentation, model ensembling, fine-tuning, expanding from 13 to 21 patients,
    threshold optimization, and Test-Time Augmentation. Not all of them helped — ensembling
    and fine-tuning underperformed the augmented baseline, for instance — and those honest
    results were kept rather than discarded, since knowing what *doesn't* help is part of the
    evidence too.
    """)

# ============ FAQ / ASK A QUESTION PAGE ============
elif page == "Ask a Question":
    st.title("💬 Ask About This Project")
    st.caption("A lightweight FAQ assistant — matches your question to common topics (dataset, accuracy, how it works, DICOM, clinical use). Not a general AI chatbot.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_q = st.text_input("Ask a question, e.g. \"What dataset was used?\"")
    if st.button("Ask") and user_q:
        answer = answer_faq(user_q)
        st.session_state.chat_history.append((user_q, answer))

    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")
        st.divider()

    st.caption("Try: \"What dataset was used?\" · \"How accurate is it?\" · \"Is this validated for clinical use?\" · \"Does it support DICOM?\"")

st.divider()
st.caption("Built by Zainab Fatima | Medical Imaging Technology Student | Educational/research prototype only — not for clinical use.")
