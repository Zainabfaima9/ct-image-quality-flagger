import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import pydicom
import cv2
import pandas as pd
import io
import os
from datetime import datetime
from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CT Image Quality Flagger",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-header {
    background: linear-gradient(
        135deg,
        #0f2540 0%,
        #1a4d7a 50%,
        #2a7ba8 100%
    );
    padding: 2.8rem 2.5rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
    color: white;
}

.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 750;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    opacity: 0.92;
    font-size: 1.05rem;
}

.section-title {
    color: #0f2540;
    font-weight: 700;
}

.kpi-row {
    display: flex;
    gap: 0.8rem;
    margin: 1rem 0 1.5rem 0;
    flex-wrap: wrap;
}

.kpi-card {
    flex: 1;
    min-width: 150px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2a7ba8;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}

.kpi-card .val {
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f2540;
}

.kpi-card .lbl {
    font-size: 0.76rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.info-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    height: 100%;
}

.info-card h4 {
    margin-top: 0;
    color: #0f2540;
}

.result-card {
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    margin: 1rem 0;
    border: 1px solid #e2e8f0;
    background: #ffffff;
}

.term-box {
    background: #f8fafc;
    border-left: 4px solid #2a7ba8;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}

.simple-box {
    background: #eef7fb;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-top: 0.5rem;
}

.warning-box {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}

.disclaimer-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
}

.stButton>button {
    border-radius: 10px;
    border: 1px solid #2a7ba8;
    font-weight: 550;
}

.stButton>button:hover {
    background-color: #2a7ba8;
    color: white;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed #2a7ba8;
    border-radius: 12px;
    padding: 1rem;
}

section[data-testid="stSidebar"] {
    background-color: #0f2540;
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

HF_REPO = "zainabfatima9/ct-image-quality-flagger"
MODEL_FILENAME = "ct_quality_model_21patients_v2.h5"

INPUT_SIZE = (224, 224)
THRESHOLD = 0.25
GRADCAM_LAYER = "block4_conv3"

WINDOW_CENTER = 40
WINDOW_WIDTH = 400


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILENAME
    )

    return tf.keras.models.load_model(model_path)


@st.cache_resource
def build_gradcam_extractor(_model):

    base_model = _model.layers[0]

    conv_layer_model = tf.keras.Model(
        base_model.input,
        base_model.get_layer(GRADCAM_LAYER).output
    )

    return conv_layer_model, base_model


# Load model
try:
    model = load_model()
    conv_layer_model, base_model = build_gradcam_extractor(model)
    model_error = None

except Exception as e:
    model = None
    conv_layer_model = None
    base_model = None
    model_error = str(e)


# ============================================================
# DICOM PROCESSING
# ============================================================

def dicom_to_array(
    dicom_bytes,
    window_center=WINDOW_CENTER,
    window_width=WINDOW_WIDTH
):

    ds = pydicom.dcmread(io.BytesIO(dicom_bytes))

    pixel_array = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    hu = pixel_array * slope + intercept

    lower = window_center - window_width / 2
    upper = window_center + window_width / 2

    hu = np.clip(hu, lower, upper)

    hu = (
        (hu - lower)
        / (upper - lower)
        * 255
    ).astype(np.uint8)

    img = (
        Image.fromarray(hu)
        .convert("RGB")
        .resize(INPUT_SIZE)
    )

    metadata = {}

    metadata_fields = [
        ("KVP", "Tube voltage (kVp)"),
        ("XRayTubeCurrent", "Tube current (mA)"),
        ("Exposure", "Exposure (mAs)"),
        ("SliceThickness", "Slice thickness (mm)"),
        ("BodyPartExamined", "Body part"),
        ("SeriesDescription", "Series description"),
        ("Manufacturer", "Manufacturer"),
        ("ManufacturerModelName", "Scanner model")
    ]

    for tag, label in metadata_fields:

        if hasattr(ds, tag):

            value = getattr(ds, tag)

            if value is not None and str(value).strip() != "":
                metadata[label] = str(value)

    return img, metadata


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(img_pil):

    img_pil = img_pil.convert("RGB").resize(INPUT_SIZE)

    img_array = np.array(img_pil).astype(np.float32)

    img_input = np.expand_dims(
        img_array,
        axis=0
    )

    img_input_pp = preprocess_input(
        img_input.copy()
    )

    return img_array, img_input_pp


# ============================================================
# TEST-TIME AUGMENTATION
# ============================================================

def get_score(img_array_pp):

    if model is None:
        raise RuntimeError(
            "The model could not be loaded."
        )

    predictions = []

    # Original
    pred = model.predict(
        img_array_pp,
        verbose=0
    )[0][0]

    predictions.append(float(pred))

    # Horizontal flip
    flipped = np.flip(
        img_array_pp,
        axis=2
    )

    pred = model.predict(
        flipped,
        verbose=0
    )[0][0]

    predictions.append(float(pred))

    # Convert back to image representation
    img_uint = img_array_pp[0]

    # IMPORTANT:
    # The existing project uses the preprocessed image
    # for the TTA rotation/crop pipeline.
    center = (112, 112)

    for angle in [-5, 5]:

        rotation_matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            img_uint,
            rotation_matrix,
            INPUT_SIZE,
            borderMode=cv2.BORDER_REFLECT
        )

        rotated_batch = np.expand_dims(
            rotated,
            axis=0
        )

        pred = model.predict(
            rotated_batch,
            verbose=0
        )[0][0]

        predictions.append(float(pred))

    # Center crop / zoom
    crop = img_uint[
        11:213,
        11:213
    ]

    zoomed = cv2.resize(
        crop,
        INPUT_SIZE
    )

    zoomed_batch = np.expand_dims(
        zoomed,
        axis=0
    )

    pred = model.predict(
        zoomed_batch,
        verbose=0
    )[0][0]

    predictions.append(float(pred))

    return float(np.mean(predictions))


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam_overlay(
    img_array,
    img_array_pp
):

    if model is None:
        raise RuntimeError(
            "The model could not be loaded."
        )

    with tf.GradientTape() as tape:

        conv_output = conv_layer_model(
            img_array_pp
        )

        tape.watch(conv_output)

        x = conv_output

        found = False

        for layer in base_model.layers:

            if found:
                x = layer(x)

            if layer.name == GRADCAM_LAYER:
                found = True

        for layer in model.layers[1:]:
            x = layer(x)

        loss = x[:, 0]

    gradients = tape.gradient(
        loss,
        conv_output
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_out = conv_output[0]

    heatmap = (
        conv_out
        @ pooled_gradients[..., tf.newaxis]
    )

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.math.reduce_max(heatmap)
        + 1e-8
    )

    heatmap = cv2.resize(
        heatmap.numpy(),
        INPUT_SIZE,
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = np.clip(
        heatmap,
        0,
        1
    )

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        img_array.astype("uint8"),
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return overlay


# ============================================================
# RISK INTERPRETATION
# ============================================================

def get_risk_category(score):

    if score >= 0.50:

        return {
            "category": "Higher-risk pattern",
            "emoji": "🔴",
            "color": "#dc2626",
            "recommendation": "Review recommended",
            "explanation": (
                "The model produced a relatively high "
                "quality-risk score. This image should "
                "be reviewed rather than automatically accepted."
            )
        }

    elif score >= THRESHOLD:

        return {
            "category": "Borderline risk",
            "emoji": "🟠",
            "color": "#ea580c",
            "recommendation": "Review recommended",
            "explanation": (
                "The score reaches the project's review "
                "threshold. The result should be interpreted "
                "with professional judgment."
            )
        }

    elif score >= 0.15:

        return {
            "category": "Lower-risk pattern",
            "emoji": "🟡",
            "color": "#ca8a04",
            "recommendation": "No automatic flag",
            "explanation": (
                "The score is below the review threshold, "
                "but the model still shows some uncertainty."
            )
        }

    else:

        return {
            "category": "Low-risk pattern",
            "emoji": "🟢",
            "color": "#16a34a",
            "recommendation": "No automatic flag",
            "explanation": (
                "The model produced a relatively low "
                "quality-risk score."
            )
        }


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def process_image(img_pil):

    img_array, img_input_pp = prepare_image(
        img_pil
    )

    score = get_score(
        img_input_pp
    )

    overlay = make_gradcam_overlay(
        img_array,
        img_input_pp
    )

    return (
        img_array,
        overlay,
        score
    )


# ============================================================
# REPORT DATA
# ============================================================

def build_report_dataframe(results):

    rows = []

    for result in results:

        risk = get_risk_category(
            result["score"]
        )

        rows.append({
            "File": result["name"],
            "Risk score": round(
                result["score"],
                4
            ),
            "Category": risk["category"],
            "Recommendation": risk["recommendation"]
        })

    return pd.DataFrame(rows)


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = []


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 🩻 CT Image Quality Flagger"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Analyze CT",
        "Analysis Report",
        "How It Works",
        "Model Performance",
        "Research & Clinical Context"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Model snapshot")

st.sidebar.metric(
    "Recall",
    "85%"
)

st.sidebar.metric(
    "ROC-AUC",
    "0.839"
)

st.sidebar.caption(
    "VGG16 transfer-learning prototype"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Built by Zainab Fatima\n"
    "Medical Imaging Technology Student"
)


# ============================================================
# MODEL ERROR
# ============================================================

if model_error:

    st.error(
        "The model could not be loaded."
    )

    with st.expander(
        "Technical error details"
    ):
        st.code(model_error)

    st.stop()


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown("""
    <div class="main-header">
        <h1>🩻 CT Image Quality Flagger</h1>

        <p>
        AI-Assisted CT Image-Quality Assessment
        for Dose-Optimization Research
        </p>

        <p style="font-size:0.9rem;">
        Medical Imaging Technology × Artificial Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="kpi-row">

        <div class="kpi-card">
            <div class="val">85%</div>
            <div class="lbl">Recall</div>
        </div>

        <div class="kpi-card">
            <div class="val">54%</div>
            <div class="lbl">Precision</div>
        </div>

        <div class="kpi-card">
            <div class="val">0.66</div>
            <div class="lbl">F1-score</div>
        </div>

        <div class="kpi-card">
            <div class="val">0.839</div>
            <div class="lbl">ROC-AUC</div>
        </div>

        <div class="kpi-card">
            <div class="val">3.3%</div>
            <div class="lbl">Full-dose FPR</div>
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.subheader(
        "Why does this problem matter?"
    )

    st.write(
        """
        CT dose optimization is a balance. Lower radiation exposure is
        desirable, but excessive dose reduction can increase image noise
        and potentially reduce image quality.

        This prototype explores whether an AI model can provide an
        additional quality-check signal for CT images showing patterns
        associated with dose-related degradation.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="info-card">

        <h4>🎯 What it does</h4>

        The model produces a continuous
        <b>quality-risk score</b> for a CT slice.

        <br><br>

        A predefined threshold can then be used
        to flag images for additional review.

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-card">

        <h4>🧠 Why interpretability matters</h4>

        <b>Grad-CAM</b> creates a heatmap showing
        image regions that contributed to the model's
        prediction.

        <br><br>

        This provides an additional visual explanation
        rather than presenting only a numerical score.

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="info-card">

        <h4>👩‍⚕️ Why MIT matters</h4>

        CT optimization is not only a software problem.

        <br><br>

        Medical Imaging Technologists work with
        acquisition parameters, protocols and
        image-quality considerations.

        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader(
        "Important: what this tool does NOT do"
    )

    st.markdown("""
    <div class="disclaimer-box">

    <b>This is an educational/research prototype.</b>

    <br><br>

    It does not diagnose disease, replace a radiologist,
    determine whether a scan is clinically diagnostic,
    or make patient-management decisions.

    Its labels are based on a <b>noise-derived image-quality
    proxy</b>, not radiologist-confirmed diagnostic ground truth.

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ANALYZE CT
# ============================================================

elif page == "Analyze CT":

    st.title("🔬 Analyze CT")

    st.write(
        """
        Upload a CT slice in DICOM, PNG or JPG format.
        The model generates a quality-risk score and a Grad-CAM
        visualization.
        """
    )

    with st.expander(
        "ℹ️ Before you upload — understand the result"
    ):

        st.markdown("""
        **Quality-risk score**

        A continuous score produced by the model.
        A higher score indicates a stronger pattern associated
        with the project's flagged category.

        **Threshold**

        The project's current review threshold is **0.25**.

        In simple words: scores at or above 0.25 are flagged
        for additional review.

        **Grad-CAM**

        Grad-CAM is a visualization technique that highlights
        regions contributing to the model's prediction.

        It is an explanation aid — it does **not** prove that
        the highlighted region is clinically abnormal.
        """)

    uploaded_files = st.file_uploader(
        "Upload CT image(s)",
        type=[
            "dcm",
            "png",
            "jpg",
            "jpeg"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            already_added = any(
                r["name"] == uploaded_file.name
                for r in st.session_state.results
            )

            if already_added:
                continue

            try:

                if uploaded_file.name.lower().endswith(".dcm"):

                    image_pil, metadata = dicom_to_array(
                        uploaded_file.read()
                    )

                else:

                    image_pil = Image.open(
                        uploaded_file
                    ).convert("RGB")

                    metadata = {}

                with st.spinner(
                    f"Analyzing {uploaded_file.name}..."
                ):

                    img_array, overlay, score = process_image(
                        image_pil
                    )

                st.session_state.results.append({

                    "name": uploaded_file.name,

                    "img": img_array,

                    "overlay": overlay,

                    "score": score,

                    "metadata": metadata,

                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                })

            except Exception as e:

                st.error(
                    f"Could not analyze {uploaded_file.name}: {e}"
                )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if st.session_state.results:

        st.divider()

        st.subheader(
            "Analysis Results"
        )

        if st.button(
            "🗑️ Clear all results"
        ):

            st.session_state.results = []

            st.rerun()

        for result in st.session_state.results:

            risk = get_risk_category(
                result["score"]
            )

            st.markdown(
                f"""
                <div class="result-card"
                     style="border-left:6px solid {risk['color']};">

                <h3>
                {risk['emoji']} {risk['category']}
                </h3>

                <b>{result['name']}</b>

                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    result["img"].astype("uint8"),
                    caption="Input image",
                    use_container_width=True
                )

            with col2:

                st.image(
                    result["overlay"],
                    caption="Grad-CAM visualization",
                    use_container_width=True
                )

            score = result["score"]

            st.progress(
                min(max(score, 0), 1),
                text=f"Quality-risk score: {score:.3f}"
            )

            if score >= THRESHOLD:

                st.warning(
                    f"**{risk['recommendation']}** — "
                    f"The score is at or above the project "
                    f"threshold of {THRESHOLD:.2f}."
                )

            else:

                st.success(
                    f"**{risk['recommendation']}** — "
                    f"The score is below the project "
                    f"threshold of {THRESHOLD:.2f}."
                )

            st.markdown(
                f"""
                <div class="simple-box">

                <b>What does this mean?</b><br>
                {risk['explanation']}

                </div>
                """,
                unsafe_allow_html=True
            )

            if result["metadata"]:

                with st.expander(
                    "📋 DICOM information"
                ):

                    metadata_df = pd.DataFrame(
                        list(
                            result["metadata"].items()
                        ),
                        columns=[
                            "Parameter",
                            "Value"
                        ]
                    )

                    st.dataframe(
                        metadata_df,
                        use_container_width=True,
                        hide_index=True
                    )

            st.markdown("---")


# ============================================================
# ANALYSIS REPORT
# ============================================================

elif page == "Analysis Report":

    st.title("📋 Analysis Report")

    if not st.session_state.results:

        st.info(
            "No analyzed images yet. "
            "Go to **Analyze CT** and upload one or more images."
        )

    else:

        st.write(
            """
            This page summarizes the current analysis session.
            The report is intended for educational and research
            demonstration purposes.
            """
        )

        df = build_report_dataframe(
            st.session_state.results
        )

        total = len(df)

        flagged = int(
            (
                df["Risk score"] >= THRESHOLD
            ).sum()
        )

        mean_score = float(
            df["Risk score"].mean()
        )

        max_score = float(
            df["Risk score"].max()
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Images analyzed",
                total
            )

        with col2:
            st.metric(
                "Images flagged",
                flagged
            )

        with col3:
            st.metric(
                "Flagged percentage",
                f"{flagged / total * 100:.1f}%"
            )

        with col4:
            st.metric(
                "Mean risk score",
                f"{mean_score:.3f}"
            )

        st.divider()

        st.subheader(
            "Series-level summary"
        )

        st.write(
            f"""
            **Maximum risk score:** {max_score:.3f}

            **Review threshold:** {THRESHOLD:.2f}

            **Interpretation:** An image is counted as flagged
            when its model score reaches or exceeds the project
            threshold.
            """
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download CSV report",
            data=csv,
            file_name="ct_quality_analysis_report.csv",
            mime="text/csv"
        )

        st.divider()

        st.subheader(
            "Methodology note"
        )

        st.info(
            """
            The reported score comes from the VGG16-based model
            using the project's five-view Test-Time Augmentation
            procedure. The result should be interpreted as a
            model-generated quality-risk signal, not as a clinical
            determination of diagnostic adequacy.
            """)


# ============================================================
# HOW IT WORKS
# ============================================================

elif page == "How It Works":

    st.title("⚙️ How It Works")

    st.write(
        """
        This section explains the project in both technical
        and simple terms so that students, clinicians and
        researchers can understand the workflow.
        """
    )

    # --------------------------------------------------------
    # PROBLEM
    # --------------------------------------------------------

    with st.expander(
        "1️⃣ The clinical problem",
        expanded=True
    ):

        st.markdown("""
        ### CT dose and image quality

        CT optimization aims to achieve sufficient image quality
        while avoiding unnecessary radiation exposure.

        When radiation dose is reduced, image noise can increase.
        If dose reduction becomes excessive, image quality may be
        affected.

        **In simple words:**  
        We want the radiation dose to be as low as reasonably
        appropriate for the imaging task, but not at the expense
        of useful image quality.
        """)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    with st.expander(
        "2️⃣ Dataset and labeling"
    ):

        st.markdown("""
        ### Dataset

        The project uses paired full-dose and low-dose chest CT
        data from the **LDCT-and-Projection-data** collection
        available through The Cancer Imaging Archive (TCIA).

        The development dataset contains scans from **21 patients**.

        ### How were the labels created?

        The target labels were derived from
        **per-patient-normalized image-noise measurements**
        comparing low-dose and full-dose images.

        **Important:** these are not radiologist-confirmed labels.

        **In simple words:**  
        Instead of asking a radiologist to manually label every
        slice, the project used differences in image noise between
        paired scans as a measurable quality-related proxy.
        """)

    # --------------------------------------------------------
    # VGG16
    # --------------------------------------------------------

    with st.expander(
        "3️⃣ VGG16 and transfer learning"
    ):

        st.markdown("""
        ### What is VGG16?

        VGG16 is a convolutional neural network originally
        developed for image-recognition tasks.

        This project uses **transfer learning**.

        ### What is transfer learning?

        Transfer learning means starting with a model that has
        already learned useful visual patterns and adapting it
        to a new task.

        **In simple words:**  
        Instead of teaching the model to recognize visual
        patterns completely from zero, we reuse learned features
        and adapt them to CT image-quality assessment.

        The final model produces a continuous quality-risk score.
        """)

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    with st.expander(
        "4️⃣ CT preprocessing"
    ):

        st.markdown("""
        ### DICOM → Hounsfield Units → windowing → resizing

        For DICOM input, pixel values are converted using the
        available **Rescale Slope** and **Rescale Intercept**
        to obtain Hounsfield Unit (HU) values.

        The project then applies a CT window of:

        - **Window Level (WL): 40**
        - **Window Width (WW): 400**

        The resulting image is converted to RGB and resized to
        **224 × 224 pixels** for the VGG16 model.

        **In simple words:**  
        Raw CT numbers are converted into a clinically meaningful
        CT intensity representation, a selected intensity range
        is displayed, and the image is resized to the dimensions
        expected by the model.
        """)

    # --------------------------------------------------------
    # TTA
    # --------------------------------------------------------

    with st.expander(
        "5️⃣ Test-Time Augmentation (TTA)"
    ):

        st.markdown("""
        ### What is TTA?

        Test-Time Augmentation means giving the same image to the
        model in several slightly modified forms.

        This implementation uses five views:

        1. Original image
        2. Horizontal flip
        3. Small rotation of −5°
        4. Small rotation of +5°
        5. Center crop / zoom

        The predictions are averaged.

        **In simple words:**  
        Instead of trusting one prediction, we ask the model to
        assess several slightly different versions of the same
        image and combine their predictions.

        This can make the final prediction less dependent on one
        particular presentation of the image.
        """)

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    with st.expander(
        "6️⃣ Threshold and review flag"
    ):

        st.markdown(f"""
        ### Threshold = {THRESHOLD:.2f}

        The model produces a continuous score rather than simply
        saying "good" or "bad."

        A threshold converts that continuous score into a
        review decision.

        **In this project:**

        - Score < **{THRESHOLD:.2f}** → no automatic review flag
        - Score ≥ **{THRESHOLD:.2f}** → review flag

        **In simple words:**  
        The threshold is simply the point at which the project
        says: "this image deserves additional attention."

        The threshold should not be interpreted as a universal
        clinical cutoff.
        """)

    # --------------------------------------------------------
    # GRADCAM
    # --------------------------------------------------------

    with st.expander(
        "7️⃣ Grad-CAM explainability"
    ):

        st.markdown("""
        ### What is Grad-CAM?

        **Grad-CAM (Gradient-weighted Class Activation Mapping)**
        is an explainability technique that creates a heatmap
        showing image regions that contributed to the model's
        prediction.

        **In simple words:**  
        It helps answer:

        > "Where was the model looking when it made this prediction?"

        The heatmap should be treated as an explanation aid,
        not proof of clinical abnormality.
        """)

    # --------------------------------------------------------
    # TESTING
    # --------------------------------------------------------

    with st.expander(
        "8️⃣ Patient-level held-out testing"
    ):

        st.markdown("""
        ### Why patient-level separation matters

        CT slices from the same patient can be highly similar.

        If slices from the same patient appeared in both training
        and testing, performance could look artificially strong.

        Therefore, the final evaluation used **fully held-out
        patients** that were not used during model development.

        The reported evaluation contains **3 held-out patients**
        and **1,058 CT slices**.

        **In simple words:**  
        The model was tested on patients it had not seen during
        development, rather than simply testing it on random
        slices from patients it had already encountered.
        """)

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    with st.expander(
        "9️⃣ Limitations"
    ):

        st.markdown("""
        ### Current limitations

        **1. Small dataset**  
        The model was developed using 21 patients from one public
        dataset.

        **2. Limited external validation**  
        Performance on other hospitals, scanners, protocols,
        populations and reconstruction methods is not established.

        **3. Noise-derived labels**  
        The target is based on image-noise measurements rather than
        radiologist-confirmed diagnostic adequacy.

        **4. Slice-level assessment**  
        A single CT slice does not represent the complete clinical
        context of a CT examination.

        **5. Research prototype**  
        The model has not undergone clinical validation or
        regulatory evaluation.

        **Therefore:** this application should not be used for
        patient-care decisions.
        """)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.markdown("""
    <div class="kpi-row">

        <div class="kpi-card">
            <div class="val">85%</div>
            <div class="lbl">Recall</div>
        </div>

        <div class="kpi-card">
            <div class="val">54%</div>
            <div class="lbl">Precision</div>
        </div>

        <div class="kpi-card">
            <div class="val">0.66</div>
            <div class="lbl">F1-score</div>
        </div>

        <div class="kpi-card">
            <div class="val">0.839</div>
            <div class="lbl">ROC-AUC</div>
        </div>

        <div class="kpi-card">
            <div class="val">3.3%</div>
            <div class="lbl">Full-dose FPR</div>
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Evaluation on 3 fully held-out test patients "
        "(1,058 CT slices)."
    )

    st.divider()

    st.subheader(
        "What do these metrics mean?"
    )

    metrics_col1, metrics_col2 = st.columns(2)

    with metrics_col1:

        st.markdown("""
        <div class="term-box">

        <b>Recall — 85%</b>

        Measures how many of the actual flagged cases
        were successfully identified by the model.

        <div class="simple-box">

        <b>In simple words:</b>
        When a case really belonged to the flagged group,
        the model detected most of those cases.

        </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="term-box">

        <b>Precision — 54%</b>

        Measures how many of the cases flagged by the model
        were actually flagged according to the project's labels.

        <div class="simple-box">

        <b>In simple words:</b>
        Not every model flag was a true flagged case.

        </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="term-box">

        <b>F1-score — 0.66</b>

        Combines precision and recall into a single measure.

        <div class="simple-box">

        <b>In simple words:</b>
        It gives a balanced summary of the model's ability
        to identify flagged cases while limiting incorrect flags.

        </div>

        </div>
        """, unsafe_allow_html=True)

    with metrics_col2:

        st.markdown("""
        <div class="term-box">

        <b>ROC-AUC — 0.839</b>

        Measures the model's ability to distinguish between
        the two groups across different score thresholds.

        <div class="simple-box">

        <b>In simple words:</b>
        It summarizes how well the model separates the groups
        overall, rather than at only one cutoff.

        </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="term-box">

        <b>Full-dose false-positive rate — 3.3%</b>

        This sanity-check measures how often the model incorrectly
        flagged the evaluated full-dose images.

        <div class="simple-box">

        <b>In simple words:</b>
        It helps check whether the model is simply flagging
        many images regardless of dose-related degradation.

        </div>

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "ROC Curve"
        )

        roc_path = os.path.join(
            "assets",
            "roc_curve.png"
        )

        if os.path.exists(roc_path):

            st.image(
                roc_path,
                use_container_width=True
            )

        else:

            st.info(
                "ROC curve image will appear here once "
                "assets/roc_curve.png is added to the repository."
            )

    with col2:

        st.subheader(
            "t-SNE Feature Separation"
        )

        tsne_path = os.path.join(
            "assets",
            "tsne_features.png"
        )

        if os.path.exists(tsne_path):

            st.image(
                tsne_path,
                use_container_width=True
            )

        else:

            st.info(
                "t-SNE visualization will appear here once "
                "assets/tsne_features.png is added to the repository."
            )

    st.divider()

    st.subheader(
        "Development journey"
    )

    st.write(
        """
        Several approaches were explored during model development,
        including data augmentation, model ensembling, fine-tuning,
        increasing the patient count from 13 to 21, threshold
        optimization and Test-Time Augmentation.

        Not every modification improved performance. Some approaches,
        including ensembling and fine-tuning, performed worse than
        the selected augmented baseline.

        Keeping unsuccessful experiments is important because model
        development is not simply about finding a high number—it is
        also about understanding which approaches actually help.
        """
    )

    st.warning(
        "These metrics describe this project's evaluation dataset. "
        "They should not be interpreted as evidence of clinical "
        "performance in other hospitals or populations."
    )


# ============================================================
# RESEARCH & CLINICAL CONTEXT
# ============================================================

elif page == "Research & Clinical Context":

    st.title(
        "🔎 Research & Clinical Context"
    )

    st.write(
        """
        This project sits at the intersection of CT dose
        optimization, image quality, artificial intelligence
        and Medical Imaging Technology.
        """
    )

    # --------------------------------------------------------
    # DOSE
    # --------------------------------------------------------

    st.markdown(
        "### ☢️ 1. CT dose optimization"
    )

    st.write(
        """
        CT optimization aims to obtain images that are adequate
        for the intended clinical task while avoiding unnecessary
        radiation exposure.

        Reducing dose can increase image noise. Therefore,
        dose reduction should be considered together with
        image quality rather than as an isolated target.
        """
    )

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    st.markdown(
        "### 🧠 2. Where AI fits"
    )

    st.write(
        """
        The role proposed in this prototype is not autonomous
        decision-making.

        Instead, AI provides an additional computational signal
        that could potentially help identify images requiring
        additional attention.
        """
    )

    st.markdown("""
    <div class="simple-box">

    <b>In simple words:</b>

    AI is being explored as a second-check layer —
    not as a replacement for the professional responsible
    for CT acquisition and image evaluation.

    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # MIT
    # --------------------------------------------------------

    st.markdown(
        "### 👩‍⚕️ 3. Why this matters for Medical Imaging Technologists"
    )

    st.write(
        """
        Medical Imaging Technologists are directly involved in
        CT acquisition, protocol implementation, patient
        positioning and image-quality considerations.

        This means that successful AI implementation in CT
        should not be considered only from a software or
        algorithmic perspective.

        The technology must fit the actual imaging workflow.
        """
    )

    # --------------------------------------------------------
    # EMMAH
    # --------------------------------------------------------

    st.markdown(
        "### 🌍 4. Why this is relevant to healthcare innovation"
    )

    st.write(
        """
        A technically accurate model is only one part of a
        healthcare technology.

        Real-world implementation also raises questions about:

        - workflow integration
        - explainability
        - user trust
        - data quality
        - generalizability
        - validation
        - patient safety
        - regulatory requirements
        - cost and accessibility

        This is why the project is framed as a
        **decision-support prototype** rather than simply
        an image-classification model.
        """
    )

    # --------------------------------------------------------
    # RESEARCH QUESTION
    # --------------------------------------------------------

    st.markdown(
        "### 💡 5. The research question behind the prototype"
    )

    st.markdown("""
    <div class="term-box">

    <b>Core question:</b>

    Can an interpretable AI system provide an additional,
    consistent signal for identifying CT images that show
    patterns associated with dose-related image-quality
    degradation?

    <div class="simple-box">

    <b>Important:</b>

    This prototype does not claim that the question has been
    clinically solved. It demonstrates one possible technical
    approach and identifies the validation steps that would
    be required before clinical translation.

    </div>

    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # FUTURE
    # --------------------------------------------------------

    st.markdown(
        "### 🚀 6. What would be needed next?"
    )

    st.write(
        """
        A stronger research system would require larger and
        more diverse datasets, external validation across
        scanners and institutions, clinically meaningful
        reference standards, prospective evaluation,
        workflow testing and appropriate regulatory assessment.

        These steps are essential before considering clinical
        deployment.
        """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built by Zainab Fatima | Medical Imaging Technology Student | "
    "Educational/research prototype only — not for clinical use."
)
