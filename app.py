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
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

HF_REPO = "zainabfatima9/ct-image-quality-flagger"
MODEL_FILENAME = "ct_quality_model_21patients_v2.h5"

INPUT_SIZE = (224, 224)

THRESHOLD = 0.25

GRADCAM_LAYER = "block4_conv3"

WINDOW_CENTER = 40
WINDOW_WIDTH = 400


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>

/* ---------- General ---------- */

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ---------- Top navigation ---------- */

.nav-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f2540;
}

/* ---------- Hero ---------- */

.hero {
    background: linear-gradient(
        135deg,
        #0b2138 0%,
        #174d73 55%,
        #2b82a8 100%
    );
    border-radius: 24px;
    padding: 3.5rem 3rem;
    color: white;
    margin-bottom: 1.5rem;
}

.hero-small {
    color: rgba(255,255,255,0.82);
    font-size: 0.95rem;
    margin-bottom: 0.7rem;
}

.hero h1 {
    font-size: 3rem;
    line-height: 1.1;
    margin: 0;
}

.hero p {
    font-size: 1.08rem;
    max-width: 760px;
    line-height: 1.7;
}

/* ---------- Cards ---------- */

.card {
    background: #ffffff;
    border: 1px solid #e4eaf0;
    border-radius: 16px;
    padding: 1.4rem;
    height: 100%;
}

.card h3 {
    color: #0f2540;
    margin-top: 0;
}

.card p {
    color: #475569;
    line-height: 1.6;
}

/* ---------- Demo cards ---------- */

.demo-card {
    background: #ffffff;
    border: 1px solid #dfe7ee;
    border-radius: 16px;
    padding: 0.7rem;
    margin-bottom: 0.5rem;
}

.demo-title {
    font-weight: 700;
    color: #0f2540;
    padding: 0.4rem;
}

/* ---------- Result ---------- */

.result-box {
    border-radius: 18px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid #e2e8f0;
    background: #ffffff;
}

.score {
    font-size: 2.7rem;
    font-weight: 750;
    color: #0f2540;
}

.result-label {
    color: #64748b;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---------- Explanation ---------- */

.explain {
    background: #f5f9fc;
    border-left: 4px solid #2a7ba8;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    line-height: 1.65;
}

.simple {
    background: #edf8fc;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin-top: 0.7rem;
}

/* ---------- Warning ---------- */

.warning {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    line-height: 1.6;
}

.disclaimer {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    line-height: 1.6;
}

/* ---------- Metrics ---------- */

.metric-box {
    background: #f8fafc;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #e2e8f0;
}

.metric-number {
    font-size: 1.7rem;
    font-weight: 750;
    color: #0f2540;
}

.metric-name {
    color: #64748b;
    font-size: 0.78rem;
}

/* ---------- Buttons ---------- */

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 2.6rem;
}

/* ---------- Upload ---------- */

div[data-testid="stFileUploader"] {
    border: 2px dashed #2a7ba8;
    border-radius: 14px;
    padding: 1rem;
}

/* ---------- Hide Streamlit menu/footer ---------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
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


try:

    model = load_model()

    conv_layer_model, base_model = build_gradcam_extractor(
        model
    )

    MODEL_READY = True
    MODEL_ERROR = None

except Exception as e:

    model = None
    conv_layer_model = None
    base_model = None

    MODEL_READY = False
    MODEL_ERROR = str(e)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "results" not in st.session_state:
    st.session_state.results = []

if "demo_selected" not in st.session_state:
    st.session_state.demo_selected = None


# ============================================================
# NAVIGATION
# ============================================================

def navigate(page_name):

    st.session_state.page = page_name
    st.rerun()


# ============================================================
# TOP NAVIGATION
# ============================================================

top1, top2, top3, top4, top5, top6 = st.columns(
    [2.2, 1.3, 1.3, 1.4, 1.5, 1.6]
)

with top1:
    st.markdown(
        '<div class="nav-title">🩻 CT Image Quality Flagger</div>',
        unsafe_allow_html=True
    )

with top2:
    if st.button("Home", use_container_width=True):
        navigate("Home")

with top3:
    if st.button("Analyze CT", use_container_width=True):
        navigate("Analyze CT")

with top4:
    if st.button("Your Report", use_container_width=True):
        navigate("Analysis Report")

with top5:
    if st.button("How AI Works", use_container_width=True):
        navigate("How It Works")

with top6:
    if st.button("Project & Results", use_container_width=True):
        navigate("Project & Results")


st.markdown("---")


# ============================================================
# DICOM PROCESSING
# ============================================================

def dicom_to_array(
    dicom_bytes,
    window_center=WINDOW_CENTER,
    window_width=WINDOW_WIDTH
):

    ds = pydicom.dcmread(
        io.BytesIO(dicom_bytes)
    )

    pixel_array = ds.pixel_array.astype(
        np.float32
    )

    slope = float(
        getattr(ds, "RescaleSlope", 1.0)
    )

    intercept = float(
        getattr(ds, "RescaleIntercept", 0.0)
    )

    hu = (
        pixel_array * slope
        + intercept
    )

    lower = (
        window_center
        - window_width / 2
    )

    upper = (
        window_center
        + window_width / 2
    )

    hu = np.clip(
        hu,
        lower,
        upper
    )

    hu = (
        (hu - lower)
        / (upper - lower)
        * 255
    ).astype(np.uint8)

    image = (
        Image.fromarray(hu)
        .convert("RGB")
        .resize(INPUT_SIZE)
    )

    metadata = {}

    fields = [
        ("KVP", "Tube voltage"),
        ("XRayTubeCurrent", "Tube current"),
        ("Exposure", "Exposure"),
        ("SliceThickness", "Slice thickness"),
        ("BodyPartExamined", "Body part"),
        ("SeriesDescription", "Series"),
        ("Manufacturer", "Manufacturer"),
        ("ManufacturerModelName", "Scanner model")
    ]

    for tag, label in fields:

        if hasattr(ds, tag):

            value = getattr(ds, tag)

            if value is not None:

                metadata[label] = str(value)

    return image, metadata


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image):

    image = (
        image
        .convert("RGB")
        .resize(INPUT_SIZE)
    )

    array = np.array(
        image
    ).astype(np.float32)

    batch = np.expand_dims(
        array,
        axis=0
    )

    processed = preprocess_input(
        batch.copy()
    )

    return array, processed


# ============================================================
# TTA
# ============================================================

def get_score(processed_image):

    predictions = []

    # Original
    prediction = model.predict(
        processed_image,
        verbose=0
    )[0][0]

    predictions.append(
        float(prediction)
    )

    # Flip
    flipped = np.flip(
        processed_image,
        axis=2
    )

    prediction = model.predict(
        flipped,
        verbose=0
    )[0][0]

    predictions.append(
        float(prediction)
    )

    # Rotation
    image = processed_image[0]

    center = (112, 112)

    for angle in [-5, 5]:

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            image,
            matrix,
            INPUT_SIZE,
            borderMode=cv2.BORDER_REFLECT
        )

        rotated = np.expand_dims(
            rotated,
            axis=0
        )

        prediction = model.predict(
            rotated,
            verbose=0
        )[0][0]

        predictions.append(
            float(prediction)
        )

    # Zoom
    crop = image[
        11:213,
        11:213
    ]

    zoomed = cv2.resize(
        crop,
        INPUT_SIZE
    )

    zoomed = np.expand_dims(
        zoomed,
        axis=0
    )

    prediction = model.predict(
        zoomed,
        verbose=0
    )[0][0]

    predictions.append(
        float(prediction)
    )

    return float(
        np.mean(predictions)
    )


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam(
    image_array,
    processed_image
):

    with tf.GradientTape() as tape:

        conv_output = conv_layer_model(
            processed_image
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

    pooled = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv = conv_output[0]

    heatmap = (
        conv
        @ pooled[..., tf.newaxis]
    )

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = (
        heatmap
        / (
            tf.reduce_max(heatmap)
            + 1e-8
        )
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
        np.uint8(
            255 * heatmap
        ),
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image_array.astype(np.uint8),
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return overlay


# ============================================================
# ANALYZE IMAGE
# ============================================================

def analyze_image(image):

    image_array, processed = prepare_image(
        image
    )

    score = get_score(
        processed
    )

    gradcam = make_gradcam(
        image_array,
        processed
    )

    return (
        image_array,
        gradcam,
        score
    )


# ============================================================
# RESULT INTERPRETATION
# ============================================================

def interpret_score(score):

    if score >= 0.50:

        return (
            "Higher-risk pattern",
            "🔴",
            "#dc2626",
            "Review recommended",
            """
            The model produced a relatively high
            quality-risk score. This image should
            receive additional professional review.
            """
        )

    if score >= THRESHOLD:

        return (
            "Borderline risk",
            "🟠",
            "#ea580c",
            "Review recommended",
            """
            The score is at or above the project's
            review threshold. Additional review is
            recommended.
            """
        )

    if score >= 0.15:

        return (
            "Lower-risk pattern",
            "🟡",
            "#ca8a04",
            "No automatic flag",
            """
            The score is below the project's review
            threshold, although some uncertainty
            remains.
            """
        )

    return (
        "Low-risk pattern",
        "🟢",
        "#16a34a",
        "No automatic flag",
        """
        The model produced a relatively low
        quality-risk score.
        """
    )


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    name,
    image_array,
    gradcam,
    score,
    metadata
):

    # Avoid duplicate entries
    existing = [
        r["name"]
        for r in st.session_state.results
    ]

    if name in existing:
        return

    st.session_state.results.append({

        "name": name,

        "image": image_array,

        "gradcam": gradcam,

        "score": score,

        "metadata": metadata,

        "time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    })


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero">

        <div class="hero-small">
            MEDICAL IMAGING TECHNOLOGY × AI
        </div>

        <h1>
            CT Image Quality Flagger
        </h1>

        <p>
            An AI-assisted research prototype exploring
            how image-quality assessment can support
            CT dose-optimization workflows.
        </p>

    </div>
    """, unsafe_allow_html=True)

    # Intro
    col1, col2 = st.columns(
        [1.45, 1]
    )

    with col1:

        st.markdown(
            "## Hi, I'm Zainab 👋"
        )

        st.write(
            """
            I'm a Medical Imaging Technology student interested
            in how artificial intelligence can be integrated into
            real medical-imaging workflows.

            I developed this prototype to explore a specific
            problem in CT: **how can we reduce radiation exposure
            while still paying attention to image quality?**

            Instead of building a disease-diagnosis system,
            this project focuses on an imaging-technology problem:
            identifying CT images showing patterns associated
            with dose-related image-quality degradation.
            """
        )

        st.markdown("""
        <div class="simple">

        <b>My goal:</b>

        Explore AI as a supportive tool for Medical Imaging
        Technologists — not as a replacement for professional
        judgment.

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">

        <h3>What happens inside?</h3>

        <p>
        <b>1. CT image</b><br>
        ↓
        </p>

        <p>
        <b>2. Image preprocessing</b><br>
        The CT image is prepared in the same format
        expected by the model.
        </p>

        <p>
        <b>3. VGG16 model</b><br>
        The model generates a quality-risk score.
        </p>

        <p>
        <b>4. Grad-CAM</b><br>
        A heatmap provides a visual explanation
        of the model's prediction.
        </p>

        <p>
        <b>5. Review signal</b><br>
        The score is compared with the project's
        threshold.
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    if st.button(
        "🔬 Start CT Analysis",
        use_container_width=True,
        type="primary"
    ):

        navigate("Analyze CT")

    st.markdown("---")

    st.markdown(
        "### Why this is more than a simple classifier"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="card">

        <h3>🧠 AI model</h3>

        <p>
        VGG16 transfer learning is used to
        generate a continuous image-quality
        risk score.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="card">

        <h3>👁️ Explainability</h3>

        <p>
        Grad-CAM provides a visual indication
        of the image regions contributing to
        the model prediction.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="card">

        <h3>🏥 Clinical context</h3>

        <p>
        The project considers dose, image quality,
        workflow and the role of Medical Imaging
        Technologists.
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="disclaimer">

    <b>Research prototype — not a clinical device</b>

    <br><br>

    This application is intended for educational and research
    demonstration only. It does not diagnose disease, determine
    clinical diagnostic adequacy, or replace a radiologist's or
    Medical Imaging Technologist's professional judgment.

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ANALYZE CT
# ============================================================

elif st.session_state.page == "Analyze CT":

    st.title("🔬 Analyze a CT Image")

    st.write(
        """
        You can either explore one of the demonstration cases
        or upload your own CT image.
        """
    )

    # ========================================================
    # DEMO CASES
    # ========================================================

    st.markdown(
        "## 🧪 Try a demonstration case"
    )

    st.caption(
        "These images are included only to demonstrate how "
        "the application works."
    )

    sample_dir = "sample_images"

    demo_cases = [

        (
            "Demo Case 1",
            "sample_acceptable_1.png"
        ),

        (
            "Demo Case 2",
            "sample_acceptable_2.png"
        ),

        (
            "Demo Case 3",
            "sample_flagged_1.png"
        ),

        (
            "Demo Case 4",
            "sample_flagged_2.png"
        )

    ]

    demo_cols = st.columns(4)

    for i, (
        title,
        filename
    ) in enumerate(demo_cases):

        path = os.path.join(
            sample_dir,
            filename
        )

        with demo_cols[i]:

            if os.path.exists(path):

                image = Image.open(
                    path
                ).convert("RGB")

                st.image(
                    image,
                    caption=title,
                    use_container_width=True
                )

                if st.button(
                    "Analyze",
                    key=f"demo_{i}",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing demonstration image..."
                    ):

                        arr, gradcam, score = analyze_image(
                            image
                        )

                    save_result(
                        title,
                        arr,
                        gradcam,
                        score,
                        {}
                    )

                    st.session_state.demo_selected = title

                    st.rerun()

            else:

                st.info(
                    f"{title}\n\n"
                    "Add the sample image to "
                    f"`sample_images/{filename}`."
                )

    st.markdown("---")

    # ========================================================
    # UPLOAD
    # ========================================================

    st.markdown(
        "## 📤 Upload your CT image"
    )

    st.write(
        """
        Supported formats:

        **DICOM (.dcm)** — preferred for CT images because
        it can contain imaging and acquisition information.

        **PNG / JPG / JPEG** — useful for demonstration images.
        """
    )

    uploaded_files = st.file_uploader(
        "Choose CT image(s)",
        type=[
            "dcm",
            "png",
            "jpg",
            "jpeg"
        ],
        accept_multiple_files=True,
        help=(
            "Large DICOM files are supported. "
            "The application upload limit is configured separately."
        )
    )

    if uploaded_files:

        for file in uploaded_files:

            already_exists = any(
                r["name"] == file.name
                for r in st.session_state.results
            )

            if already_exists:
                continue

            try:

                with st.spinner(
                    f"Analyzing {file.name}..."
                ):

                    if file.name.lower().endswith(
                        ".dcm"
                    ):

                        image, metadata = dicom_to_array(
                            file.read()
                        )

                    else:

                        image = Image.open(
                            file
                        ).convert("RGB")

                        metadata = {}

                    arr, gradcam, score = analyze_image(
                        image
                    )

                save_result(
                    file.name,
                    arr,
                    gradcam,
                    score,
                    metadata
                )

                st.success(
                    f"{file.name} analyzed successfully."
                )

            except Exception as e:

                st.error(
                    f"Could not analyze {file.name}: {e}"
                )

    # ========================================================
    # SHOW RESULTS
    # ========================================================

    if st.session_state.results:

        st.markdown("---")

        st.markdown(
            "## 📊 Your result"
        )

        # Most recent result
        result = st.session_state.results[-1]

        (
            category,
            emoji,
            color,
            recommendation,
            explanation
        ) = interpret_score(
            result["score"]
        )

        st.markdown(
            f"""
            <div class="result-box"
                 style="border-left:6px solid {color};">

                <div class="result-label">
                    MODEL ASSESSMENT
                </div>

                <h2>
                    {emoji} {category}
                </h2>

                <div class="score">
                    {result["score"]:.3f}
                </div>

                <div class="result-label">
                    Quality-risk score
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        r1, r2 = st.columns(2)

        with r1:

            st.markdown(
                f"""
                <div class="explain">

                <b>What does the result mean?</b>

                <br><br>

                {explanation}

                <div class="simple">

                <b>Simple explanation:</b><br>

                The model is giving us a signal about
                how strongly this image resembles the
                project's flagged pattern.

                It is <b>not</b> saying that the patient
                has a disease.

                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with r2:

            st.markdown(
                f"""
                <div class="explain">

                <b>Project threshold: {THRESHOLD:.2f}</b>

                <br><br>

                A score of {THRESHOLD:.2f} or higher is
                considered a review flag in this project.

                <div class="simple">

                <b>Simple explanation:</b><br>

                The threshold is simply the point where
                the model's score triggers an additional
                review signal.

                It is <b>not a universal clinical cutoff</b>.

                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            "### 👁️ See what the model focused on"
        )

        image_col1, image_col2 = st.columns(2)

        with image_col1:

            st.image(
                result["image"],
                caption="Original CT image",
                use_container_width=True
            )

        with image_col2:

            st.image(
                result["gradcam"],
                caption="Grad-CAM visualization",
                use_container_width=True
            )

        with st.expander(
            "What is Grad-CAM?"
        ):

            st.write(
                """
                Grad-CAM stands for Gradient-weighted
                Class Activation Mapping.

                It creates a heatmap showing image regions
                that contributed to the model's prediction.

                **In simple words:** it helps us see
                where the model was looking when it made
                its prediction.

                The heatmap is an explanation aid. It does
                not prove that the highlighted region is
                clinically abnormal.
                """
            )

        if result["metadata"]:

            with st.expander(
                "📋 DICOM information"
            ):

                st.dataframe(
                    pd.DataFrame(
                        list(
                            result["metadata"].items()
                        ),
                        columns=[
                            "Parameter",
                            "Value"
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

        st.markdown("""
        <div class="warning">

        ⚠️ <b>Important:</b>

        This result is generated by an experimental AI model.
        It should not be used to accept, reject, repeat or
        modify a clinical CT examination.

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "🗑️ Clear analysis"
        ):

            st.session_state.results = []

            st.rerun()


# ============================================================
# REPORT
# ============================================================

elif st.session_state.page == "Analysis Report":

    st.title(
        "📋 Analysis Report"
    )

    if not st.session_state.results:

        st.info(
            "No images have been analyzed yet."
        )

        if st.button(
            "🔬 Analyze a CT"
        ):

            navigate("Analyze CT")

    else:

        results = st.session_state.results

        scores = [
            r["score"]
            for r in results
        ]

        flagged = sum(
            s >= THRESHOLD
            for s in scores
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Images analyzed",
                len(results)
            )

        with c2:
            st.metric(
                "Review flags",
                flagged
            )

        with c3:
            st.metric(
                "Average score",
                f"{np.mean(scores):.3f}"
            )

        rows = []

        for r in results:

            (
                category,
                emoji,
                color,
                recommendation,
                explanation
            ) = interpret_score(
                r["score"]
            )

            rows.append({

                "File": r["name"],

                "Risk score": round(
                    r["score"],
                    3
                ),

                "Assessment": category,

                "Recommendation": recommendation

            })

        report_df = pd.DataFrame(
            rows
        )

        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True
        )

        csv = report_df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "📥 Download CSV report",
            csv,
            "ct_quality_report.csv",
            "text/csv"
        )


# ============================================================
# HOW AI WORKS
# ============================================================

elif st.session_state.page == "How It Works":

    st.title(
        "🧠 How the AI Works"
    )

    st.write(
        """
        Don't worry if terms like VGG16, TTA or Grad-CAM
        are new. Each one is explained in simple language.
        """
    )

    # VGG16
    with st.expander(
        "1. VGG16 — the model used in this project",
        expanded=True
    ):

        st.markdown("""
        **VGG16** is a type of deep-learning model designed
        to recognize visual patterns.

        This project uses **transfer learning**.

        **Transfer learning means:** starting with a model
        that has already learned useful visual features and
        adapting it to a new problem.

        **In simple words:** instead of teaching the model
        everything from zero, we reuse useful visual patterns
        and adapt the model for CT image-quality assessment.
        """)

    # Preprocessing
    with st.expander(
        "2. CT preprocessing"
    ):

        st.markdown("""
        A DICOM CT image contains numerical pixel information.
        For CT images, those values can be converted into
        **Hounsfield Units (HU)**.

        The project then uses a CT window:

        **Window Level = 40**  
        **Window Width = 400**

        Finally, the image is resized to **224 × 224 pixels**
        for the VGG16 model.

        **In simple words:** the CT image is converted and
        prepared into the same visual format that the model
        expects.
        """)

    # TTA
    with st.expander(
        "3. Test-Time Augmentation (TTA)"
    ):

        st.markdown("""
        **Test-Time Augmentation (TTA)** means showing the
        same image to the model in several slightly modified
        forms.

        This project uses five views:

        • Original image  
        • Horizontal flip  
        • −5° rotation  
        • +5° rotation  
        • Small crop/zoom  

        The five predictions are averaged.

        **In simple words:** we ask the model to look at
        slightly different versions of the same image and
        combine its answers.
        """)

    # GradCAM
    with st.expander(
        "4. Grad-CAM — how we visualize the model's attention"
    ):

        st.markdown("""
        **Grad-CAM** creates a heatmap showing regions that
        contributed to the model's prediction.

        **In simple words:** it gives us a visual clue about
        where the model was looking.

        It does NOT mean:

        ❌ the highlighted area is definitely abnormal  
        ❌ the model understands the image like a radiologist  
        ❌ the heatmap proves causation  

        It is an **interpretability tool**.
        """)

    # Threshold
    with st.expander(
        "5. Threshold — when does the app flag an image?"
    ):

        st.markdown(f"""
        The model produces a score rather than simply
        saying "yes" or "no."

        The project uses a threshold of **{THRESHOLD:.2f}**.

        **Score below {THRESHOLD:.2f}:**
        no automatic review flag.

        **Score ≥ {THRESHOLD:.2f}:**
        review flag.

        **In simple words:** the threshold is the cutoff
        that turns the model's numerical score into a
        review signal.

        It should not be considered a universal clinical
        threshold.
        """)

    # Labels
    with st.expander(
        "6. How were the labels created?"
    ):

        st.markdown("""
        The model was trained using paired full-dose and
        low-dose CT data.

        Quality-related labels were derived from
        **image-noise measurements** comparing the paired
        scans.

        **Important:** these labels were not assigned by
        radiologists.

        **In simple words:** the project uses measurable
        image-noise differences as a quality-related proxy.
        That is not the same thing as asking a radiologist
        whether an image is diagnostically acceptable.
        """)

    # Testing
    with st.expander(
        "7. How was the model tested?"
    ):

        st.markdown("""
        The final evaluation used **3 fully held-out patients**
        containing **1,058 CT slices**.

        **Held-out patient** means the patient was kept separate
        from model development.

        This matters because images from the same patient can
        be very similar.

        **In simple words:** the model was tested on patients
        it had not seen during development.
        """)

    # Limitations
    with st.expander(
        "8. What are the limitations?"
    ):

        st.markdown("""
        • Only 21 patients were used for development.

        • The data come from one public dataset.

        • External validation is still required.

        • The labels are noise-derived rather than
          radiologist-confirmed.

        • The model evaluates image-quality patterns,
          not disease.

        • It has not been clinically validated.

        • It is not intended for patient-care decisions.
        """)


# ============================================================
# PROJECT & RESULTS
# ============================================================

elif st.session_state.page == "Project & Results":

    st.title(
        "📊 Project & Results"
    )

    st.markdown(
        "### Model performance"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    metrics = [
        ("85%", "Recall"),
        ("54%", "Precision"),
        ("0.66", "F1-score"),
        ("0.839", "ROC-AUC"),
        ("3.3%", "Full-dose FPR")
    ]

    for column, (
        value,
        label
    ) in zip(
        [c1, c2, c3, c4, c5],
        metrics
    ):

        with column:

            st.markdown(
                f"""
                <div class="metric-box">

                <div class="metric-number">
                    {value}
                </div>

                <div class="metric-name">
                    {label}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.markdown(
        "### What problem is this project addressing?"
    )

    st.write(
        """
        CT dose optimization is a balance between radiation
        exposure and image quality.

        The project explores whether AI can provide an
        additional signal for identifying images showing
        patterns associated with dose-related quality
        degradation.
        """
    )

    st.markdown(
        "### Why Medical Imaging Technology?"
    )

    st.write(
        """
        CT optimization is directly connected to the work of
        Medical Imaging Technologists, who operate imaging
        systems, implement protocols, position patients and
        monitor image quality.

        For that reason, an AI system for CT should not be
        designed only as an algorithm. It should also consider
        how the technology fits into the real imaging workflow.
        """
    )

    st.markdown(
        "### 🔬 Research limitations"
    )

    st.warning(
        """
        These performance numbers come from this project's
        evaluation data. They do not establish clinical
        effectiveness.

        Larger datasets, external validation, multiple scanners,
        different institutions and clinically meaningful
        reference standards would be required before clinical
        deployment could be considered.
        """
    )

    # ROC
    roc_path = os.path.join(
        "assets",
        "roc_curve.png"
    )

    tsne_path = os.path.join(
        "assets",
        "tsne_features.png"
    )

    if os.path.exists(roc_path):

        st.markdown(
            "### ROC Curve"
        )

        st.image(
            roc_path,
            use_container_width=True
        )

    if os.path.exists(tsne_path):

        st.markdown(
            "### Feature Visualization"
        )

        st.image(
            tsne_path,
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "CT Image Quality Flagger • "
    "Built by Zainab Fatima • "
    "Educational/research prototype — not for clinical use."
)
