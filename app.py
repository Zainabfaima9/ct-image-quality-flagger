import os
import io
import cv2
import numpy as np
import pandas as pd
import pydicom
import streamlit as st
import tensorflow as tf

from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.vgg16 import preprocess_input


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CT Image Quality Flagger",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

HF_REPO = "zainabfatima9/ct-image-quality-flagger"
MODEL_FILENAME = "ct_quality_model_21patients_v2.h5"

IMAGE_SIZE = (224, 224)

# Project-specific review threshold
THRESHOLD = 0.25

GRADCAM_LAYER = "block4_conv3"

WINDOW_CENTER = 40
WINDOW_WIDTH = 400


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "results" not in st.session_state:
    st.session_state.results = []

if "selected_result" not in st.session_state:
    st.session_state.selected_result = None

if "analyze_mode" not in st.session_state:
    st.session_state.analyze_mode = None


def navigate(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>

    :root {
        --navy: #102a43;
        --blue: #247ba0;
        --blue-light: #eaf5f9;
        --ink: #172b4d;
        --muted: #64748b;
        --line: #dbe5ec;
        --bg: #f7fafc;
        --green: #237a57;
        --orange: #b45309;
        --red: #b42318;
    }

    .stApp {
        background: #f7fafc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.1rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ========================================================
       BRAND
       ======================================================== */

    .brand-name {
        font-size: 1.2rem;
        font-weight: 800;
        color: #102a43;
        margin-bottom: 0;
    }

    .brand-subtitle {
        color: #64748b;
        font-size: .76rem;
        margin-top: .15rem;
    }

    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        background: linear-gradient(
            135deg,
            #0f2942 0%,
            #174e73 55%,
            #2b82a8 100%
        );
        border-radius: 24px;
        padding: 3.2rem 3.2rem;
        color: white;
        margin: .8rem 0 1.6rem 0;
        box-shadow: 0 16px 38px rgba(16,42,67,.15);
    }

    .hero-kicker {
        font-size: .72rem;
        letter-spacing: .12em;
        font-weight: 700;
        opacity: .82;
        margin-bottom: 1rem;
    }

    .hero h1 {
        font-size: clamp(2.1rem, 5vw, 3.45rem);
        line-height: 1.05;
        margin: 0 0 .85rem 0;
        color: white;
        font-weight: 800;
    }

    .hero p {
        max-width: 780px;
        font-size: 1.03rem;
        line-height: 1.65;
        margin: 0;
        color: rgba(255,255,255,.91);
    }

    /* ========================================================
       CARDS
       ======================================================== */

    .card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.35rem;
        box-shadow: 0 4px 16px rgba(15,23,42,.035);
        height: 100%;
    }

    .card-title {
        color: #102a43;
        font-weight: 750;
        font-size: 1.03rem;
        margin-bottom: .45rem;
    }

    .card-text {
        color: #526174;
        line-height: 1.62;
        font-size: .92rem;
    }

    .eyebrow {
        color: #247ba0;
        font-size: .74rem;
        font-weight: 800;
        letter-spacing: .1em;
        text-transform: uppercase;
    }

    /* ========================================================
       SCORE
       ======================================================== */

    .score {
        font-size: 3.25rem;
        line-height: 1;
        font-weight: 850;
        color: #102a43;
    }

    .score-label {
        color: #64748b;
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .06em;
    }

    /* ========================================================
       EXPLANATION
       ======================================================== */

    .explain {
        background: #eef7fa;
        border-left: 4px solid #247ba0;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        color: #334e68;
        line-height: 1.65;
    }

    .action-box {
        background: white;
        border: 1px solid #dbe5ec;
        border-radius: 16px;
        padding: 1.25rem;
        margin-top: .8rem;
    }

    .action-title {
        color: #102a43;
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: .55rem;
    }

    .action-text {
        color: #526174;
        line-height: 1.65;
        font-size: .92rem;
    }

    .limitation-box {
        background: #fffaf0;
        border: 1px solid #f1dfb5;
        border-radius: 16px;
        padding: 1.3rem;
        color: #59451c;
        line-height: 1.65;
    }

    .research-box {
        background: #f2f7fb;
        border: 1px solid #d7e5ef;
        border-radius: 16px;
        padding: 1.3rem;
        color: #334e68;
        line-height: 1.65;
    }

    /* ========================================================
       OPTION CARDS
       ======================================================== */

    .option-card {
        background: white;
        border: 1px solid #dce6ed;
        border-radius: 20px;
        padding: 1.65rem;
        min-height: 235px;
        box-shadow: 0 5px 18px rgba(15,23,42,.04);
    }

    .option-icon {
        font-size: 2rem;
        margin-bottom: .7rem;
    }

    .option-title {
        color: #102a43;
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: .55rem;
    }

    .option-text {
        color: #526174;
        line-height: 1.65;
        font-size: .92rem;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 10px;
        min-height: 2.65rem;
        font-weight: 650;
    }

    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background: white;
        border: 1.5px dashed #8db9ca;
        border-radius: 16px;
        padding: .6rem;
    }

    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 700px) {

        .block-container {
            padding-left: .85rem;
            padding-right: .85rem;
        }

        .hero {
            padding: 2rem 1.35rem;
            border-radius: 18px;
        }

        .hero h1 {
            font-size: 2.05rem;
        }

        .hero p {
            font-size: .92rem;
        }

        .hero-kicker {
            font-size: .64rem;
        }

        .brand-name {
            font-size: 1.05rem;
        }

        .option-card {
            min-height: auto;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP NAVIGATION
# ============================================================

brand, nav = st.columns(
    [1.25, 2.75],
    vertical_alignment="center",
)

with brand:

    st.markdown(
        '<div class="brand-name">🩻 CT Image Quality Flagger</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'AI-assisted CT image-quality research prototype'
        '</div>',
        unsafe_allow_html=True,
    )


with nav:

    n1, n2, n3, n4 = st.columns(4)

    with n1:
        if st.button(
            "Home",
            use_container_width=True,
        ):
            st.session_state.analyze_mode = None
            navigate("Home")

    with n2:
        if st.button(
            "Analyze",
            use_container_width=True,
        ):
            navigate("Analyze")

    with n3:
        if st.button(
            "Results",
            use_container_width=True,
        ):
            navigate("Results")

    with n4:
        if st.button(
            "Learn",
            use_container_width=True,
        ):
            navigate("Learn")


st.divider()


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILENAME,
    )

    return tf.keras.models.load_model(path)


@st.cache_resource
def build_gradcam_extractor(model):

    base = model.layers[0]

    extractor = tf.keras.Model(
        base.input,
        base.get_layer(GRADCAM_LAYER).output,
    )

    return extractor, base


try:

    model = load_model()

    conv_layer_model, base_model = (
        build_gradcam_extractor(model)
    )

    MODEL_READY = True
    MODEL_ERROR = None

except Exception as exc:

    MODEL_READY = False
    MODEL_ERROR = str(exc)


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def dicom_to_image(data):

    ds = pydicom.dcmread(
        io.BytesIO(data)
    )

    pixels = ds.pixel_array.astype(
        np.float32
    )

    slope = float(
        getattr(
            ds,
            "RescaleSlope",
            1,
        )
    )

    intercept = float(
        getattr(
            ds,
            "RescaleIntercept",
            0,
        )
    )

    hu = (
        pixels * slope
        + intercept
    )

    low = (
        WINDOW_CENTER
        - WINDOW_WIDTH / 2
    )

    high = (
        WINDOW_CENTER
        + WINDOW_WIDTH / 2
    )

    hu = np.clip(
        hu,
        low,
        high,
    )

    hu = (
        (hu - low)
        / (high - low)
        * 255
    )

    hu = np.clip(
        hu,
        0,
        255,
    ).astype(
        np.uint8
    )

    image = (
        Image.fromarray(hu)
        .convert("RGB")
        .resize(IMAGE_SIZE)
    )

    metadata = {}

    fields = [
        ("KVP", "Tube voltage"),
        ("XRayTubeCurrent", "Tube current"),
        ("Exposure", "Exposure"),
        ("SliceThickness", "Slice thickness"),
        ("BodyPartExamined", "Body part"),
        ("Manufacturer", "Scanner manufacturer"),
        ("ManufacturerModelName", "Scanner model"),
        ("SeriesDescription", "Series description"),
    ]

    for tag, label in fields:

        if hasattr(ds, tag):

            metadata[label] = str(
                getattr(
                    ds,
                    tag,
                )
            )

    return image, metadata


def prepare_image(image):

    image = (
        image
        .convert("RGB")
        .resize(IMAGE_SIZE)
    )

    arr = np.array(
        image
    ).astype(
        np.uint8
    )

    model_input = np.expand_dims(
        arr.astype(
            np.float32
        ),
        axis=0,
    )

    model_input = preprocess_input(
        model_input
    )

    return (
        arr,
        model_input,
    )


def get_score(model_input):

    preds = []

    # Original
    preds.append(
        float(
            model.predict(
                model_input,
                verbose=0,
            )[0][0]
        )
    )

    # Horizontal flip
    flipped = np.flip(
        model_input,
        axis=2,
    )

    preds.append(
        float(
            model.predict(
                flipped,
                verbose=0,
            )[0][0]
        )
    )

    # Small rotations
    image = model_input[0]

    center = (
        112,
        112,
    )

    for angle in (-5, 5):

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )

        rotated = cv2.warpAffine(
            image,
            matrix,
            IMAGE_SIZE,
            borderMode=cv2.BORDER_REFLECT,
        )

        rotated = np.expand_dims(
            rotated,
            axis=0,
        )

        preds.append(
            float(
                model.predict(
                    rotated,
                    verbose=0,
                )[0][0]
            )
        )

    # Crop / zoom
    crop = image[
        11:213,
        11:213,
    ]

    zoomed = cv2.resize(
        crop,
        IMAGE_SIZE,
    )

    zoomed = np.expand_dims(
        zoomed,
        axis=0,
    )

    preds.append(
        float(
            model.predict(
                zoomed,
                verbose=0,
            )[0][0]
        )
    )

    return float(
        np.mean(preds)
    )


def make_gradcam(
    image_array,
    model_input,
):

    with tf.GradientTape() as tape:

        conv_output = (
            conv_layer_model(
                model_input
            )
        )

        tape.watch(
            conv_output
        )

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

    grads = tape.gradient(
        loss,
        conv_output,
    )

    pooled = tf.reduce_mean(
        grads,
        axis=(0, 1, 2),
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
        0,
    )

    heatmap /= (
        tf.reduce_max(
            heatmap
        )
        + 1e-8
    )

    heatmap = cv2.resize(
        heatmap.numpy(),
        IMAGE_SIZE,
        interpolation=cv2.INTER_CUBIC,
    )

    heatmap = np.clip(
        heatmap,
        0,
        1,
    )

    heatmap_color = cv2.applyColorMap(
        (
            heatmap * 255
        ).astype(
            np.uint8
        ),
        cv2.COLORMAP_JET,
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB,
    )

    image_array = np.clip(
        image_array,
        0,
        255,
    ).astype(
        np.uint8
    )

    overlay = cv2.addWeighted(
        image_array,
        0.60,
        heatmap_color,
        0.40,
        0,
    )

    return np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )


def analyze(image):

    arr, model_input = (
        prepare_image(image)
    )

    score = get_score(
        model_input
    )

    gradcam = make_gradcam(
        arr,
        model_input,
    )

    return (
        arr,
        gradcam,
        score,
    )


# ============================================================
# RESULT INTERPRETATION
# ============================================================

def interpretation(score):

    if score >= 0.50:

        return {
            "level": "Higher quality-risk signal",
            "emoji": "🔴",
            "status": "Review recommended",
            "summary": (
                "The model produced a relatively high "
                "quality-risk score."
            ),
            "action": (
                "Prioritize manual image-quality assessment. "
                "Review image noise, motion, artifacts and "
                "anatomical visibility. If the image appears "
                "inadequate for the intended examination, follow "
                "the department's established quality-control "
                "pathway or seek appropriate senior/radiologist review."
            ),
        }

    if score >= THRESHOLD:

        return {
            "level": "Borderline quality-risk signal",
            "emoji": "🟠",
            "status": "Additional review recommended",
            "summary": (
                "The model score has reached the project's "
                "review threshold."
            ),
            "action": (
                "Perform an additional image-quality review. "
                "Pay attention to noise, motion, artifacts and "
                "whether the required anatomy remains adequately "
                "visible. Use professional judgment and follow "
                "local quality-control procedures."
            ),
        }

    if score >= 0.15:

        return {
            "level": "Lower quality-risk signal",
            "emoji": "🟡",
            "status": "No automatic review flag",
            "summary": (
                "The score is below the project's review threshold, "
                "although the model output still contains uncertainty."
            ),
            "action": (
                "Continue routine image-quality assessment. "
                "Check image noise, motion, artifacts, anatomical "
                "coverage and whether the image is suitable for "
                "the intended examination."
            ),
        }

    return {
        "level": "Low quality-risk signal",
        "emoji": "🟢",
        "status": "No automatic review flag",
        "summary": (
            "The model produced a relatively low "
            "quality-risk score."
        ),
        "action": (
            "Continue the normal image-quality workflow. "
            "The AI result does not replace professional "
            "assessment of the examination."
        ),
    }


def add_result(
    name,
    arr,
    gradcam,
    score,
    metadata,
):

    names = [
        r["name"]
        for r in st.session_state.results
    ]

    if name not in names:

        st.session_state.results.append(
            {
                "name": name,
                "image": arr,
                "gradcam": gradcam,
                "score": score,
                "metadata": metadata,
            }
        )

    st.session_state.selected_result = name


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                MEDICAL IMAGING TECHNOLOGY × ARTIFICIAL INTELLIGENCE
            </div>

            <h1>
                CT Image Quality Flagger
            </h1>

            <p>
                An AI-assisted research prototype exploring how
                image-quality assessment could support CT dose-
                optimization workflows without replacing
                professional judgment.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1.45, 1],
        gap="large",
    )

    with left:

        st.markdown(
            '<div class="eyebrow">THE QUESTION</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### What happens when CT dose goes down?"
        )

        st.write(
            """
            Reducing radiation exposure can benefit patient safety,
            but reducing dose can also increase image noise and
            affect image quality.

            This project explores a simple question:

            **Can AI provide an additional signal when an image
            begins to resemble a lower-quality pattern?**
            """
        )

        st.markdown(
            """
            <div class="research-box">

            <b>This prototype does not decide whether a scan is
            clinically acceptable.</b>

            It produces a model-generated quality-risk signal that
            can be considered alongside normal professional
            image-quality assessment.

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "🔬 Try the CT Analyzer",
            type="primary",
            use_container_width=True,
        ):

            st.session_state.analyze_mode = None
            navigate("Analyze")

    with right:

        st.markdown(
            '<div class="eyebrow">IN SIMPLE TERMS</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### What does the tool actually do?"
        )

        st.markdown(
            """
            **1. 🩻 Takes a CT image**  
            DICOM, PNG or JPG can be analyzed.

            **2. 🧠 Runs the AI model**  
            A VGG16-based model generates a quality-risk score.

            **3. 🚩 Checks the project threshold**  
            A score ≥ 0.25 creates a review signal.

            **4. 👩‍⚕️ Supports human review**  
            The technologist still assesses the actual image.
            """
        )

    st.write("")

    st.markdown("### Why could this be useful?")

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            🎯 Consistency
            </div>

            <div class="card-text">
            Human image-quality judgments can vary with experience,
            workload and attention. A model can apply the same
            computational criterion consistently.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            📊 Scale
            </div>

            <div class="card-text">
            A CT examination may contain hundreds of images.
            Automated screening could help identify cases that
            deserve closer review.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            🔎 Subtle cases
            </div>

            <div class="card-text">
            The longer-term research goal is not obvious noise.
            Future validation should examine smaller and more
            realistic dose-related changes in image quality.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown("### One important limitation")

    st.markdown(
        """
        <div class="limitation-box">

        <b>Why can the demo cases look obviously noisy?</b>

        The CT data used for this proof-of-concept contains paired
        full-dose and low-dose images with relatively strong
        differences in image noise. That makes the distinction
        easier for a first-stage research experiment.

        <br><br>

        In routine clinical practice, dose reductions can be much
        more subtle. The important future question is whether this
        approach remains useful when image-quality differences are
        difficult to recognize visually.

        <br><br>

        <b>This is a limitation, not a hidden result.</b>

        The prototype demonstrates that the model can learn a
        quality-related distinction in the study data. It does
        <b>not</b> establish reliable performance across different
        patients, scanners, institutions or realistic clinical
        dose-reduction settings.

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        "### Where could this fit in a real workflow?"
    )

    st.write(
        """
        The most realistic future role would be as a decision-support
        or quality-control tool rather than an automatic rescan system.

        Repeated quality flags could potentially contribute to
        protocol-level quality audits. An individual flagged
        examination could also receive additional human review.

        A repeat scan would remain a professional clinical decision
        based on whether the examination is actually diagnostically
        inadequate — not simply because an AI score crossed a threshold.
        """
    )

    st.warning(
        """
        **Research prototype only:** This application has not been
        clinically validated. Do not use its output to diagnose
        patients, accept/reject clinical examinations, automatically
        repeat scans, or change CT radiation-dose protocols.
        """
    )


# ============================================================
# ANALYZE
# ============================================================

elif st.session_state.page == "Analyze":

    # ========================================================
    # CHOOSE MODE
    # ========================================================

    if st.session_state.analyze_mode is None:

        st.markdown(
            '<div class="eyebrow">CT ANALYSIS</div>',
            unsafe_allow_html=True,
        )

        st.title("Analyze a CT Image")

        st.write(
            """
            Explore the prototype using your own CT image or
            start with one of the built-in demonstration cases.
            """
        )

        st.markdown(
            "### How would you like to begin?"
        )

        st.caption(
            "Choose an option below to continue."
        )

        upload_col, demo_col = st.columns(
            2,
            gap="large",
        )

        # ----------------------------------------------------
        # UPLOAD CARD
        # ----------------------------------------------------

        with upload_col:

            st.markdown(
                """
                <div class="option-card">

                    <div class="option-icon">
                    📤
                    </div>

                    <div class="option-title">
                    Upload Your CT
                    </div>

                    <div class="option-text">

                    Analyze your own CT image using the
                    AI-assisted quality-risk model.

                    <br><br>

                    <b>Supported:</b>
                    DICOM, PNG, JPG and JPEG

                    <br><br>

                    DICOM is preferred because it may contain
                    useful acquisition information.

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            if st.button(
                "Upload a CT image →",
                type="primary",
                use_container_width=True,
                disabled=not MODEL_READY,
            ):

                st.session_state.analyze_mode = "upload"
                st.rerun()

        # ----------------------------------------------------
        # DEMO CARD
        # ----------------------------------------------------

        with demo_col:

            st.markdown(
                """
                <div class="option-card">

                    <div class="option-icon">
                    🧪
                    </div>

                    <div class="option-title">
                    Try Demo Cases
                    </div>

                    <div class="option-text">

                    Explore prepared CT examples to understand
                    how the model, quality-risk score and Grad-CAM
                    visualization work.

                    <br><br>

                    <b>Best for:</b>
                    first-time users, demonstrations and learning.

                    <br><br>

                    No upload is required.

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            if st.button(
                "Explore demo cases →",
                use_container_width=True,
                disabled=not MODEL_READY,
            ):

                st.session_state.analyze_mode = "demo"
                st.rerun()

        st.write("")

        st.markdown(
            """
            <div class="explain">

            <b>Not sure where to start?</b>

            Try the <b>Demo Cases</b> first. They let you explore
            the complete workflow without uploading your own CT image.

            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # UPLOAD MODE
    # ========================================================

    elif st.session_state.analyze_mode == "upload":

        top_left, top_right = st.columns(
            [3, 1]
        )

        with top_left:

            st.markdown(
                '<div class="eyebrow">YOUR CT IMAGE</div>',
                unsafe_allow_html=True,
            )

            st.title("Upload Your CT")

        with top_right:

            if st.button(
                "← Back",
                use_container_width=True,
            ):

                st.session_state.analyze_mode = None
                st.rerun()

        st.write(
            """
            Upload one or more CT images for analysis.
            The model will generate a quality-risk signal for each image.
            """
        )

        st.markdown(
            """
            <div class="explain">

            <b>Recommended: DICOM (.dcm)</b>

            <br>

            DICOM may contain both the CT image and acquisition
            information such as tube voltage, tube current and
            slice thickness.

            <br><br>

            PNG / JPG / JPEG are also supported for image-based
            testing.

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        files = st.file_uploader(
            "Choose CT image(s)",
            type=[
                "dcm",
                "png",
                "jpg",
                "jpeg",
            ],
            accept_multiple_files=True,
            help=(
                "The Streamlit server configuration controls "
                "the maximum upload size."
            ),
        )

        if files:

            for file in files:

                if any(
                    r["name"] == file.name
                    for r in st.session_state.results
                ):
                    continue

                try:

                    with st.spinner(
                        f"Analyzing {file.name}..."
                    ):

                        if file.name.lower().endswith(".dcm"):

                            image, metadata = (
                                dicom_to_image(
                                    file.read()
                                )
                            )

                        else:

                            image = (
                                Image.open(file)
                                .convert("RGB")
                            )

                            metadata = {}

                        arr, gradcam, score = analyze(
                            image
                        )

                    add_result(
                        file.name,
                        arr,
                        gradcam,
                        score,
                        metadata,
                    )

                    st.success(
                        f"{file.name} analyzed successfully."
                    )

                except Exception as exc:

                    st.error(
                        f"Could not analyze {file.name}."
                    )

                    st.caption(
                        str(exc)
                    )

            st.write("")

            if st.button(
                "View analysis result →",
                type="primary",
            ):

                navigate("Results")

    # ========================================================
    # DEMO CASES MODE
    # ========================================================

    elif st.session_state.analyze_mode == "demo":

        top_left, top_right = st.columns(
            [3, 1]
        )

        with top_left:

            st.markdown(
                '<div class="eyebrow">'
                'DEMONSTRATION LIBRARY'
                '</div>',
                unsafe_allow_html=True,
            )

            st.title("Explore Demo Cases")

        with top_right:

            if st.button(
                "← Back",
                use_container_width=True,
            ):

                st.session_state.analyze_mode = None
                st.rerun()

        st.write(
            """
            Explore prepared CT examples and see how the model
            generates and explains its quality-risk signal.
            """
        )

        st.info(
            """
            **Demonstration only:** These cases are provided to
            demonstrate the application workflow. They are not
            clinical validation cases.
            """
        )

        st.write("")

        demos = [
            (
                "Demo Case 1",
                "sample_acceptable_1.png",
                "Demonstration image",
            ),
            (
                "Demo Case 2",
                "sample_acceptable_2.png",
                "Demonstration image",
            ),
            (
                "Demo Case 3",
                "sample_flagged_1.png",
                "Demonstration image",
            ),
            (
                "Demo Case 4",
                "sample_flagged_2.png",
                "Demonstration image",
            ),
        ]

        cols = st.columns(
            2,
            gap="large",
        )

        for i, (
            title,
            filename,
            description,
        ) in enumerate(demos):

            path = os.path.join(
                "sample_images",
                filename,
            )

            with cols[i % 2]:

                if os.path.exists(path):

                    demo_image = (
                        Image.open(path)
                        .convert("RGB")
                    )

                    st.markdown(
                        '<div class="card">',
                        unsafe_allow_html=True,
                    )

                    st.image(
                        demo_image,
                        use_container_width=True,
                    )

                    st.markdown(
                        f"### {title}"
                    )

                    st.caption(
                        description
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    if st.button(
                        f"Analyze {title} →",
                        key=f"demo_{i}",
                        type="primary",
                        use_container_width=True,
                    ):

                        with st.spinner(
                            "Analyzing demonstration case..."
                        ):

                            arr, gradcam, score = analyze(
                                demo_image
                            )

                        add_result(
                            title,
                            arr,
                            gradcam,
                            score,
                            {},
                        )

                        navigate("Results")

                else:

                    st.warning(
                        f"{title} is not available."
                    )


# ============================================================
# RESULTS
# ============================================================

elif st.session_state.page == "Results":

    st.markdown(
        '<div class="eyebrow">ANALYSIS RESULT</div>',
        unsafe_allow_html=True,
    )

    st.title(
        "What does the model suggest?"
    )

    if not st.session_state.results:

        st.info(
            "No images have been analyzed yet."
        )

        if st.button(
            "Go to CT Analysis",
            type="primary",
        ):

            navigate("Analyze")

    else:

        names = [
            r["name"]
            for r in st.session_state.results
        ]

        selected = st.selectbox(
            "Select an analyzed image",
            names,
            index=max(
                0,
                names.index(
                    st.session_state.selected_result
                )
                if st.session_state.selected_result in names
                else 0,
            ),
        )

        result = next(
            r
            for r in st.session_state.results
            if r["name"] == selected
        )

        score = result["score"]

        info = interpretation(
            score
        )

        # ----------------------------------------------------
        # RESULT CARD
        # ----------------------------------------------------

        top1, top2 = st.columns(
            [1.05, 1.95],
            gap="large",
        )

        with top1:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="eyebrow">'
                'MODEL ASSESSMENT'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"## {info['emoji']} {info['level']}"
            )

            st.markdown(
                '<div class="score-label">'
                'Quality-risk score'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="score">'
                f'{score:.3f}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.progress(
                min(
                    max(score, 0),
                    1,
                )
            )

            st.markdown(
                f"**{info['status']}**"
            )

            st.caption(
                "Model output on a 0–1 scale; "
                "not a probability."
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        with top2:

            st.markdown(
                "### What does this mean?"
            )

            st.markdown(
                f"""
                <div class="explain">

                {info['summary']}

                <br><br>

                A higher score means the image more strongly
                resembles the pattern associated with the
                project's flagged group.

                <br><br>

                <b>This score does NOT establish:</b>

                <br>
                • whether the image is diagnostically acceptable
                <br>
                • whether a patient has a disease
                <br>
                • whether the scan should be repeated
                <br>
                • whether CT radiation dose should be changed

                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # TECHNOLOGIST GUIDANCE
        # ----------------------------------------------------

        st.write("")

        st.markdown(
            '<div class="eyebrow">'
            'TECHNOLOGIST GUIDANCE'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### So, what should the technologist do?"
        )

        st.markdown(
            f"""
            <div class="action-box">

            <div class="action-title">
            👩‍⚕️ Recommended next step
            </div>

            <div class="action-text">

            {info['action']}

            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        check1, check2 = st.columns(
            2,
            gap="large",
        )

        with check1:

            st.markdown(
                """
                **🔎 What to review**

                - Image noise
                - Motion
                - Artifacts
                - Required anatomical coverage
                - Visibility of relevant structures
                - Suitability for the intended examination
                """
            )

        with check2:

            st.markdown(
                """
                **🚫 What NOT to do from the AI score alone**

                - Do not automatically repeat the scan
                - Do not automatically increase radiation dose
                - Do not reject an examination solely from the score
                - Do not treat the score as a diagnosis
                """
            )

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        st.write("")

        st.markdown(
            '<div class="eyebrow">'
            'MODEL INTERPRETABILITY'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### 👁️ Where did the model look?"
        )

        st.caption(
            """
            Grad-CAM provides a visual explanation of regions
            that contributed to the model prediction. It is an
            interpretability aid — not a diagnostic heatmap.
            """
        )

        c1, c2 = st.columns(
            2,
            gap="large",
        )

        with c1:

            st.image(
                np.clip(
                    result["image"],
                    0,
                    255,
                ).astype(
                    np.uint8
                ),
                caption="Original CT image",
                use_container_width=True,
            )

        with c2:

            st.image(
                np.clip(
                    result["gradcam"],
                    0,
                    255,
                ).astype(
                    np.uint8
                ),
                caption="Grad-CAM visualization",
                use_container_width=True,
            )

        st.info(
            """
            **Important:** A highlighted region does not mean that
            the region is abnormal, contains pathology, or is
            objectively responsible for poor image quality.
            Grad-CAM only provides an approximate view of model attention.
            """
        )

        # ----------------------------------------------------
        # DICOM INFORMATION
        # ----------------------------------------------------

        if result["metadata"]:

            st.markdown(
                "### 📋 DICOM acquisition information"
            )

            st.caption(
                """
                These values come from DICOM metadata when they
                are present in the uploaded file.
                """
            )

            metadata_df = pd.DataFrame(
                list(
                    result["metadata"].items()
                ),
                columns=[
                    "Parameter",
                    "Value",
                ],
            )

            st.dataframe(
                metadata_df,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        with st.expander(
            "Why does the project use 0.25?"
        ):

            st.write(
                """
                A threshold is a decision point used to convert
                a continuous model score into a review signal.

                In this research prototype, 0.25 was selected as
                the project-specific review threshold.

                It is not a universal definition of acceptable
                CT image quality.

                A clinically useful threshold would require
                extensive validation on independent patients,
                scanners, institutions and clinically meaningful
                image-quality assessments.
                """
            )

        st.warning(
            """
            **Clinical safety:** This prototype is not clinically
            validated. Its output should be treated only as a
            research/educational signal. Do not use it alone to
            diagnose disease, repeat examinations, reject clinical
            scans or modify radiation-dose protocols.
            """
        )


# ============================================================
# LEARN
# ============================================================

elif st.session_state.page == "Learn":

    st.markdown(
        '<div class="eyebrow">PROJECT GUIDE</div>',
        unsafe_allow_html=True,
    )

    st.title(
        "Understand the Technology"
    )

    st.write(
        """
        The main concepts behind the prototype are explained here
        in simple language.
        """
    )

    topics = [

        (
            "🧠 VGG16",
            """
            VGG16 is a deep-learning model originally developed
            for image recognition.

            This project uses transfer learning, where visual
            features learned previously are adapted to the
            CT image-quality task.

            In simple terms, the model learns visual patterns
            that help distinguish between the groups represented
            in the training data.
            """
        ),

        (
            "🔄 Test-Time Augmentation",
            """
            The model does not rely on only one presentation
            of an image.

            Five views are used:

            • original image
            • horizontally flipped image
            • small rotation
            • opposite small rotation
            • small crop/zoom

            The predictions are averaged.

            The purpose is to reduce sensitivity to small changes
            in image presentation.
            """
        ),

        (
            "👁️ Grad-CAM",
            """
            Grad-CAM is an interpretability technique that helps
            visualize regions that contributed to a model prediction.

            Think of it as asking:

            "Which parts of this image were influential for the model?"

            It does not prove that the highlighted region is abnormal
            or clinically important.
            """
        ),

        (
            "🩻 DICOM",
            """
            DICOM is a standard format used in medical imaging.

            A DICOM file can contain the image together with
            acquisition information such as tube voltage,
            tube current, exposure and slice thickness.
            """
        ),

        (
            "📈 Quality-risk score",
            """
            The model produces a numerical output between 0 and 1.

            A higher value means the image more strongly resembles
            the pattern associated with the project's flagged group.

            Important:

            This is NOT a probability that the image is poor quality.

            For example, a score of 0.60 does not mean
            "60% poor quality."
            """
        ),

        (
            "🎯 The 0.25 threshold",
            """
            The project uses 0.25 as its review threshold.

            Below 0.25:
            No automatic review flag.

            0.25 or above:
            Review signal.

            This threshold belongs to this research prototype.
            It is not a universal clinical rule.
            """
        ),

        (
            "📊 Recall and precision",
            """
            Recall asks:

            "Of the cases belonging to the flagged group,
            how many did the model identify?"

            Precision asks:

            "Of the cases flagged by the model,
            how many belonged to the flagged group?"

            Reported project evaluation:

            Recall: 85%
            Precision: 54%
            F1-score: 0.66
            ROC-AUC: 0.839
            Full-dose false-positive rate: 3.3%

            These results describe this project's evaluation.
            They do not establish clinical effectiveness.
            """
        ),
    ]

    for title, explanation in topics:

        with st.expander(title):
            st.write(explanation)

    st.divider()

    # ========================================================
    # WHY AI?
    # ========================================================

    st.markdown(
        '<div class="eyebrow">WHY AI?</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "### If the noise is visible, why use AI?"
    )

    st.write(
        """
        This is an important limitation and a fair question.

        Some of the study/demo images contain relatively obvious
        differences in noise. A human can identify many of these
        cases visually.

        That does not mean the AI has already solved the harder
        clinical problem.
        """
    )

    reason1, reason2, reason3 = st.columns(3)

    with reason1:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            01 — Consistency
            </div>

            <div class="card-text">
            An automated model can apply the same learned criterion
            repeatedly instead of relying entirely on subjective
            visual judgment.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with reason2:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            02 — Scale
            </div>

            <div class="card-text">
            A complete CT examination can contain hundreds of images.
            Automated screening could help prioritize examinations
            that deserve closer attention.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with reason3:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            03 — Future subtle cases
            </div>

            <div class="card-text">
            The more important future test is whether the approach
            can identify smaller, less obvious quality changes that
            are harder to judge consistently by visual inspection.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # ========================================================
    # LIMITATION
    # ========================================================

    st.markdown(
        '<div class="eyebrow">RESEARCH LIMITATION</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="limitation-box">

        <b>The current experiment should be interpreted as
        proof-of-concept.</b>

        <br><br>

        The training/evaluation data contain paired full-dose
        and low-dose CT images with relatively clear differences
        in image noise. This is useful for testing whether the
        model can learn the intended distinction, but it does
        not recreate every real-world CT dose-optimization scenario.

        <br><br>

        In real clinical environments, useful dose reduction may
        produce much more subtle changes in image noise and
        diagnostic quality.

        <br><br>

        Therefore, future work should evaluate the approach using
        larger and independent datasets, more realistic dose
        variations, different scanners and institutions, and
        clinically meaningful image-quality assessments.

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ========================================================
    # FUTURE WORKFLOW
    # ========================================================

    st.markdown(
        '<div class="eyebrow">FUTURE WORKFLOW</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Where could this tool fit in practice?"
    )

    st.write(
        """
        The intended concept is not:

        **AI flags image → automatically repeat scan**

        A more realistic future workflow would be:

        **CT examination → AI screening signal → human
        image-quality review → clinical/QC decision**

        At a broader level, repeated flags could potentially
        be studied during protocol-level quality audits.

        For example, if a particular dose-reduction protocol
        repeatedly produces images that receive quality-risk
        flags, that pattern could become a reason for further
        protocol evaluation.

        Any clinical action would remain the responsibility of
        appropriately qualified professionals and institutional
        protocols.
        """
    )

    st.warning(
        "Research prototype only — not for clinical use."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CT Image Quality Flagger • Zainab Fatima • "
    "Medical Imaging Technology • "
    "Educational/research prototype only"
)
