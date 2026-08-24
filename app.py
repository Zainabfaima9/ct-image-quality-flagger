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

APP_NAME = "CT Image Quality Flagger"

HF_REPO = "zainabfatima9/ct-image-quality-flagger"
MODEL_FILENAME = "ct_quality_model_21patients_v2.h5"

IMAGE_SIZE = (224, 224)

THRESHOLD = 0.25

GRADCAM_LAYER = "block4_conv3"

WINDOW_CENTER = 40
WINDOW_WIDTH = 400

DEMO_FOLDER = "sample_images"


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "results" not in st.session_state:
    st.session_state.results = []

if "selected_result" not in st.session_state:
    st.session_state.selected_result = None


def navigate(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f6f9fc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ---------- TOP BRAND ---------- */

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }

    .brand-icon {
        font-size: 1.65rem;
        line-height: 1;
    }

    .brand-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #102a43;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 3px;
    }

    /* ---------- NAV ---------- */

    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }

    .nav-spacer {
        height: 1px;
    }

    /* ---------- HERO ---------- */

    .hero {
        background:
            linear-gradient(
                135deg,
                #102a43 0%,
                #174e73 58%,
                #267fa4 100%
            );

        border-radius: 22px;
        padding: 3.1rem 3rem;
        margin: 1.1rem 0 1.8rem 0;

        box-shadow:
            0 18px 45px rgba(16, 42, 67, 0.15);
    }

    .hero-kicker {
        color: rgba(255,255,255,.78);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .12em;
        margin-bottom: .9rem;
    }

    .hero-title {
        color: white;
        font-size: clamp(2.1rem, 5vw, 3.35rem);
        font-weight: 800;
        line-height: 1.05;
        margin: 0 0 1rem 0;
    }

    .hero-text {
        color: rgba(255,255,255,.91);
        max-width: 760px;
        font-size: 1rem;
        line-height: 1.65;
        margin: 0;
    }

    /* ---------- SECTION LABEL ---------- */

    .eyebrow {
        color: #247ba0;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .11em;
        text-transform: uppercase;
        margin-bottom: .25rem;
    }

    /* ---------- CARDS ---------- */

    .card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 17px;
        padding: 1.35rem;
        box-shadow: 0 5px 18px rgba(15,23,42,.035);
    }

    .option-card {
        background: white;
        border: 1px solid #dfe7ee;
        border-radius: 18px;
        padding: 1.45rem;
        min-height: 190px;
        box-shadow: 0 5px 18px rgba(15,23,42,.035);
    }

    .option-icon {
        font-size: 1.7rem;
        margin-bottom: .55rem;
    }

    .option-title {
        color: #102a43;
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: .35rem;
    }

    .option-text {
        color: #64748b;
        font-size: .9rem;
        line-height: 1.55;
    }

    /* ---------- SCORE ---------- */

    .score {
        color: #102a43;
        font-size: 3.2rem;
        font-weight: 850;
        line-height: 1;
        margin: .4rem 0 .7rem 0;
    }

    .score-label {
        color: #64748b;
        font-size: .73rem;
        font-weight: 800;
        letter-spacing: .07em;
        text-transform: uppercase;
    }

    /* ---------- EXPLANATION ---------- */

    .explain {
        background: #eef7fa;
        border-left: 4px solid #247ba0;
        border-radius: 11px;
        padding: 1rem 1.1rem;
        color: #334e68;
        font-size: .9rem;
        line-height: 1.6;
    }

    .action-box {
        background: #f8fafc;
        border: 1px solid #dbe5ec;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        margin-top: .8rem;
    }

    .action-title {
        color: #102a43;
        font-weight: 800;
        margin-bottom: .4rem;
    }

    .action-text {
        color: #526174;
        font-size: .9rem;
        line-height: 1.6;
    }

    /* ---------- RESULT STATUS ---------- */

    .status {
        display: inline-block;
        padding: .42rem .72rem;
        border-radius: 999px;
        font-size: .76rem;
        font-weight: 800;
        margin-bottom: .7rem;
    }

    .status-review {
        background: #fff1e8;
        color: #b45309;
    }

    .status-ok {
        background: #eaf7ef;
        color: #16734a;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: .75rem;
        padding-top: .8rem;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 10px;
        min-height: 2.55rem;
        font-weight: 700;
    }

    /* ---------- UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background: white;
        border: 1.5px dashed #8db9ca;
        border-radius: 15px;
        padding: .55rem;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 700px) {

        .block-container {
            padding-left: .8rem;
            padding-right: .8rem;
            padding-top: 1rem;
        }

        .hero {
            padding: 2rem 1.35rem;
            border-radius: 18px;
        }

        .hero-title {
            font-size: 2.05rem;
        }

        .hero-text {
            font-size: .9rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP NAVIGATION
# ============================================================

top_left, top_right = st.columns([1.15, 2.85])

with top_left:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🩻</div>
            <div>
                <div class="brand-title">CT Image Quality Flagger</div>
                <div class="brand-subtitle">
                    AI-assisted research prototype
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with top_right:

    nav1, nav2, nav3, nav4 = st.columns(4)

    with nav1:
        if st.button("Home", use_container_width=True):
            navigate("Home")

    with nav2:
        if st.button("Analyze", use_container_width=True):
            navigate("Analyze")

    with nav3:
        if st.button("Results", use_container_width=True):
            navigate("Results")

    with nav4:
        if st.button("Learn", use_container_width=True):
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

    conv_layer_model, base_model = build_gradcam_extractor(model)

    MODEL_READY = True
    MODEL_ERROR = None

except Exception as exc:

    MODEL_READY = False
    MODEL_ERROR = str(exc)


# ============================================================
# IMAGE PROCESSING
# ============================================================

def dicom_to_image(data):

    ds = pydicom.dcmread(io.BytesIO(data))

    pixels = ds.pixel_array.astype(np.float32)

    slope = float(
        getattr(ds, "RescaleSlope", 1)
    )

    intercept = float(
        getattr(ds, "RescaleIntercept", 0)
    )

    hu = pixels * slope + intercept

    low = WINDOW_CENTER - WINDOW_WIDTH / 2
    high = WINDOW_CENTER + WINDOW_WIDTH / 2

    hu = np.clip(
        hu,
        low,
        high,
    )

    hu = (
        (hu - low)
        /
        (high - low)
        *
        255
    )

    hu = np.clip(
        hu,
        0,
        255,
    ).astype(np.uint8)

    image = (
        Image
        .fromarray(hu)
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
                getattr(ds, tag)
            )

    return image, metadata


def prepare_image(image):

    image = (
        image
        .convert("RGB")
        .resize(IMAGE_SIZE)
    )

    arr = np.array(image).astype(np.uint8)

    model_input = np.expand_dims(
        arr.astype(np.float32),
        axis=0,
    )

    model_input = preprocess_input(
        model_input
    )

    return arr, model_input


# ============================================================
# MODEL SCORE + TTA
# ============================================================

def get_score(model_input):

    predictions = []

    # Original
    predictions.append(
        float(
            model.predict(
                model_input,
                verbose=0
            )[0][0]
        )
    )

    # Horizontal flip
    flipped = np.flip(
        model_input,
        axis=2
    )

    predictions.append(
        float(
            model.predict(
                flipped,
                verbose=0
            )[0][0]
        )
    )

    image = model_input[0]

    center = (
        IMAGE_SIZE[0] // 2,
        IMAGE_SIZE[1] // 2,
    )

    # Small rotations
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

        predictions.append(
            float(
                model.predict(
                    rotated,
                    verbose=0
                )[0][0]
            )
        )

    # Small crop / zoom
    crop = image[
        11:213,
        11:213
    ]

    zoomed = cv2.resize(
        crop,
        IMAGE_SIZE
    )

    zoomed = np.expand_dims(
        zoomed,
        axis=0,
    )

    predictions.append(
        float(
            model.predict(
                zoomed,
                verbose=0
            )[0][0]
        )
    )

    return float(
        np.mean(predictions)
    )


# ============================================================
# GRAD-CAM
# ============================================================

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
        @
        pooled[..., tf.newaxis]
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
        +
        1e-8
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
        (heatmap * 255).astype(
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
    ).astype(np.uint8)

    overlay = cv2.addWeighted(
        image_array,
        .60,
        heatmap_color,
        .40,
        0,
    )

    return np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze(image):

    arr, model_input = prepare_image(
        image
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
# INTERPRETATION
# ============================================================

def interpretation(score):

    if score >= THRESHOLD:

        if score >= 0.50:

            return {
                "label": "Review needed",
                "emoji": "🔴",
                "class": "status-review",
                "message": (
                    "The model produced a relatively high "
                    "quality-risk score."
                ),
                "action": (
                    "Do not immediately repeat the scan. "
                    "First review the image for noise, artifacts, "
                    "and whether the anatomy needed for the task "
                    "is adequately visible."
                ),
            }

        return {
            "label": "Review needed",
            "emoji": "🟠",
            "class": "status-review",
            "message": (
                "The score reached the project's review threshold."
            ),
            "action": (
                "Take a closer look at the image before making "
                "any technical decision. Check whether noise or "
                "other quality issues could affect interpretation."
            ),
        }

    return {
        "label": "No automatic flag",
        "emoji": "🟢",
        "class": "status-ok",
        "message": (
            "The model score is below the project's review threshold."
        ),
        "action": (
            "No additional action is triggered by this prototype. "
            "Continue the normal professional image-quality assessment."
        ),
    }


# ============================================================
# SAVE RESULT
# ============================================================

def add_result(
    name,
    arr,
    gradcam,
    score,
    metadata,
):

    names = [
        result["name"]
        for result in st.session_state.results
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
                MEDICAL IMAGING TECHNOLOGY × AI
            </div>

            <div class="hero-title">
                CT Image Quality Flagger
            </div>

            <div class="hero-text">
                An AI-assisted research prototype that flags CT images
                showing patterns associated with reduced image quality,
                helping the technologist decide which images deserve
                a closer review.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1.35, 1],
        gap="large",
    )

    with left:

        st.markdown(
            '<div class="eyebrow">ABOUT THE PROJECT</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Hi, I'm Zainab 👋")

        st.write(
            """
            I'm a Medical Imaging Technology student interested in
            how AI can support safer and more consistent medical
            imaging workflows.

            I built this prototype around one practical question:
            **when CT dose is reduced, how can we identify images
            that may deserve a closer quality review?**
            """
        )

        st.info(
            "This tool supports professional review — it does not replace it."
        )

    with right:

        st.markdown(
            """
            <div class="card">

            <div class="eyebrow">
            WHAT THIS TOOL DOES
            </div>

            <h3 style="color:#102a43;">
            A second look, not a final decision
            </h3>

            <p style="color:#526174; line-height:1.6;">
            The model gives a quality-risk score. If the score reaches
            the project's threshold, the image is flagged for closer
            review by the imaging professional.
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown(
        '<div class="eyebrow">GET STARTED</div>',
        unsafe_allow_html=True,
    )

    a, b = st.columns(2, gap="large")

    with a:

        st.markdown(
            """
            <div class="option-card">

                <div class="option-icon">📤</div>

                <div class="option-title">
                    Upload your CT
                </div>

                <div class="option-text">
                    Upload a DICOM, PNG or JPG image and
                    run the AI-assisted quality analysis.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "Upload & Analyze →",
            type="primary",
            use_container_width=True,
        ):
            navigate("Analyze")

    with b:

        st.markdown(
            """
            <div class="option-card">

                <div class="option-icon">🧪</div>

                <div class="option-title">
                    Try a demo case
                </div>

                <div class="option-text">
                    Explore prepared examples without uploading
                    your own CT image.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "Explore Demo Cases →",
            use_container_width=True,
        ):
            navigate("Demos")

    st.write("")

    st.caption(
        "Research prototype • Not clinically validated • "
        "Not for diagnosis or protocol modification"
    )


# ============================================================
# ANALYZE
# ============================================================

elif st.session_state.page == "Analyze":

    st.markdown(
        '<div class="eyebrow">CT ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    st.title("Upload your CT image")

    st.write(
        "Upload one or more CT images. DICOM is preferred because "
        "it may contain acquisition information."
    )

    if not MODEL_READY:

        st.error(
            "The AI model could not be loaded."
        )

        st.code(
            MODEL_ERROR
        )

    else:

        files = st.file_uploader(
            "Choose CT image(s)",
            type=[
                "dcm",
                "png",
                "jpg",
                "jpeg",
            ],
            accept_multiple_files=True,
        )

        if files:

            for file in files:

                existing = any(
                    r["name"] == file.name
                    for r in st.session_state.results
                )

                if existing:
                    continue

                try:

                    with st.spinner(
                        f"Analyzing {file.name}..."
                    ):

                        if file.name.lower().endswith(
                            ".dcm"
                        ):

                            image, metadata = (
                                dicom_to_image(
                                    file.read()
                                )
                            )

                        else:

                            image = (
                                Image.open(
                                    file
                                )
                                .convert("RGB")
                            )

                            metadata = {}

                        arr, gradcam, score = (
                            analyze(image)
                        )

                    add_result(
                        file.name,
                        arr,
                        gradcam,
                        score,
                        metadata,
                    )

                    st.success(
                        f"{file.name} analyzed."
                    )

                except Exception as exc:

                    st.error(
                        f"Could not analyze {file.name}."
                    )

                    st.exception(exc)

            st.write("")

            if st.button(
                "View Results →",
                type="primary",
                use_container_width=True,
            ):

                navigate("Results")


# ============================================================
# DEMO CASES
# ============================================================

elif st.session_state.page == "Demos":

    st.markdown(
        '<div class="eyebrow">DEMONSTRATION</div>',
        unsafe_allow_html=True,
    )

    st.title("Demo Cases")

    st.write(
        "These prepared examples demonstrate how the prototype "
        "responds to different image-quality patterns."
    )

    demos = [
        (
            "Demo Case 1",
            "sample_acceptable_1.png",
        ),
        (
            "Demo Case 2",
            "sample_acceptable_2.png",
        ),
        (
            "Demo Case 3",
            "sample_flagged_1.png",
        ),
        (
            "Demo Case 4",
            "sample_flagged_2.png",
        ),
    ]

    cols = st.columns(4)

    for i, (
        title,
        filename,
    ) in enumerate(demos):

        path = os.path.join(
            DEMO_FOLDER,
            filename,
        )

        with cols[i]:

            if os.path.exists(path):

                demo_image = (
                    Image.open(path)
                    .convert("RGB")
                )

                st.image(
                    demo_image,
                    caption=title,
                    use_container_width=True,
                )

                if st.button(
                    f"Analyze",
                    key=f"demo_{i}",
                    use_container_width=True,
                    disabled=not MODEL_READY,
                ):

                    with st.spinner(
                        "Analyzing demo case..."
                    ):

                        arr, gradcam, score = (
                            analyze(
                                demo_image
                            )
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
                    f"{title} unavailable"
                )


# ============================================================
# RESULTS
# ============================================================

elif st.session_state.page == "Results":

    st.markdown(
        '<div class="eyebrow">AI-ASSISTED REVIEW</div>',
        unsafe_allow_html=True,
    )

    st.title("Analysis Result")

    if not st.session_state.results:

        st.info(
            "No CT image has been analyzed yet."
        )

        if st.button(
            "Analyze a CT image →",
            type="primary",
        ):

            navigate("Analyze")

    else:

        names = [
            result["name"]
            for result in st.session_state.results
        ]

        default_index = 0

        if (
            st.session_state.selected_result
            in names
        ):

            default_index = names.index(
                st.session_state.selected_result
            )

        selected = st.selectbox(
            "Analyzed image",
            names,
            index=default_index,
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
        # MAIN RESULT
        # ----------------------------------------------------

        left, right = st.columns(
            [1, 1.55],
            gap="large",
        )

        with left:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="eyebrow">MODEL OUTPUT</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="status {info["class"]}">
                    {info["emoji"]} {info["label"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="score-label">Quality-risk score</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="score">{score:.3f}</div>',
                unsafe_allow_html=True,
            )

            st.progress(
                min(
                    max(score, 0),
                    1,
                )
            )

            st.caption(
                f"Project review threshold: {THRESHOLD:.2f}"
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        with right:

            st.markdown(
                '<div class="eyebrow">WHAT DOES THIS MEAN?</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="explain">

                {info["message"]}

                <br><br>

                The score describes how strongly the image
                resembles the pattern the model was trained
                to flag. It does <b>not</b> diagnose disease
                and does not establish clinical acceptability.

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # ----------------------------------------------------
        # TECHNOLOGIST ACTION
        # ----------------------------------------------------

        st.markdown(
            '<div class="eyebrow">PRACTICAL INTERPRETATION</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="action-box">

                <div class="action-title">
                    👩‍⚕️ What should the technologist do?
                </div>

                <div class="action-text">
                    {info["action"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        # ----------------------------------------------------
        # IMAGE + GRAD CAM
        # ----------------------------------------------------

        st.markdown(
            '<div class="eyebrow">MODEL INTERPRETABILITY</div>',
            unsafe_allow_html=True,
        )

        st.subheader("Where did the model focus?")

        st.caption(
            "Grad-CAM highlights image regions that contributed "
            "to the model prediction. It is a visualization aid, "
            "not a diagnostic heatmap."
        )

        c1, c2 = st.columns(
            2,
            gap="large",
        )

        with c1:

            st.image(
                result["image"],
                caption="CT image",
                use_container_width=True,
            )

        with c2:

            st.image(
                result["gradcam"],
                caption="Grad-CAM",
                use_container_width=True,
            )

        with st.expander(
            "How to read the Grad-CAM"
        ):

            st.write(
                """
                Warmer highlighted regions indicate areas that
                contributed more strongly to the model's prediction.

                This does not mean that the highlighted area is
                abnormal. The technologist should still assess the
                complete image, including noise, artifacts and
                anatomical visibility.
                """
            )

        # ----------------------------------------------------
        # DICOM
        # ----------------------------------------------------

        if result["metadata"]:

            st.markdown(
                '<div class="eyebrow">ACQUISITION INFORMATION</div>',
                unsafe_allow_html=True,
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
        # WHY THIS IS NOT AN AUTOMATIC RESCAN
        # ----------------------------------------------------

        with st.expander(
            "Why doesn't a flag automatically mean 'rescan'?"
        ):

            st.write(
                """
                A flagged image is a prompt for review, not an
                automatic rescan instruction.

                In a real workflow, the technologist would first
                determine whether image quality is actually
                insufficient for the clinical task.

                For a research setting, repeated flags could also
                help identify CT protocols that may need evaluation
                rather than changing a patient's scan automatically.
                """
            )

        st.warning(
            """
            Research prototype only. Do not use this score to
            diagnose patients, reject clinical scans, or change
            CT radiation-dose protocols.
            """
        )


# ============================================================
# LEARN
# ============================================================

elif st.session_state.page == "Learn":

    st.markdown(
        '<div class="eyebrow">QUICK GUIDE</div>',
        unsafe_allow_html=True,
    )

    st.title("How the tool works")

    st.write(
        "A few short explanations for the terms used in the results."
    )

    with st.expander("🧠 What is VGG16?"):

        st.write(
            """
            VGG16 is a deep-learning image model. In this project,
            transfer learning was used so that visual features learned
            previously could be adapted to the CT image-quality task.
            """
        )

    with st.expander("🔄 What is Test-Time Augmentation?"):

        st.write(
            """
            The same CT image is viewed in five slightly different
            ways and the predictions are averaged. This can make the
            final score less sensitive to small changes in image
            presentation.
            """
        )

    with st.expander("👁️ What is Grad-CAM?"):

        st.write(
            """
            Grad-CAM creates a heatmap showing which image regions
            contributed to the model's prediction. It helps us inspect
            the model rather than treating its output as a black box.
            """
        )

    with st.expander("📈 What is the quality-risk score?"):

        st.write(
            f"""
            The model outputs a numerical score between 0 and 1.

            In this prototype, {THRESHOLD:.2f} is the review threshold.
            A score at or above this value produces a review signal.

            This is a project-specific threshold, not a universal
            clinical cutoff.
            """
        )

    with st.expander(
        "🎯 Why can an image be flagged?"
    ):

        st.write(
            """
            The model was trained to recognize patterns associated
            with reduced image quality in the project dataset.

            A flag should therefore be treated as a prompt to inspect
            the image more carefully — especially for noise and other
            quality limitations.

            The model does not determine whether an image is clinically
            diagnostic.
            """
        )

    with st.expander(
        "⚠️ Why are the demo cases not enough to prove clinical value?"
    ):

        st.write(
            """
            The project dataset contains relatively clear differences
            between the paired full-dose and low-dose images.

            Real-world dose reduction can produce much subtler changes.
            Therefore, future evaluation would need larger,
            independent datasets containing more realistic dose
            variations and different scanners and institutions.
            """
        )

    st.divider()

    st.warning(
        "Educational/research prototype — not for clinical use."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        CT Image Quality Flagger · Zainab Fatima ·
        Medical Imaging Technology · Research Prototype
    </div>
    """,
    unsafe_allow_html=True,
)
