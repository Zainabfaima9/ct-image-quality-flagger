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
# PAGE CONFIG — MUST COME BEFORE ANY st COMMAND
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
        background-color: #f7f9fc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ---------- HEADER ---------- */

    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        padding: 8px 0 14px 0;
    }

    .app-brand {
        font-size: 1.18rem;
        font-weight: 800;
        color: #102a43;
        line-height: 1.2;
        white-space: nowrap;
    }

    .app-subtitle {
        font-size: 0.75rem;
        color: #6b7c93;
        margin-top: 3px;
    }

    /* ---------- HERO ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #102a43 0%,
            #174e73 60%,
            #247ba0 100%
        );

        border-radius: 22px;

        padding: 2.8rem 3rem;

        margin: 0.7rem 0 1.6rem 0;

        color: white;

        box-shadow:
            0 12px 30px rgba(16, 42, 67, 0.13);
    }

    .hero-kicker {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        opacity: 0.82;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: clamp(2rem, 4vw, 3.25rem);
        font-weight: 800;
        line-height: 1.05;
        margin: 0 0 0.8rem 0;
        color: white;
    }

    .hero-text {
        max-width: 760px;
        font-size: 1rem;
        line-height: 1.6;
        color: rgba(255,255,255,0.92);
        margin: 0;
    }

    /* ---------- CARDS ---------- */

    .info-card {
        background: white;
        border: 1px solid #e1e8ef;
        border-radius: 16px;
        padding: 1.3rem;
        height: 100%;
        box-shadow: 0 4px 14px rgba(15,23,42,0.035);
    }

    .card-title {
        font-weight: 750;
        font-size: 1rem;
        color: #102a43;
        margin-bottom: 0.4rem;
    }

    .card-text {
        color: #526174;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    /* ---------- SCORE ---------- */

    .score-box {
        background: white;
        border: 1px solid #e1e8ef;
        border-radius: 18px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(15,23,42,0.035);
    }

    .score-number {
        font-size: 3.2rem;
        font-weight: 850;
        color: #102a43;
        line-height: 1;
        margin: 0.5rem 0;
    }

    .score-caption {
        font-size: 0.75rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    /* ---------- INTERPRETABILITY ---------- */

    .interpret-box {
        background: #eef7fa;
        border-left: 4px solid #247ba0;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        color: #334e68;
        line-height: 1.55;
        margin-top: 0.8rem;
    }

    .action-box {
        background: #f8fafc;
        border: 1px solid #dbe5ec;
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        margin-top: 1rem;
    }

    .action-title {
        color: #102a43;
        font-weight: 800;
        margin-bottom: 0.55rem;
    }

    .small-note {
        color: #64748b;
        font-size: 0.82rem;
    }

    /* ---------- UPLOAD ---------- */

    [data-testid="stFileUploader"] {
        background: white;
        border: 1.5px dashed #8db9ca;
        border-radius: 15px;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 10px;
        min-height: 2.6rem;
        font-weight: 650;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 700px) {

        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }

        .hero {
            padding: 2rem 1.35rem;
            border-radius: 18px;
        }

        .hero-title {
            font-size: 2rem;
        }

        .hero-text {
            font-size: 0.9rem;
        }

        .app-brand {
            font-size: 0.95rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP HEADER
# ============================================================

header_left, header_right = st.columns(
    [1.7, 2.3],
    vertical_alignment="center"
)

with header_left:
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <div class="app-brand">🩻 {APP_NAME}</div>
                <div class="app-subtitle">
                    AI-assisted CT image-quality research prototype
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:

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
        if st.button("About", use_container_width=True):
            navigate("About")


st.divider()


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILENAME,
    )

    return tf.keras.models.load_model(
        model_path,
        compile=False,
    )


@st.cache_resource
def build_gradcam_extractor(model):

    base_model = model.layers[0]

    gradcam_layer = base_model.get_layer(
        GRADCAM_LAYER
    )

    extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=gradcam_layer.output,
    )

    return extractor, base_model


try:

    model = load_model()

    conv_layer_model, base_model = build_gradcam_extractor(
        model
    )

    MODEL_READY = True
    MODEL_ERROR = None

except Exception as exc:

    MODEL_READY = False
    MODEL_ERROR = str(exc)


# ============================================================
# IMAGE PROCESSING
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
            1
        )
    )

    intercept = float(
        getattr(
            ds,
            "RescaleIntercept",
            0
        )
    )

    hu = pixels * slope + intercept

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
        high
    )

    hu = (
        (hu - low)
        / (high - low)
        * 255
    )

    hu = np.clip(
        hu,
        0,
        255
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
                getattr(ds, tag)
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
        arr.astype(np.float32),
        axis=0
    )

    model_input = preprocess_input(
        model_input
    )

    return arr, model_input


# ============================================================
# TEST-TIME AUGMENTATION
# ============================================================

def get_score(model_input):

    predictions = []

    # 1 — Original
    predictions.append(
        float(
            model.predict(
                model_input,
                verbose=0
            )[0][0]
        )
    )

    # 2 — Horizontal flip
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
        IMAGE_SIZE[1] // 2
    )

    # 3 & 4 — Small rotations
    for angle in (-5, 5):

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            image,
            matrix,
            IMAGE_SIZE,
            borderMode=cv2.BORDER_REFLECT
        )

        rotated = np.expand_dims(
            rotated,
            axis=0
        )

        predictions.append(
            float(
                model.predict(
                    rotated,
                    verbose=0
                )[0][0]
            )
        )

    # 5 — Small crop / zoom
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
        axis=0
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

def make_gradcam(image_array, model_input):

    with tf.GradientTape() as tape:

        conv_output = conv_layer_model(
            model_input
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

        prediction = x[:, 0]

    gradients = tape.gradient(
        prediction,
        conv_output
    )

    if gradients is None:

        return image_array

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv = conv_output[0]

    heatmap = (
        conv
        @ pooled_gradients[..., tf.newaxis]
    )

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    maximum = tf.reduce_max(
        heatmap
    )

    heatmap = (
        heatmap
        / (maximum + 1e-8)
    )

    heatmap = cv2.resize(
        heatmap.numpy(),
        IMAGE_SIZE,
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = np.clip(
        heatmap,
        0,
        1
    )

    heatmap_color = cv2.applyColorMap(
        (
            heatmap * 255
        ).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    image_array = np.clip(
        image_array,
        0,
        255
    ).astype(
        np.uint8
    )

    overlay = cv2.addWeighted(
        image_array,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    return np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )


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
        model_input
    )

    return (
        arr,
        gradcam,
        score
    )


# ============================================================
# INTERPRETATION
# ============================================================

def interpretation(score):

    if score >= 0.50:

        return {
            "label": "Higher review signal",
            "icon": "🔴",
            "message": (
                "The model produced a relatively high "
                "quality-risk score."
            ),
            "action": (
                "Review the image carefully for excessive "
                "noise or loss of useful anatomical detail."
            ),
        }

    elif score >= THRESHOLD:

        return {
            "label": "Review needed",
            "icon": "🟠",
            "message": (
                "The score reached the project's "
                "review threshold."
            ),
            "action": (
                "Take a closer look at image quality "
                "before considering the scan acceptable."
            ),
        }

    elif score >= 0.15:

        return {
            "label": "Lower review signal",
            "icon": "🟡",
            "message": (
                "The score is below the project's "
                "review threshold."
            ),
            "action": (
                "No automatic flag was triggered. "
                "Continue normal professional image assessment."
            ),
        }

    else:

        return {
            "label": "Low review signal",
            "icon": "🟢",
            "message": (
                "The model produced a relatively low "
                "quality-risk score."
            ),
            "action": (
                "No automatic review flag was triggered. "
                "The technologist should still perform normal QC."
            ),
        }


# ============================================================
# SAVE RESULT
# ============================================================

def add_result(
    name,
    image,
    gradcam,
    score,
    metadata
):

    existing_names = [
        r["name"]
        for r in st.session_state.results
    ]

    if name in existing_names:

        # Replace existing result
        st.session_state.results = [
            r
            for r in st.session_state.results
            if r["name"] != name
        ]

    st.session_state.results.append(
        {
            "name": name,
            "image": image,
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

    # --------------------------------------------------------
    # INTRO
    # --------------------------------------------------------

    left, right = st.columns(
        [1.4, 1],
        gap="large"
    )

    with left:

        st.subheader(
            "Hi, I'm Zainab 👋"
        )

        st.write(
            """
            I'm a Medical Imaging Technology student interested in
            how AI can support medical-imaging workflows.

            This project began with a practical question:

            **When CT dose is reduced, how can we identify images
            that may need a closer quality review?**
            """
        )

        st.caption(
            "The model provides a supporting signal — "
            "not a clinical decision."
        )

    with right:

        st.markdown(
            """
            <div class="info-card">

                <div class="card-title">
                    What this tool does
                </div>

                <div class="card-text">
                    It analyzes a CT image and produces a
                    quality-risk score. Images reaching the
                    project threshold are flagged for closer review.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # --------------------------------------------------------
    # QUICK ACTIONS
    # --------------------------------------------------------

    st.subheader("Start here")

    c1, c2 = st.columns(
        2,
        gap="large"
    )

    with c1:

        if st.button(
            "📤 Upload a CT Image",
            type="primary",
            use_container_width=True
        ):

            navigate("Analyze")

        st.caption(
            "Analyze a DICOM, PNG or JPG image."
        )

    with c2:

        if st.button(
            "🧪 Explore Demo Cases",
            use_container_width=True
        ):

            navigate("Demo Cases")

        st.caption(
            "See how the model behaves on example images."
        )

    st.write("")

    st.info(
        """
        **Important:** This is an educational/research prototype.
        It has not been clinically validated and should not be used
        to diagnose patients or independently change CT acquisition protocols.
        """
    )


# ============================================================
# ANALYZE PAGE
# ============================================================

elif st.session_state.page == "Analyze":

    st.title("Analyze a CT Image")

    st.write(
        "Choose how you want to test the prototype."
    )

    st.write("")

    option1, option2 = st.columns(
        2,
        gap="large"
    )

    # --------------------------------------------------------
    # UPLOAD OPTION
    # --------------------------------------------------------

    with option1:

        st.markdown(
            """
            <div class="info-card">

                <div class="card-title">
                    📤 Upload your CT
                </div>

                <div class="card-text">
                    Upload a DICOM, PNG or JPG image for analysis.
                    DICOM is preferred because it may also contain
                    acquisition information.
                </div>

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
                "jpeg"
            ],
            accept_multiple_files=True,
        )

        if files:

            if not MODEL_READY:

                st.error(
                    "The AI model could not be loaded."
                )

                st.caption(
                    MODEL_ERROR
                )

            else:

                for file in files:

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

                            arr, gradcam, score = analyze(
                                image
                            )

                        add_result(
                            file.name,
                            arr,
                            gradcam,
                            score,
                            metadata
                        )

                        st.success(
                            f"{file.name} analyzed."
                        )

                    except Exception as exc:

                        st.error(
                            f"Could not analyze {file.name}."
                        )

                        st.caption(
                            str(exc)
                        )

                if st.button(
                    "View Result →",
                    type="primary",
                    use_container_width=True
                ):

                    navigate("Results")

    # --------------------------------------------------------
    # DEMO OPTION
    # --------------------------------------------------------

    with option2:

        st.markdown(
            """
            <div class="info-card">

                <div class="card-title">
                    🧪 Demo Cases
                </div>

                <div class="card-text">
                    Not ready to upload your own image?
                    Explore prepared cases and see the complete
                    analysis workflow.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "Open Demo Cases →",
            use_container_width=True
        ):

            navigate("Demo Cases")

    st.write("")

    st.divider()

    st.caption(
        "Supported: DICOM (.dcm), PNG, JPG and JPEG."
    )


# ============================================================
# DEMO CASES
# ============================================================

elif st.session_state.page == "Demo Cases":

    st.title("Demo Cases")

    st.write(
        "These examples demonstrate how the quality-risk score "
        "and review flag work."
    )

    st.caption(
        "Demo cases are for understanding the prototype only."
    )

    demo_folder = "sample_images"

    demos = [
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
        ),
    ]

    columns = st.columns(
        4,
        gap="medium"
    )

    for index, (title, filename) in enumerate(
        demos
    ):

        path = os.path.join(
            demo_folder,
            filename
        )

        with columns[index]:

            if os.path.exists(path):

                demo_image = (
                    Image.open(path)
                    .convert("RGB")
                )

                st.image(
                    demo_image,
                    caption=title,
                    use_container_width=True
                )

                if st.button(
                    f"Analyze",
                    key=f"demo_{index}",
                    use_container_width=True,
                    disabled=not MODEL_READY
                ):

                    with st.spinner(
                        "Analyzing demo case..."
                    ):

                        arr, gradcam, score = analyze(
                            demo_image
                        )

                    add_result(
                        title,
                        arr,
                        gradcam,
                        score,
                        {}
                    )

                    navigate("Results")

            else:

                st.warning(
                    f"{title} image not found."
                )

    st.write("")

    st.info(
        """
        **Why use demo cases?**
        They make it easy to understand the workflow before
        testing your own CT images.
        """
    )


# ============================================================
# RESULTS
# ============================================================

elif st.session_state.page == "Results":

    st.title("Analysis Result")

    if not st.session_state.results:

        st.info(
            "No CT image has been analyzed yet."
        )

        if st.button(
            "Analyze a CT Image",
            type="primary"
        ):

            navigate("Analyze")

    else:

        names = [
            r["name"]
            for r in st.session_state.results
        ]

        selected_index = 0

        if (
            st.session_state.selected_result
            in names
        ):

            selected_index = names.index(
                st.session_state.selected_result
            )

        selected = st.selectbox(
            "Analyzed image",
            names,
            index=selected_index
        )

        result = next(
            r
            for r in st.session_state.results
            if r["name"] == selected
        )

        score = result["score"]

        interpretation_result = interpretation(
            score
        )

        label = interpretation_result["label"]
        icon = interpretation_result["icon"]
        message = interpretation_result["message"]
        action = interpretation_result["action"]

        # ----------------------------------------------------
        # RESULT HEADER
        # ----------------------------------------------------

        left, right = st.columns(
            [1, 2],
            gap="large"
        )

        with left:

            st.markdown(
                f"""
                <div class="score-box">

                    <div class="score-caption">
                        Quality-risk score
                    </div>

                    <div class="score-number">
                        {score:.3f}
                    </div>

                    <div>
                        {icon} <b>{label}</b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:

            st.subheader(
                "What does the result mean?"
            )

            st.write(
                message
            )

            if score >= THRESHOLD:

                st.warning(
                    f"""
                    Review flag triggered.
                    The score is ≥ {THRESHOLD:.2f}.
                    """
                )

            else:

                st.success(
                    f"""
                    No automatic review flag.
                    The score is < {THRESHOLD:.2f}.
                    """
                )

        st.write("")

        # ----------------------------------------------------
        # MOST IMPORTANT INTERPRETABILITY SECTION
        # ----------------------------------------------------

        st.subheader(
            "What should the technologist do?"
        )

        if score >= THRESHOLD:

            st.markdown(
                f"""
                <div class="action-box">

                    <div class="action-title">
                        🔎 Take a closer look
                    </div>

                    <div>
                        {action}
                    </div>

                    <br>

                    <div>
                        <b>Check for:</b>
                    </div>

                    <div class="small-note">
                        • Excessive image noise<br>
                        • Loss of useful anatomical detail<br>
                        • Whether the image remains suitable
                          for the intended examination
                    </div>

                    <br>

                    <div>
                        <b>Then:</b>
                    </div>

                    <div class="small-note">
                        Use your normal professional image-quality
                        assessment. The AI flag is an additional
                        signal, not a reason by itself to repeat
                        a scan or change the dose.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="action-box">

                    <div class="action-title">
                        ✓ No automatic review flag
                    </div>

                    <div>
                        {action}
                    </div>

                    <br>

                    <div class="small-note">
                        A low model score does not replace normal
                        image-quality assessment.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        st.write("")

        st.subheader(
            "Why did the model give this score?"
        )

        st.caption(
            "Grad-CAM provides a visual explanation of the "
            "image regions that contributed to the model prediction."
        )

        image_col, heatmap_col = st.columns(
            2,
            gap="large"
        )

        with image_col:

            st.image(
                result["image"],
                caption="CT image",
                use_container_width=True
            )

        with heatmap_col:

            st.image(
                result["gradcam"],
                caption="Grad-CAM",
                use_container_width=True
            )

        with st.expander(
            "How to read the Grad-CAM"
        ):

            st.write(
                """
                Brighter regions indicate areas that contributed
                more strongly to the model's prediction.

                This is an explanation aid — it does not mean that
                the highlighted region is diseased or objectively
                abnormal.
                """
            )

        # ----------------------------------------------------
        # DICOM INFORMATION
        # ----------------------------------------------------

        if result["metadata"]:

            st.write("")

            st.subheader(
                "DICOM information"
            )

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

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        with st.expander(
            "Why is the threshold 0.25?"
        ):

            st.write(
                """
                In this research prototype, 0.25 is the selected
                decision threshold.

                A score of 0.25 or higher produces a review flag.

                This threshold is specific to this project.
                It is not a universal clinical cutoff.
                """
            )

        st.warning(
            """
            Research prototype only. Do not use this output alone
            to diagnose disease, reject a clinical scan, repeat a scan,
            or change CT radiation-dose protocols.
            """
        )


# ============================================================
# ABOUT
# ============================================================

elif st.session_state.page == "About":

    st.title("About the Project")

    st.write(
        """
        **CT Image Quality Flagger** is an AI-assisted research
        prototype developed to explore whether machine learning can
        provide an additional signal when reviewing CT image quality
        during dose-optimization research.
        """
    )

    st.subheader(
        "The idea"
    )

    st.write(
        """
        When radiation dose is reduced, image noise can increase and
        useful image detail can decrease. The prototype asks whether
        an AI model can learn patterns associated with this change and
        flag images that deserve closer review.
        """
    )

    st.subheader(
        "What is inside?"
    )

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            """
            **VGG16**

            Transfer learning model used
            for image-quality classification.
            """
        )

    with b:

        st.markdown(
            """
            **Test-Time Augmentation**

            Five slightly different views
            are averaged for the final score.
            """
        )

    with c:

        st.markdown(
            """
            **Grad-CAM**

            Visual explanation of regions
            influencing the prediction.
            """
        )

    st.write("")

    st.subheader(
        "Important limitation"
    )

    st.write(
        """
        The training data used in this project contain paired
        full-dose and low-dose CT images, with quality labels based
        on a noise-related image-quality proxy rather than
        radiologist-confirmed diagnostic ground truth.

        Some low-dose examples show relatively obvious noise.
        Real-world dose reductions can be much subtler. Therefore,
        further validation on larger, diverse and clinically realistic
        datasets would be required before considering clinical use.
        """
    )

    st.warning(
        """
        This application is an educational/research prototype.
        It is not clinically validated.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CT Image Quality Flagger • Zainab Fatima • "
    "Medical Imaging Technology • Research prototype"
)
