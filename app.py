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

# Grad-CAM layer used in the VGG16 base
GRADCAM_LAYER = "block4_conv3"

# Chest CT soft-tissue window
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
# PROFESSIONAL UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GENERAL ---------- */

    .stApp {
        background: #f7fafc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1rem;
        padding-bottom: 2.5rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ---------- NAVBAR ---------- */

    .nav-title {
        color: #102a43;
        font-size: 1.05rem;
        font-weight: 800;
        line-height: 1.2;
        margin: 0;
    }

    .nav-subtitle {
        color: #718096;
        font-size: 0.72rem;
        margin-top: 0.15rem;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 2.45rem;
        font-weight: 650;
        white-space: nowrap;
    }

    /* ---------- HERO ---------- */

    .hero-box {
        background: linear-gradient(
            135deg,
            #102a43 0%,
            #174e73 55%,
            #247ba0 100%
        );
        border-radius: 22px;
        padding: 2.7rem 2.8rem;
        margin: 1rem 0 1.5rem 0;
        box-shadow: 0 12px 30px rgba(16, 42, 67, 0.13);
    }

    .hero-kicker {
        color: rgba(255,255,255,0.78);
        font-size: 0.70rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        margin-bottom: 0.7rem;
    }

    .hero-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.05;
        margin: 0;
    }

    .hero-text {
        color: rgba(255,255,255,0.90);
        font-size: 1rem;
        line-height: 1.6;
        max-width: 760px;
        margin-top: 0.8rem;
    }

    /* ---------- INTRO ---------- */

    .intro-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.2rem 1.35rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 3px 12px rgba(15,23,42,0.03);
    }

    .intro-title {
        color: #102a43;
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: 0.35rem;
    }

    .intro-text {
        color: #526174;
        font-size: 0.90rem;
        line-height: 1.6;
    }

    /* ---------- SMALL LABELS ---------- */

    .eyebrow {
        color: #247ba0;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    /* ---------- OPTION CARDS ---------- */

    .option-card {
        background: white;
        border: 1px solid #dfe7ee;
        border-radius: 18px;
        padding: 1.5rem;
        min-height: 170px;
        box-shadow: 0 4px 14px rgba(15,23,42,0.035);
    }

    .option-icon {
        font-size: 1.7rem;
        margin-bottom: 0.5rem;
    }

    .option-title {
        color: #102a43;
        font-size: 1.08rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .option-text {
        color: #64748b;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    /* ---------- RESULT CARD ---------- */

    .result-card {
        background: white;
        border: 1px solid #e1e8ef;
        border-radius: 18px;
        padding: 1.35rem;
        box-shadow: 0 4px 15px rgba(15,23,42,0.035);
    }

    .result-score {
        color: #102a43;
        font-size: 3.1rem;
        font-weight: 850;
        line-height: 1;
    }

    .score-label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    /* ---------- EXPLANATION ---------- */

    .explain-box {
        background: #eef7fa;
        border-left: 4px solid #247ba0;
        border-radius: 10px;
        padding: 0.95rem 1.05rem;
        color: #334e68;
        font-size: 0.90rem;
        line-height: 1.6;
    }

    .action-box {
        background: #f8fafc;
        border: 1px solid #dce5ec;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        color: #334e68;
        font-size: 0.90rem;
        line-height: 1.6;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 700px) {

        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }

        .hero-box {
            padding: 2rem 1.35rem;
            border-radius: 18px;
        }

        .hero-title {
            font-size: 2.1rem;
        }

        .hero-text {
            font-size: 0.90rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION BAR
# ============================================================

nav_brand, nav_buttons = st.columns(
    [1.45, 3.55],
    vertical_alignment="center"
)

with nav_brand:
    st.markdown(
        """
        <div class="nav-title">🩻 CT Image Quality Flagger</div>
        <div class="nav-subtitle">
            AI-assisted research prototype
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_buttons:

    n1, n2, n3, n4 = st.columns(
        [1, 1, 1, 1],
        gap="small"
    )

    with n1:
        if st.button("Home", use_container_width=True):
            navigate("Home")

    with n2:
        if st.button("Analyze", use_container_width=True):
            navigate("Analyze")

    with n3:
        if st.button("Results", use_container_width=True):
            navigate("Results")

    with n4:
        if st.button("Learn", use_container_width=True):
            navigate("Learn")


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

    return tf.keras.models.load_model(model_path)


@st.cache_resource
def build_gradcam_extractor(model):

    base_model = model.layers[0]

    extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(
            GRADCAM_LAYER
        ).output,
    )

    return extractor, base_model


try:

    model = load_model()

    gradcam_extractor, base_model = build_gradcam_extractor(
        model
    )

    MODEL_READY = True
    MODEL_ERROR = None

except Exception as exc:

    MODEL_READY = False
    MODEL_ERROR = str(exc)


# ============================================================
# DICOM PROCESSING
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


# ============================================================
# IMAGE PREPARATION
# ============================================================

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
        IMAGE_SIZE[1] // 2
    )

    # Small rotations
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

    # Small center crop / zoom
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

        conv_output = gradcam_extractor(
            model_input
        )

        tape.watch(
            conv_output
        )

        x = conv_output

        found_layer = False

        for layer in base_model.layers:

            if found_layer:
                x = layer(x)

            if layer.name == GRADCAM_LAYER:
                found_layer = True

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

    heatmap /= (
        tf.reduce_max(heatmap)
        + 1e-8
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
    ).astype(np.uint8)

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
            "label": "Higher review priority",
            "icon": "🔴",
            "status": "Review needed",
            "message": (
                "The model produced a relatively high "
                "quality-risk score."
            ),
            "action": (
                "Review the image carefully for excessive noise "
                "or loss of useful anatomical detail. If the image "
                "may be non-diagnostic, follow the department's "
                "normal image-quality and repeat-scan policy."
            ),
        }

    elif score >= THRESHOLD:

        return {
            "label": "Borderline quality signal",
            "icon": "🟠",
            "status": "Review needed",
            "message": (
                "The score reached the project's review threshold."
            ),
            "action": (
                "Take a closer look at image noise and anatomical "
                "detail before deciding whether the image is "
                "adequate for its intended clinical purpose."
            ),
        }

    elif score >= 0.15:

        return {
            "label": "Lower-risk pattern",
            "icon": "🟡",
            "status": "No automatic flag",
            "message": (
                "The score is below the project's review threshold, "
                "but the model output is not a clinical quality assessment."
            ),
            "action": (
                "No additional action is suggested by the model alone. "
                "Continue with the normal professional image-quality "
                "assessment."
            ),
        }

    else:

        return {
            "label": "Low-risk pattern",
            "icon": "🟢",
            "status": "No automatic flag",
            "message": (
                "The model produced a relatively low quality-risk score."
            ),
            "action": (
                "Continue with the normal professional image-quality "
                "assessment. The model does not replace visual or "
                "clinical review."
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
        result["name"]
        for result in st.session_state.results
    ]

    if name not in existing_names:

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

    # Hero
    st.markdown(
        """
        <div class="hero-box">

            <div class="hero-kicker">
                MEDICAL IMAGING TECHNOLOGY × AI
            </div>

            <div class="hero-title">
                CT Image Quality Flagger
            </div>

            <div class="hero-text">
                An AI-assisted research prototype that flags CT images
                showing patterns associated with reduced image quality,
                helping the technologist identify images that may deserve
                a closer review.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # Intro
    st.markdown(
        """
        <div class="intro-box">

            <div class="intro-title">
                Hi, I'm Zainab 👋
            </div>

            <div class="intro-text">
                I'm a Medical Imaging Technology student interested in
                how AI can support safer and more consistent
                medical-imaging workflows. I built this prototype around
                one practical question: can AI help flag CT images that
                may deserve a closer quality review?
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="eyebrow">GET STARTED</div>',
        unsafe_allow_html=True
    )

    st.subheader("What would you like to do?")

    option1, option2 = st.columns(
        2,
        gap="large"
    )

    with option1:

        st.markdown(
            """
            <div class="option-card">

                <div class="option-icon">📤</div>

                <div class="option-title">
                    Upload a CT image
                </div>

                <div class="option-text">
                    Upload a DICOM, PNG or JPG image and
                    see the model's quality-risk signal.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "Upload & Analyze →",
            type="primary",
            use_container_width=True
        ):
            navigate("Analyze")

    with option2:

        st.markdown(
            """
            <div class="option-card">

                <div class="option-icon">🧪</div>

                <div class="option-title">
                    Explore demo cases
                </div>

                <div class="option-text">
                    Try prepared examples to see how the
                    score, interpretation and Grad-CAM work.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "View Demo Cases →",
            use_container_width=True
        ):
            navigate("Demos")

    st.write("")
    st.divider()

    st.markdown(
        '<div class="eyebrow">THE IDEA</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### Why flag image quality?

        CT dose reduction can lower radiation exposure, but excessive
        reduction may increase image noise and reduce useful image detail.

        This prototype explores whether AI can provide an **additional
        review signal** so that potentially problematic images are easier
        to identify.

        **The AI does not replace the Medical Imaging Technologist's
        assessment.**
        """
    )

    st.info(
        "Research prototype only. The result should not be used to "
        "diagnose patients, accept/reject clinical scans, or change "
        "CT acquisition protocols."
    )


# ============================================================
# ANALYZE
# ============================================================

elif st.session_state.page == "Analyze":

    st.markdown(
        '<div class="eyebrow">CT ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.title("Analyze a CT Image")

    st.caption(
        "Choose an image below. The model produces a quality-risk "
        "score and a visual explanation."
    )

    if not MODEL_READY:

        st.error(
            "The AI model could not be loaded."
        )

        st.code(
            MODEL_ERROR
        )

    else:

        # ----------------------------------------------------
        # UPLOAD SECTION
        # ----------------------------------------------------

        st.markdown("### 📤 Upload your CT image")

        st.caption(
            "DICOM is preferred because it can contain acquisition "
            "information in addition to the image."
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
            label_visibility="collapsed",
        )

        if uploaded_files:

            for file in uploaded_files:

                if any(
                    result["name"] == file.name
                    for result in st.session_state.results
                ):
                    continue

                try:

                    with st.spinner(
                        f"Analyzing {file.name}..."
                    ):

                        if file.name.lower().endswith(".dcm"):

                            image, metadata = dicom_to_image(
                                file.read()
                            )

                        else:

                            image = (
                                Image
                                .open(file)
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
                        f"{file.name} analyzed successfully."
                    )

                except Exception as exc:

                    st.error(
                        f"Could not analyze {file.name}."
                    )

                    st.caption(
                        str(exc)
                    )

            if st.session_state.selected_result:

                if st.button(
                    "View Result →",
                    type="primary",
                    use_container_width=True
                ):
                    navigate("Results")

        # ----------------------------------------------------
        # DEMO CASES SHORTCUT
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="eyebrow">NOT READY TO UPLOAD?</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            "### Try a demonstration case"
        )

        st.caption(
            "Use prepared examples to understand the workflow first."
        )

        if st.button(
            "Explore Demo Cases →",
            use_container_width=True
        ):
            navigate("Demos")

        # ----------------------------------------------------
        # SIMPLE EXPLANATION
        # ----------------------------------------------------

        st.divider()

        with st.expander(
            "What does the score mean?"
        ):

            st.write(
                f"""
                The model produces a **quality-risk score** between
                0 and 1.

                A score of **{THRESHOLD:.2f} or above** triggers the
                project's review signal.

                This threshold is specific to this research prototype.
                It is **not a universal clinical cutoff**.
                """
            )


# ============================================================
# DEMO CASES
# ============================================================

elif st.session_state.page == "Demos":

    st.markdown(
        '<div class="eyebrow">DEMONSTRATION</div>',
        unsafe_allow_html=True
    )

    st.title("Demo Cases")

    st.caption(
        "These examples are included to demonstrate how the application works."
    )

    if not MODEL_READY:

        st.error(
            "The AI model could not be loaded."
        )

        st.code(
            MODEL_ERROR
        )

    else:

        demo_folder = "sample_images"

        demos = [
            (
                "Demo Case 1",
                "sample_acceptable_1.png",
                "Example with a lower-risk model signal."
            ),
            (
                "Demo Case 2",
                "sample_acceptable_2.png",
                "Example with a lower-risk model signal."
            ),
            (
                "Demo Case 3",
                "sample_flagged_1.png",
                "Example with a higher review signal."
            ),
            (
                "Demo Case 4",
                "sample_flagged_2.png",
                "Example with a higher review signal."
            ),
        ]

        cols = st.columns(
            4,
            gap="medium"
        )

        for i, (
            title,
            filename,
            description
        ) in enumerate(demos):

            path = os.path.join(
                demo_folder,
                filename
            )

            with cols[i]:

                if os.path.exists(path):

                    demo_image = (
                        Image
                        .open(path)
                        .convert("RGB")
                    )

                    st.image(
                        demo_image,
                        caption=title,
                        use_container_width=True
                    )

                    st.caption(
                        description
                    )

                    if st.button(
                        f"Analyze {title}",
                        key=f"demo_{i}",
                        use_container_width=True
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
                        f"{title} unavailable."
                    )

        st.divider()

        st.info(
            "Demo cases are for demonstrating the model workflow. "
            "They are not clinical examples or diagnostic references."
        )


# ============================================================
# RESULTS
# ============================================================

elif st.session_state.page == "Results":

    st.markdown(
        '<div class="eyebrow">MODEL RESULT</div>',
        unsafe_allow_html=True
    )

    st.title("CT Image Assessment")

    if not st.session_state.results:

        st.info(
            "No CT images have been analyzed yet."
        )

        if st.button(
            "Analyze a CT Image →",
            type="primary"
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
            "Select analyzed image",
            names,
            index=default_index
        )

        result = next(
            result
            for result in st.session_state.results
            if result["name"] == selected
        )

        score = result["score"]

        interpretation_result = interpretation(
            score
        )

        # ----------------------------------------------------
        # MAIN RESULT
        # ----------------------------------------------------

        left, right = st.columns(
            [1, 1.55],
            gap="large"
        )

        with left:

            st.markdown(
                """
                <div class="result-card">
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="eyebrow">MODEL ASSESSMENT</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                ### {interpretation_result["icon"]}
                {interpretation_result["label"]}
                """
            )

            st.markdown(
                '<div class="score-label">Quality-risk score</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="result-score">
                    {score:.3f}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(
                    max(score, 0),
                    1
                )
            )

            st.markdown(
                f"**{interpretation_result['status']}**"
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        with right:

            st.markdown("### What does this mean?")

            st.markdown(
                f"""
                <div class="explain-box">

                {interpretation_result["message"]}

                <br><br>

                The score indicates how strongly the image resembles
                the pattern the model was trained to flag.

                <br><br>

                <b>It does not determine whether the scan is
                clinically acceptable.</b>

                </div>
                """,
                unsafe_allow_html=True
            )

            if score >= THRESHOLD:

                st.warning(
                    f"Review signal triggered — score ≥ {THRESHOLD:.2f}"
                )

            else:

                st.success(
                    f"No automatic flag — score < {THRESHOLD:.2f}"
                )

        # ----------------------------------------------------
        # WHAT SHOULD THE TECHNOLOGIST DO?
        # ----------------------------------------------------

        st.write("")

        st.markdown(
            '<div class="eyebrow">PRACTICAL INTERPRETATION</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            "### What should the technologist do?"
        )

        st.markdown(
            f"""
            <div class="action-box">

            <b>Recommended next step:</b><br><br>

            {interpretation_result["action"]}

            <br><br>

            The AI flag is a <b>prompt to review</b>, not an instruction
            to repeat the scan or change the radiation dose.

            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # VISUAL EXPLANATION
        # ----------------------------------------------------

        st.write("")

        st.markdown(
            '<div class="eyebrow">MODEL EXPLANATION</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            "### Where was the model looking?"
        )

        st.caption(
            "Grad-CAM highlights image regions that contributed "
            "to the model prediction. It does not identify disease."
        )

        image_col, gradcam_col = st.columns(
            2,
            gap="large"
        )

        with image_col:

            st.image(
                result["image"],
                caption="CT image",
                use_container_width=True
            )

        with gradcam_col:

            st.image(
                result["gradcam"],
                caption="Grad-CAM",
                use_container_width=True
            )

        # ----------------------------------------------------
        # DICOM INFORMATION
        # ----------------------------------------------------

        if result["metadata"]:

            st.write("")

            with st.expander(
                "View available DICOM information"
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

        # ----------------------------------------------------
        # SCORE EXPLANATION
        # ----------------------------------------------------

        with st.expander(
            "How should I understand the 0.25 threshold?"
        ):

            st.write(
                """
                The threshold is simply the point at which this
                prototype produces a review flag.

                **Below 0.25:** the model does not automatically flag
                the image.

                **0.25 or above:** the model produces a review signal.

                This value was selected for this project's research
                workflow. It is not a validated clinical cutoff.
                """
            )

        # ----------------------------------------------------
        # IMPORTANT LIMITATION
        # ----------------------------------------------------

        with st.expander(
            "Important limitation of this prototype"
        ):

            st.write(
                """
                The development dataset contains paired full-dose
                and low-dose CT images where the image-quality
                differences can be relatively obvious.

                This makes it useful for a proof-of-concept, but real
                clinical dose reduction may produce much more subtle
                changes in noise and image quality.

                Future validation should therefore test the model on
                larger datasets, different scanners and institutions,
                and more subtle real-world dose variations.
                """
            )

        st.warning(
            """
            **Research prototype only:** Do not use this output to
            diagnose patients, independently accept/reject clinical
            images, repeat scans, or modify CT radiation-dose protocols.
            """
        )


# ============================================================
# LEARN
# ============================================================

elif st.session_state.page == "Learn":

    st.markdown(
        '<div class="eyebrow">ABOUT THE MODEL</div>',
        unsafe_allow_html=True
    )

    st.title("How it works")

    st.caption(
        "A quick explanation of the main terms used in the application."
    )

    # --------------------------------------------------------
    # VGG16
    # --------------------------------------------------------

    with st.expander("🧠 VGG16 — the image model"):

        st.write(
            """
            VGG16 is a deep-learning model that learns visual patterns
            from images.

            In this project, transfer learning was used: a VGG16-based
            model was adapted to the CT image-quality task instead of
            training an entire visual model from scratch.
            """
        )

    # --------------------------------------------------------
    # TTA
    # --------------------------------------------------------

    with st.expander("🔄 TTA — Test-Time Augmentation"):

        st.write(
            """
            TTA means the model looks at several slightly modified
            versions of the same image.

            This prototype averages five views:

            • original image  
            • flipped image  
            • small rotation  
            • opposite small rotation  
            • small crop/zoom  

            The goal is to make the final prediction less dependent
            on a small change in image presentation.
            """
        )

    # --------------------------------------------------------
    # GRAD CAM
    # --------------------------------------------------------

    with st.expander("👁️ Grad-CAM — visual explanation"):

        st.write(
            """
            Grad-CAM is an explainability technique.

            It creates a heatmap showing regions that contributed
            to the model's prediction.

            In simple terms:

            **"Where was the model looking?"**

            The heatmap does not prove that a highlighted region is
            abnormal or clinically important.
            """
        )

    # --------------------------------------------------------
    # DICOM
    # --------------------------------------------------------

    with st.expander("🩻 DICOM — medical image format"):

        st.write(
            """
            DICOM is a standard format used for medical imaging.

            A CT DICOM file can contain both the image and information
            about image acquisition, such as tube voltage, tube current,
            exposure and slice thickness.
            """
        )

    # --------------------------------------------------------
    # QUALITY RISK SCORE
    # --------------------------------------------------------

    with st.expander("📊 Quality-risk score"):

        st.write(
            """
            The model produces a score between 0 and 1.

            A higher score means the image more strongly resembles
            the pattern the model was trained to flag.

            It is a model output — not a direct measurement of
            clinical image quality.
            """
        )

    # --------------------------------------------------------
    # RECALL / PRECISION
    # --------------------------------------------------------

    with st.expander("📈 Model evaluation"):

        st.write(
            """
            The project was evaluated using patient-level held-out
            testing.

            Reported results:

            • Recall: 85%  
            • Precision: 54%  
            • F1-score: 0.66  
            • ROC-AUC: 0.839  
            • Full-dose false-positive rate: 3.3%

            These results describe this prototype's evaluation.
            They do not establish clinical effectiveness.
            """
        )

    # --------------------------------------------------------
    # PROJECT IDEA
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="eyebrow">WHY THIS PROJECT?</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### The practical question

        Lower CT radiation dose can improve patient safety, but
        excessive dose reduction can increase image noise.

        The idea behind this prototype is not:

        **"Let AI decide whether a CT scan is good or bad."**

        It is:

        **"Can AI provide an additional signal that tells the
        technologist which images may deserve a closer look?"**

        That distinction is important because the final assessment
        remains a professional and clinical decision.
        """
    )

    st.warning(
        "This project is an educational/research prototype and "
        "has not been clinically validated."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CT Image Quality Flagger • Zainab Fatima • "
    "Medical Imaging Technology • Research prototype only"
)
