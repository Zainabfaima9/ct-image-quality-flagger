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
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background-color: #f7f9fc;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ---------- NAVIGATION ---------- */

    div[data-testid="stHorizontalBlock"] {
        gap: 0.65rem;
    }

    .nav-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #102a43;
        margin-bottom: 0;
    }

    .nav-subtitle {
        font-size: 0.72rem;
        color: #6b7c93;
        margin-top: 2px;
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
        padding: 42px 44px;
        margin-top: 18px;
        margin-bottom: 28px;
        color: white;
    }

    .hero-kicker {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: #c9e9f2;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 3rem;
        line-height: 1.05;
        font-weight: 800;
        margin: 0;
        color: white;
    }

    .hero-text {
        font-size: 1rem;
        line-height: 1.65;
        max-width: 760px;
        margin-top: 15px;
        color: #e8f2f7;
    }


    /* ---------- CARDS ---------- */

    .info-card {
        background: white;
        border: 1px solid #e1e8ef;
        border-radius: 16px;
        padding: 22px;
        min-height: 150px;
    }

    .info-title {
        color: #102a43;
        font-weight: 750;
        font-size: 1rem;
        margin-bottom: 7px;
    }

    .info-text {
        color: #607286;
        font-size: 0.88rem;
        line-height: 1.55;
    }


    /* ---------- SECTION LABEL ---------- */

    .section-label {
        color: #247ba0;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }


    /* ---------- RESULT SCORE ---------- */

    .score-box {
        background: white;
        border: 1px solid #e1e8ef;
        border-radius: 18px;
        padding: 25px;
    }

    .score-number {
        font-size: 3.2rem;
        font-weight: 850;
        line-height: 1;
        color: #102a43;
    }

    .score-caption {
        color: #718096;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }


    /* ---------- EXPLANATION ---------- */

    .simple-box {
        background: #eef7fa;
        border-left: 4px solid #247ba0;
        border-radius: 10px;
        padding: 16px 18px;
        color: #334e68;
        line-height: 1.55;
        font-size: 0.9rem;
    }


    /* ---------- UPLOAD ---------- */

    [data-testid="stFileUploader"] {
        background: white;
        border: 1.5px dashed #8db9ca;
        border-radius: 15px;
        padding: 8px;
    }


    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 9px;
        min-height: 2.55rem;
        font-weight: 650;
    }


    /* ---------- MOBILE ---------- */

    @media (max-width: 700px) {

        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }

        .hero-box {
            padding: 28px 23px;
            border-radius: 17px;
        }

        .hero-title {
            font-size: 2.15rem;
        }

        .hero-text {
            font-size: 0.9rem;
        }

        .nav-title {
            font-size: 0.95rem;
        }

        .nav-subtitle {
            display: none;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP NAVIGATION
# ============================================================

nav_left, nav_right = st.columns([1.4, 4.6], vertical_alignment="center")

with nav_left:
    st.markdown(
        """
        <div class="nav-title">🩻 CT Image Quality Flagger</div>
        <div class="nav-subtitle">AI-assisted research prototype</div>
        """,
        unsafe_allow_html=True,
    )

with nav_right:

    n1, n2, n3, n4, n5 = st.columns(5)

    with n1:
        if st.button("Home", use_container_width=True):
            navigate("Home")

    with n2:
        if st.button("Analyze CT", use_container_width=True):
            navigate("Analyze")

    with n3:
        if st.button("Demo Cases", use_container_width=True):
            navigate("Demo")

    with n4:
        if st.button("Results", use_container_width=True):
            navigate("Results")

    with n5:
        if st.button("Learn", use_container_width=True):
            navigate("Learn")


st.divider()


# ============================================================
# MODEL
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
        outputs=base_model.get_layer(GRADCAM_LAYER).output,
    )

    return extractor, base_model


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

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))

    hu = pixels * slope + intercept

    low = WINDOW_CENTER - WINDOW_WIDTH / 2
    high = WINDOW_CENTER + WINDOW_WIDTH / 2

    hu = np.clip(hu, low, high)

    hu = (
        (hu - low)
        / (high - low)
        * 255
    )

    hu = np.clip(hu, 0, 255).astype(np.uint8)

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
            metadata[label] = str(getattr(ds, tag))

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

    model_input = preprocess_input(model_input)

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
                verbose=0,
            )[0][0]
        )
    )

    # Horizontal flip
    flipped = np.flip(
        model_input,
        axis=2,
    )

    predictions.append(
        float(
            model.predict(
                flipped,
                verbose=0,
            )[0][0]
        )
    )

    # Small rotations
    image = model_input[0]

    center = (112, 112)

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
                    verbose=0,
                )[0][0]
            )
        )

    # Small crop / zoom
    crop = image[11:213, 11:213]

    zoomed = cv2.resize(
        crop,
        IMAGE_SIZE,
    )

    zoomed = np.expand_dims(
        zoomed,
        axis=0,
    )

    predictions.append(
        float(
            model.predict(
                zoomed,
                verbose=0,
            )[0][0]
        )
    )

    return float(np.mean(predictions))


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam(image_array, model_input):

    with tf.GradientTape() as tape:

        conv_output = conv_layer_model(
            model_input
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
        conv_output,
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    conv = conv_output[0]

    heatmap = conv @ pooled_gradients[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(
        heatmap,
        0,
    )

    heatmap /= (
        tf.reduce_max(heatmap)
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
        (heatmap * 255).astype(np.uint8),
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
        0.60,
        heatmap_color,
        0.40,
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

    return arr, gradcam, score


# ============================================================
# INTERPRETATION
# ============================================================

def interpret_score(score):

    if score >= 0.50:

        return {
            "status": "Review needed",
            "icon": "🔴",
            "level": "Higher-risk pattern",
            "action": (
                "Do not automatically reject or repeat the scan. "
                "Review the image for noise and diagnostic visibility, "
                "and compare with the clinical task."
            ),
        }

    elif score >= THRESHOLD:

        return {
            "status": "Review needed",
            "icon": "🟠",
            "level": "Borderline risk",
            "action": (
                "Give this image a closer quality check. "
                "Look specifically for excessive noise or loss of "
                "important anatomical detail before making any decision."
            ),
        }

    elif score >= 0.15:

        return {
            "status": "No automatic flag",
            "icon": "🟡",
            "level": "Lower-risk pattern",
            "action": (
                "No additional action is triggered by the model. "
                "Continue normal professional image-quality assessment."
            ),
        }

    else:

        return {
            "status": "No automatic flag",
            "icon": "🟢",
            "level": "Low-risk pattern",
            "action": (
                "The model does not identify a strong quality-risk pattern. "
                "Continue the usual professional review."
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

    existing_names = [
        r["name"]
        for r in st.session_state.results
    ]

    if name not in existing_names:

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

        st.markdown("### Hi, I'm Zainab 👋")

        st.write(
            """
            I'm a Medical Imaging Technology student interested in
            how AI can support safer and more consistent medical
            imaging workflows.

            I built this project around a simple question:

            **When CT dose is reduced, how can we identify images
            that may need a closer quality review?**
            """
        )

        st.write(
            "The tool is designed as a decision-support research prototype — "
            "not as a replacement for professional judgment."
        )

        if st.button(
            "Start CT Analysis →",
            type="primary",
            use_container_width=True,
        ):
            navigate("Analyze")

    with right:

        st.markdown("### What the tool does")

        st.markdown(
            """
            **1. Takes a CT image**  
            Upload DICOM, PNG or JPG.

            **2. Produces a quality-risk score**  
            The model estimates how strongly the image resembles
            patterns associated with reduced quality.

            **3. Highlights model attention**  
            Grad-CAM provides a visual explanation.

            **4. Suggests the next review step**  
            The technologist remains responsible for the final assessment.
            """
        )

    st.write("")

    st.markdown("### Choose how you want to begin")

    c1, c2 = st.columns(
        2,
        gap="large",
    )

    with c1:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-title">
                    📤 Upload your CT
                </div>

                <div class="info-text">
                    Analyze your own DICOM, PNG or JPG image
                    and view the model result.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "Upload & Analyze",
            use_container_width=True,
        ):
            navigate("Analyze")

    with c2:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-title">
                    🧪 Explore Demo Cases
                </div>

                <div class="info-text">
                    Try prepared examples to understand how
                    the quality-risk flag works.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "View Demo Cases",
            use_container_width=True,
        ):
            navigate("Demo")

    st.write("")

    st.caption(
        "Research prototype • Not clinically validated • "
        "Not for diagnosis or CT protocol modification"
    )


# ============================================================
# ANALYZE
# ============================================================

elif st.session_state.page == "Analyze":

    st.markdown(
        '<div class="section-label">CT ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    st.title("Analyze a CT Image")

    st.write(
        "Upload a CT image to generate a quality-risk score "
        "and model explanation."
    )

    if not MODEL_READY:

        st.error(
            "The AI model could not be loaded."
        )

        st.code(
            MODEL_ERROR,
            language="text",
        )

    else:

        uploaded_files = st.file_uploader(
            "Upload CT image",
            type=[
                "dcm",
                "png",
                "jpg",
                "jpeg",
            ],
            accept_multiple_files=True,
        )

        st.caption(
            "DICOM is preferred because it can also contain CT acquisition information."
        )

        if uploaded_files:

            for file in uploaded_files:

                if any(
                    r["name"] == file.name
                    for r in st.session_state.results
                ):
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

            if uploaded_files:

                st.write("")

                if st.button(
                    "View Analysis Report →",
                    type="primary",
                    use_container_width=True,
                ):
                    navigate("Results")

    st.divider()

    st.markdown("### What does the score mean?")

    st.markdown(
        f"""
        <div class="simple-box">

        <b>Quality-risk score</b> = how strongly the model's
        learned pattern resembles an image that should receive
        closer quality review.

        <br><br>

        <b>Project threshold: {THRESHOLD:.2f}</b>

        <br>

        At or above the threshold → <b>Review needed</b>

        <br>

        Below the threshold → <b>No automatic flag</b>

        <br><br>

        This is a project-specific threshold, not a universal
        clinical cutoff.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DEMO CASES
# ============================================================

elif st.session_state.page == "Demo":

    st.markdown(
        '<div class="section-label">DEMONSTRATION</div>',
        unsafe_allow_html=True,
    )

    st.title("Demo Cases")

    st.write(
        "Try the prepared cases to see how the analysis workflow works."
    )

    demo_folder = "sample_images"

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

    for i, (title, filename) in enumerate(demos):

        path = os.path.join(
            demo_folder,
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

                st.info(
                    f"{title} unavailable"
                )

    st.divider()

    st.caption(
        "Demo images are included to demonstrate the application workflow. "
        "They are not clinical validation cases."
    )


# ============================================================
# RESULTS
# ============================================================

elif st.session_state.page == "Results":

    st.markdown(
        '<div class="section-label">ANALYSIS REPORT</div>',
        unsafe_allow_html=True,
    )

    st.title("Analysis Report")

    if not st.session_state.results:

        st.info(
            "No CT images have been analyzed yet."
        )

        if st.button(
            "Analyze a CT Image →",
            type="primary",
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
            index=selected_index,
        )

        result = next(
            r
            for r in st.session_state.results
            if r["name"] == selected
        )

        score = result["score"]

        interpretation = interpret_score(
            score
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        left, right = st.columns(
            [1, 1.6],
            gap="large",
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

                    <br>

                    <b>
                        {interpretation["icon"]}
                        {interpretation["status"]}
                    </b>

                    <br><br>

                    <span style="color:#718096;">
                        {interpretation["level"]}
                    </span>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:

            st.markdown("### What should the technologist do?")

            st.markdown(
                f"""
                <div class="simple-box">

                <b>{interpretation["action"]}</b>

                <br><br>

                The model is a <b>flagging aid</b>.
                It does not decide whether an image is
                diagnostically acceptable.

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        if score >= THRESHOLD:

            st.warning(
                "Review signal triggered — inspect the image before making a clinical-quality decision."
            )

        else:

            st.success(
                "No automatic review flag — continue the normal professional image-quality assessment."
            )

        # ----------------------------------------------------
        # IMAGE + GRADCAM
        # ----------------------------------------------------

        st.divider()

        st.markdown("### Visual explanation")

        st.caption(
            "Grad-CAM shows regions that contributed to the model prediction. "
            "It is an explanation aid, not a diagnostic heatmap."
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
                caption="Grad-CAM — model attention",
                use_container_width=True,
            )

        # ----------------------------------------------------
        # SIMPLE INTERPRETATION
        # ----------------------------------------------------

        st.markdown("### How to read this")

        st.markdown(
            """
            **CT image:** the image being assessed.

            **Grad-CAM:** areas that contributed more strongly
            to the model's prediction.

            **Important:** highlighted areas do not automatically
            mean abnormal anatomy or a diagnostic problem.
            The technologist must assess the actual image.
            """
        )

        # ----------------------------------------------------
        # DICOM
        # ----------------------------------------------------

        if result["metadata"]:

            with st.expander(
                "View DICOM acquisition information"
            ):

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
                The project uses 0.25 as its research
                decision threshold.

                A score at or above 0.25 produces a
                review signal.

                This threshold was selected for this
                prototype and should not be interpreted
                as a universal clinical cutoff.
                """
            )

        st.divider()

        st.warning(
            "Clinical safety: this prototype has not been clinically validated. "
            "Do not use its output alone to diagnose disease, reject a clinical scan, "
            "or change CT radiation-dose protocols."
        )


# ============================================================
# LEARN
# ============================================================

elif st.session_state.page == "Learn":

    st.markdown(
        '<div class="section-label">QUICK GUIDE</div>',
        unsafe_allow_html=True,
    )

    st.title("How It Works")

    st.write(
        "A quick explanation of the main terms used by the application."
    )

    with st.expander("🧠 What is VGG16?"):

        st.write(
            """
            VGG16 is a deep-learning image model.

            In this project, transfer learning was used:
            a model that already learned visual features
            was adapted for the CT image-quality task.
            """
        )

    with st.expander("🔄 What is Test-Time Augmentation?"):

        st.write(
            """
            The same CT image is viewed in five slightly
            modified ways — including a flip, small rotations
            and a small zoom.

            Their predictions are averaged to produce the
            final score.
            """
        )

    with st.expander("👁️ What is Grad-CAM?"):

        st.write(
            """
            Grad-CAM is a visual explanation method.

            It highlights areas that contributed to the
            model's prediction.

            It helps answer:

            "Where was the model looking?"

            It does not prove that the highlighted region
            is abnormal.
            """
        )

    with st.expander("🩻 What is DICOM?"):

        st.write(
            """
            DICOM is the standard format used for medical
            imaging.

            It can contain both the image and acquisition
            information such as tube voltage, tube current
            and slice thickness.
            """
        )

    with st.expander("📈 What is the quality-risk score?"):

        st.write(
            """
            It is the model's numerical output.

            A higher score means the image more strongly
            resembles the pattern the model was trained
            to flag for review.

            It is not a clinical measurement.
            """
        )

    st.divider()

    st.markdown("### Project context")

    st.write(
        """
        The model was developed using paired full-dose and
        low-dose chest CT data from 21 patients.

        The quality labels were based on a noise-related
        image-quality proxy rather than radiologist-confirmed
        diagnostic ground truth.

        Testing was performed using fully held-out patients.
        """
    )

    st.info(
        "The current project is a proof-of-concept. "
        "Real-world validation would need larger datasets, "
        "more subtle dose variations, different scanners and "
        "independent clinical assessment."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CT Image Quality Flagger • Zainab Fatima • "
    "Medical Imaging Technology • Research prototype"
)
