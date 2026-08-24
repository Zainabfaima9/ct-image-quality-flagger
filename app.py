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
# CUSTOM CSS
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

    #MainMenu,
    footer {
        visibility: hidden;
    }

    /* ---------- NAVIGATION ---------- */

    .nav-brand {
        font-size: 1.05rem;
        font-weight: 800;
        color: #102a43;
        margin-bottom: 0;
    }

    .nav-sub {
        font-size: .72rem;
        color: #718096;
        margin-top: -2px;
    }

    .nav-divider {
        margin-top: .35rem;
        margin-bottom: 1.3rem;
        border-bottom: 1px solid #dbe5ec;
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
        padding: 2.7rem 3rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 32px rgba(16,42,67,.13);
    }

    .hero-kicker {
        font-size: .7rem;
        letter-spacing: .12em;
        font-weight: 750;
        opacity: .78;
        margin-bottom: .75rem;
    }

    .hero h1 {
        color: white;
        font-size: clamp(2rem, 5vw, 3.25rem);
        line-height: 1.05;
        margin: 0 0 .75rem 0;
        font-weight: 800;
    }

    .hero p {
        color: rgba(255,255,255,.9);
        max-width: 720px;
        line-height: 1.6;
        margin: 0;
        font-size: 1rem;
    }

    /* ---------- CARDS ---------- */

    .choice-card {
        background: white;
        border: 1px solid #dfe7ee;
        border-radius: 18px;
        padding: 1.4rem;
        height: 100%;
        box-shadow: 0 4px 15px rgba(15,23,42,.035);
    }

    .choice-icon {
        font-size: 1.8rem;
        margin-bottom: .5rem;
    }

    .choice-title {
        color: #102a43;
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: .35rem;
    }

    .choice-text {
        color: #607084;
        font-size: .9rem;
        line-height: 1.55;
        min-height: 48px;
    }

    .simple-card {
        background: white;
        border: 1px solid #e0e7ee;
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 3px 12px rgba(15,23,42,.03);
    }

    /* ---------- LABELS ---------- */

    .eyebrow {
        color: #247ba0;
        font-size: .7rem;
        font-weight: 800;
        letter-spacing: .11em;
        text-transform: uppercase;
        margin-bottom: .25rem;
    }

    /* ---------- RESULT ---------- */

    .score-box {
        background: white;
        border: 1px solid #dfe7ee;
        border-radius: 18px;
        padding: 1.35rem;
        height: 100%;
    }

    .score {
        font-size: 3rem;
        font-weight: 850;
        line-height: 1;
        color: #102a43;
        margin: .25rem 0 .45rem;
    }

    .score-label {
        font-size: .72rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: .07em;
        font-weight: 750;
    }

    .action-box {
        border-radius: 16px;
        padding: 1.15rem 1.3rem;
        margin-top: 1rem;
        border: 1px solid #d9e5ec;
        background: #f7fbfd;
    }

    .action-title {
        font-size: 1rem;
        font-weight: 800;
        color: #102a43;
        margin-bottom: .35rem;
    }

    .action-text {
        color: #43566b;
        line-height: 1.55;
        font-size: .9rem;
    }

    /* ---------- INFO ---------- */

    .small-info {
        color: #607084;
        font-size: .82rem;
        line-height: 1.55;
    }

    .why-box {
        background: #eef7fa;
        border-left: 4px solid #247ba0;
        border-radius: 10px;
        padding: .9rem 1rem;
        color: #40566a;
        font-size: .86rem;
        line-height: 1.55;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 9px;
        min-height: 2.55rem;
        font-weight: 650;
    }

    /* ---------- FILE UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background: white;
        border: 1.5px dashed #8db9ca;
        border-radius: 15px;
        padding: .5rem;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 700px) {

        .block-container {
            padding-left: .8rem;
            padding-right: .8rem;
        }

        .hero {
            padding: 2rem 1.35rem;
            border-radius: 18px;
        }

        .hero h1 {
            font-size: 2.1rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP NAVIGATION
# ============================================================

nav_left, nav_right = st.columns([1.5, 3.5], vertical_alignment="center")

with nav_left:
    st.markdown(
        """
        <div class="nav-brand">🩻 CT Image Quality Flagger</div>
        <div class="nav-sub">AI-assisted research prototype</div>
        """,
        unsafe_allow_html=True,
    )

with nav_right:
    n1, n2, n3, n4 = st.columns(4)

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

st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)


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

    hu = np.clip(
        hu,
        0,
        255
    ).astype(np.uint8)

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

    arr = np.array(image).astype(np.uint8)

    model_input = np.expand_dims(
        arr.astype(np.float32),
        axis=0,
    )

    model_input = preprocess_input(model_input)

    return arr, model_input


# ============================================================
# MODEL SCORE
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

    # Small rotations
    image = model_input[0]

    center = (
        IMAGE_SIZE[0] // 2,
        IMAGE_SIZE[1] // 2
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
    model_input
):

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

    grads = tape.gradient(
        loss,
        conv_output
    )

    pooled = tf.reduce_mean(
        grads,
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
        1
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
        255
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
# RESULT INTERPRETATION
# ============================================================

def interpretation(score):

    if score >= THRESHOLD:

        return {
            "status": "Review Needed",
            "icon": "🟠",
            "color": "warning",
            "meaning": (
                "The model detected a pattern that "
                "resembles the image-quality problems "
                "it was trained to flag."
            ),
            "action": (
                "Visually review the image for excessive "
                "noise, artifacts, or loss of useful detail "
                "and confirm whether the image remains "
                "adequate for the intended examination."
            ),
            "next": (
                "If the image is genuinely non-diagnostic, "
                "follow your department's normal clinical "
                "protocol. Do not repeat a scan based on "
                "the AI flag alone."
            ),
        }

    return {
        "status": "No Review Flag",
        "icon": "🟢",
        "color": "success",
        "meaning": (
            "The model did not produce a quality-risk score "
            "high enough to trigger this project's review threshold."
        ),
        "action": (
            "Continue the normal image-quality assessment. "
            "The AI result does not replace the technologist's "
            "visual and clinical judgement."
        ),
        "next": (
            "No additional action is suggested by the model. "
            "Follow the usual departmental workflow."
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
    metadata
):

    existing = [
        r["name"]
        for r in st.session_state.results
    ]

    if name not in existing:

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

            <h1>
                CT Image Quality Flagger
            </h1>

            <p>
                An AI-assisted research prototype that flags CT images
                showing patterns associated with reduced image quality,
                helping the technologist decide which images deserve
                a closer review.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1.5, 1],
        gap="large"
    )

    with left:

        st.markdown("### The idea")

        st.write(
            """
            When CT radiation dose is reduced, image noise can increase.
            The important question is not simply whether an image looks noisy,
            but whether the image quality may have crossed a level that
            deserves another look.
            """
        )

        st.markdown(
            """
            <div class="why-box">
            <b>What this prototype does:</b>
            It provides an additional AI-based review signal.
            The final judgement remains with the imaging professional.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "Start CT Analysis →",
            type="primary",
            use_container_width=True,
        ):
            navigate("Analyze")

    with right:

        st.markdown(
            """
            <div class="simple-card">

            <div class="eyebrow">
            WORKFLOW
            </div>

            <h3>
            Image → AI score → Review
            </h3>

            <p class="small-info">
            1. Upload a CT image<br>
            2. Model produces a quality-risk score<br>
            3. Review flag appears if score ≥ 0.25<br>
            4. Technologist makes the final assessment
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.caption(
        "Research prototype • Not clinically validated • "
        "Not for diagnosis or automatic CT protocol changes"
    )


# ============================================================
# ANALYZE LANDING
# ============================================================

elif st.session_state.page == "Analyze":

    st.markdown(
        '<div class="eyebrow">CT ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    st.title("Choose how to begin")

    st.caption(
        "Upload your own CT image or explore a prepared demonstration case."
    )

    st.write("")

    left, right = st.columns(
        2,
        gap="large"
    )

    # ----------------------------------------
    # UPLOAD OPTION
    # ----------------------------------------

    with left:

        st.markdown(
            """
            <div class="choice-card">

            <div class="choice-icon">📤</div>

            <div class="choice-title">
            Upload your CT
            </div>

            <div class="choice-text">
            Analyze a DICOM, PNG or JPG image using
            the AI model.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if not MODEL_READY:

            st.error(
                "The AI model could not be loaded."
            )

            st.caption(
                MODEL_ERROR
            )

        files = st.file_uploader(
            "Choose CT image",
            type=[
                "dcm",
                "png",
                "jpg",
                "jpeg"
            ],
            accept_multiple_files=True,
            disabled=not MODEL_READY,
        )

        if files and MODEL_READY:

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

                            image, metadata = dicom_to_image(
                                file.read()
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
                        f"{file.name} analyzed."
                    )

                except Exception as exc:

                    st.error(
                        f"Could not analyze {file.name}."
                    )

                    st.caption(
                        str(exc)
                    )

            if files:

                if st.button(
                    "View Result →",
                    type="primary",
                    use_container_width=True,
                ):

                    navigate("Results")


    # ----------------------------------------
    # DEMO OPTION
    # ----------------------------------------

    with right:

        st.markdown(
            """
            <div class="choice-card">

            <div class="choice-icon">🧪</div>

            <div class="choice-title">
            Try Demo Cases
            </div>

            <div class="choice-text">
            Explore prepared CT cases without uploading
            your own image.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "Open Demo Cases →",
            use_container_width=True,
        ):

            navigate("Demo Cases")

    st.write("")
    st.divider()

    with st.expander("Why does this matter?"):

        st.write(
            """
            The model was developed as a proof-of-concept using
            paired full-dose and low-dose CT data. The differences
            in some study images can be visually obvious.

            That is a limitation, not the final goal.

            In real-world dose optimization, image-quality changes
            may be much more subtle. Future validation should therefore
            test the system on realistic dose reductions, different
            scanners, institutions and patients.
            """
        )


# ============================================================
# DEMO CASES
# ============================================================

elif st.session_state.page == "Demo Cases":

    st.markdown(
        '<div class="eyebrow">DEMONSTRATION</div>',
        unsafe_allow_html=True,
    )

    st.title("Demo Cases")

    st.caption(
        "These cases are included to demonstrate the workflow."
    )

    if st.button("← Back to Analyze"):

        navigate("Analyze")

    st.write("")

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

    cols = st.columns(4)

    for i, (
        title,
        filename
    ) in enumerate(demos):

        path = os.path.join(
            demo_folder,
            filename
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
                    "Analyze",
                    key=f"demo_{i}",
                    use_container_width=True,
                    disabled=not MODEL_READY,
                ):

                    with st.spinner(
                        "Analyzing case..."
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
                    f"{title} unavailable."
                )


# ============================================================
# RESULTS
# ============================================================

elif st.session_state.page == "Results":

    st.markdown(
        '<div class="eyebrow">AI ASSESSMENT</div>',
        unsafe_allow_html=True,
    )

    st.title("Result")

    if not st.session_state.results:

        st.info(
            "No CT image has been analyzed yet."
        )

        if st.button(
            "Start Analysis →",
            type="primary"
        ):

            navigate("Analyze")

    else:

        names = [
            r["name"]
            for r in st.session_state.results
        ]

        selected = st.selectbox(
            "Analyzed image",
            names,
            index=(
                names.index(
                    st.session_state.selected_result
                )
                if st.session_state.selected_result
                in names
                else 0
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

        # ------------------------------------
        # TOP RESULT
        # ------------------------------------

        left, right = st.columns(
            [1, 1.6],
            gap="large"
        )

        with left:

            st.markdown(
                """
                <div class="score-box">
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="eyebrow">MODEL OUTPUT</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"## {info['icon']} {info['status']}"
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
                    1
                )
            )

            st.caption(
                f"Review threshold: {THRESHOLD:.2f}"
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        with right:

            st.markdown("### What does the result mean?")

            st.write(
                info["meaning"]
            )

            # --------------------------------
            # PRACTICAL ACTION
            # --------------------------------

            st.markdown(
                f"""
                <div class="action-box">

                <div class="action-title">
                👨‍⚕️ What should the technologist do?
                </div>

                <div class="action-text">
                {info["action"]}
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="action-box">

                <div class="action-title">
                ➜ What happens next?
                </div>

                <div class="action-text">
                {info["next"]}
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # ------------------------------------
        # IMPORTANT CLARIFICATION
        # ------------------------------------

        if score >= THRESHOLD:

            st.warning(
                """
                **Review flag ≠ automatic rescan.**

                The AI flag means the image deserves closer
                human assessment. A repeat scan should only be
                considered when clinically justified and according
                to the department's established protocol.
                """
            )

        else:

            st.success(
                """
                **No AI review flag.**

                This does not prove that the image is clinically
                acceptable. Continue the normal image-quality check.
                """
            )

        # ------------------------------------
        # IMAGES
        # ------------------------------------

        st.markdown("### Image & model explanation")

        st.caption(
            "Grad-CAM highlights areas that contributed more strongly "
            "to the model's prediction. It does not identify disease."
        )

        c1, c2 = st.columns(
            2,
            gap="large"
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

        # ------------------------------------
        # DICOM
        # ------------------------------------

        if result["metadata"]:

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
                    ],
                )

                st.dataframe(
                    metadata_df,
                    use_container_width=True,
                    hide_index=True,
                )

        # ------------------------------------
        # SCORE EXPLANATION
        # ------------------------------------

        with st.expander(
            "How should I understand the score?"
        ):

            st.write(
                f"""
                The score ranges from 0 to 1.

                In this prototype, **0.25** is the selected
                review threshold.

                **Below 0.25:** no automatic review flag.

                **0.25 or above:** review signal.

                The threshold is specific to this research
                project. It is not a universal clinical cutoff.
                """
            )

        st.warning(
            """
            **Research prototype only.**

            The AI output should not be used alone to diagnose,
            reject a clinical scan, or change radiation-dose settings.
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

    st.title("How the AI works")

    st.caption(
        "A few important terms, explained simply."
    )

    topics = [

        (
            "🧠 VGG16",
            "The deep-learning image model used in this project. "
            "Transfer learning adapts an already-trained visual model "
            "to the CT image-quality task."
        ),

        (
            "🔄 Test-Time Augmentation",
            "The same CT image is tested in five slightly different "
            "versions and the results are averaged. This helps reduce "
            "sensitivity to small changes in image presentation."
        ),

        (
            "👁️ Grad-CAM",
            "A visualization method showing which parts of the image "
            "contributed more strongly to the model's prediction. "
            "It should be treated as an explanation aid, not a diagnosis."
        ),

        (
            "🩻 DICOM",
            "The standard medical-imaging format. It can contain both "
            "the CT image and acquisition information such as tube voltage, "
            "tube current and slice thickness."
        ),

        (
            "📈 Quality-risk score",
            "A numerical model output. A higher value means the image "
            "more strongly resembles the pattern the model was trained "
            "to flag."
        ),

        (
            "🎯 Threshold",
            "The point used to convert the model score into a review "
            "signal. This project uses 0.25."
        ),

        (
            "📊 Recall / Precision",
            "Recall describes how many flagged cases the model detected. "
            "Precision describes how many of the cases it flagged were "
            "actually in the flagged group."
        ),

    ]

    for title, explanation in topics:

        with st.expander(title):

            st.write(
                explanation
            )

    st.divider()

    st.markdown("### Project limitations")

    st.write(
        """
        This prototype was evaluated using patient-level held-out
        CT data. The quality labels were based on a noise-related
        image-quality proxy rather than radiologist-confirmed
        diagnostic ground truth.

        The dataset also contains some strongly different
        full-dose and low-dose examples. Real clinical dose
        reductions can be more subtle.

        Therefore, future validation should include larger datasets,
        more realistic dose variations, different scanners and
        independent clinical assessment.
        """
    )

    st.warning(
        "Educational/research prototype — not for clinical use."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CT Image Quality Flagger • Zainab Fatima • "
    "Medical Imaging Technology • Research prototype"
)
