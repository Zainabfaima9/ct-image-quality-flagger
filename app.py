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
# PAGE CONFIG — MUST COME BEFORE ANY STREAMLIT UI
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
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f7fafc;
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
        margin-bottom: 12px;
    }

    .app-brand {
        color: #102a43;
        font-size: 1.15rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .app-subtitle {
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 3px;
    }

    /* ---------- HERO ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #102a43 0%,
            #174e73 58%,
            #247ba0 100%
        );

        border-radius: 22px;
        padding: 3rem 3.2rem;
        margin: 1.2rem 0 1.7rem 0;
        color: white;
        box-shadow: 0 14px 35px rgba(16, 42, 67, 0.14);
    }

    .hero-kicker {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        font-weight: 700;
        opacity: 0.82;
        margin-bottom: 0.9rem;
    }

    .hero-title {
        font-size: clamp(2.2rem, 5vw, 3.4rem);
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.8rem;
        color: white;
    }

    .hero-text {
        max-width: 760px;
        color: rgba(255,255,255,0.92);
        font-size: 1rem;
        line-height: 1.6;
        margin: 0;
    }

    /* ---------- CARDS ---------- */

    .card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 17px;
        padding: 1.35rem;
        box-shadow: 0 4px 15px rgba(15,23,42,0.035);
        height: 100%;
    }

    .card-title {
        color: #102a43;
        font-size: 1.05rem;
        font-weight: 750;
        margin-bottom: 0.4rem;
    }

    .card-text {
        color: #526174;
        font-size: 0.91rem;
        line-height: 1.6;
    }

    /* ---------- SECTION LABEL ---------- */

    .eyebrow {
        color: #247ba0;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    /* ---------- RESULT ---------- */

    .score {
        color: #102a43;
        font-size: 3.2rem;
        font-weight: 850;
        line-height: 1;
    }

    .score-label {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .interpretation {
        background: #eef7fa;
        border-left: 4px solid #247ba0;
        border-radius: 11px;
        padding: 1rem 1.1rem;
        color: #334e68;
        line-height: 1.6;
        font-size: 0.9rem;
    }

    .action-box {
        background: white;
        border: 1px solid #dbe5ec;
        border-radius: 15px;
        padding: 1.2rem 1.3rem;
        margin-top: 1rem;
    }

    .action-title {
        color: #102a43;
        font-weight: 800;
        margin-bottom: 0.65rem;
    }

    .action-item {
        color: #526174;
        line-height: 1.55;
        margin-bottom: 0.55rem;
    }

    /* ---------- UPLOAD ---------- */

    [data-testid="stFileUploader"] {
        background: white;
        border: 1.5px dashed #8db9ca;
        border-radius: 15px;
        padding: 0.5rem;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 10px;
        min-height: 2.55rem;
        font-weight: 650;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 700px) {

        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }

        .hero {
            padding: 2rem 1.3rem;
            border-radius: 18px;
        }

        .hero-title {
            font-size: 2.1rem;
        }

        .hero-text {
            font-size: 0.9rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP HEADER
# ============================================================

header_left, header_right = st.columns([1.7, 3.3])

with header_left:
    st.markdown(
        """
        <div class="app-brand">🩻 CT Image Quality Flagger</div>
        <div class="app-subtitle">
            AI-assisted CT image-quality research prototype
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
# DICOM
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

    hu = np.clip(hu, low, high)

    hu = (
        (hu - low)
        / (high - low)
        * 255
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


# ============================================================
# IMAGE PREPARATION
# ============================================================

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
# TEST-TIME AUGMENTATION
# ============================================================

def get_score(model_input):

    predictions = []

    predictions.append(
        float(
            model.predict(
                model_input,
                verbose=0,
            )[0][0]
        )
    )

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
        conv_output,
    )

    pooled = tf.reduce_mean(
        grads,
        axis=(0, 1, 2),
    )

    conv = conv_output[0]

    heatmap = conv @ pooled[..., tf.newaxis]

    heatmap = tf.squeeze(
        heatmap
    )

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
# ANALYSIS
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

    if score >= 0.50:

        return {
            "label": "Higher review signal",
            "emoji": "🔴",
            "message": (
                "The model produced a relatively "
                "high quality-risk score."
            ),
            "action": (
                "Review the image carefully before "
                "considering it acceptable for the intended "
                "clinical task."
            ),
        }

    elif score >= THRESHOLD:

        return {
            "label": "Review recommended",
            "emoji": "🟠",
            "message": (
                "The score has reached the project's "
                "review threshold."
            ),
            "action": (
                "Take a closer look at image noise, "
                "anatomical detail and whether the "
                "relevant clinical information remains visible."
            ),
        }

    else:

        return {
            "label": "No automatic flag",
            "emoji": "🟢",
            "message": (
                "The model score is below the project's "
                "review threshold."
            ),
            "action": (
                "No AI review flag was generated. "
                "Normal professional image-quality assessment "
                "still applies."
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

            <div class="hero-title">
                CT Image Quality Flagger
            </div>

            <p class="hero-text">
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
        [1.35, 1],
        gap="large",
    )

    with left:

        st.markdown(
            "### Hi, I'm Zainab 👋"
        )

        st.write(
            """
            I'm a Medical Imaging Technology student interested in
            how artificial intelligence can support safer and more
            consistent medical-imaging workflows.

            I built this prototype around a simple question:

            **When CT dose is reduced, how can we identify images
            that may deserve a closer quality review?**
            """
        )

        if st.button(
            "Start CT Analysis →",
            type="primary",
            use_container_width=True,
        ):

            navigate("Analyze")

    with right:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
                What this tool does
            </div>

            <div class="card-text">

            <b>1. Analyze</b><br>
            The model produces a quality-risk score.

            <br><br>

            <b>2. Flag</b><br>
            Images reaching the project threshold receive a review signal.

            <br><br>

            <b>3. Explain</b><br>
            Grad-CAM provides a visual indication of regions
            contributing to the prediction.

            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.info(
        """
        **Important:** This is an educational/research prototype.
        It is not clinically validated and should not be used to
        diagnose patients, automatically reject scans, or change
        CT radiation-dose protocols.
        """
    )


# ============================================================
# ANALYZE
# ============================================================

elif st.session_state.page == "Analyze":

    st.markdown(
        '<div class="eyebrow">CT ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    st.title(
        "Choose how to start"
    )

    st.caption(
        "Upload your own CT image or explore a prepared demonstration case."
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

        option1, option2 = st.columns(
            2,
            gap="large",
        )

        with option1:

            st.markdown(
                """
                <div class="card">

                <div class="card-title">
                    📤 Upload your CT
                </div>

                <div class="card-text">
                Analyze a DICOM, PNG or JPG image.
                DICOM is preferred when acquisition information
                is also needed.
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            uploaded = st.file_uploader(
                "Choose CT image(s)",
                type=[
                    "dcm",
                    "png",
                    "jpg",
                    "jpeg",
                ],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )

            if uploaded:

                for file in uploaded:

                    try:

                        if any(
                            r["name"] == file.name
                            for r in st.session_state.results
                        ):
                            continue

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

                if st.button(
                    "View result →",
                    type="primary",
                    use_container_width=True,
                ):

                    navigate("Results")

        with option2:

            st.markdown(
                """
                <div class="card">

                <div class="card-title">
                    🧪 Demo cases
                </div>

                <div class="card-text">
                Not ready to upload an image?
                Explore prepared cases to see how the
                application works.
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            if st.button(
                "Explore demo cases →",
                use_container_width=True,
            ):

                navigate("Demo")


# ============================================================
# DEMO CASES
# ============================================================

elif st.session_state.page == "Demo":

    st.markdown(
        '<div class="eyebrow">DEMONSTRATION</div>',
        unsafe_allow_html=True,
    )

    st.title(
        "Explore Demo Cases"
    )

    st.caption(
        "These cases are included only to demonstrate the workflow."
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

    columns = st.columns(4)

    for i, (title, filename) in enumerate(
        demos
    ):

        path = os.path.join(
            demo_folder,
            filename,
        )

        with columns[i]:

            if os.path.exists(path):

                demo_image = (
                    Image
                    .open(path)
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

                st.warning(
                    f"{filename} not found."
                )

    st.write("")

    if st.button(
        "← Back to analysis",
    ):

        navigate("Analyze")


# ============================================================
# RESULTS
# ============================================================

elif st.session_state.page == "Results":

    st.markdown(
        '<div class="eyebrow">ANALYSIS RESULT</div>',
        unsafe_allow_html=True,
    )

    st.title(
        "CT Image Quality Result"
    )

    if not st.session_state.results:

        st.info(
            "No image has been analyzed yet."
        )

        if st.button(
            "Analyze a CT image →",
            type="primary",
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

        left, right = st.columns(
            [1, 1.5],
            gap="large",
        )

        with left:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="eyebrow">MODEL SIGNAL</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"## {info['emoji']} {info['label']}"
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
                "### What does this result mean?"
            )

            st.markdown(
                f"""
                <div class="interpretation">

                {info["message"]}

                <br><br>

                The score reflects how strongly the image
                resembles patterns the model learned to flag.
                It does <b>not</b> diagnose disease and does not
                determine clinical acceptability by itself.

                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # TECHNOLOGIST INTERPRETATION
        # ----------------------------------------------------

        st.write("")

        st.markdown(
            "### 👩‍⚕️ What should the technologist do?"
        )

        st.markdown(
            f"""
            <div class="action-box">

            <div class="action-title">
                {info["emoji"]} {info["action"]}
            </div>

            <div class="action-item">
            <b>1. Review the image.</b>
            Look for excessive noise, loss of anatomical detail,
            or other factors that could affect the intended examination.
            </div>

            <div class="action-item">
            <b>2. Consider the clinical task.</b>
            Ask whether the anatomy and information required for
            the examination can still be adequately assessed.
            </div>

            <div class="action-item">
            <b>3. Do not automatically repeat the scan.</b>
            An AI flag alone is not a reason to rescan a patient
            or increase radiation dose.
            </div>

            <div class="action-item">
            <b>4. Think beyond one image.</b>
            If similar quality concerns repeatedly occur under
            the same protocol, the pattern may be useful during
            image-quality or protocol review.

            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # VISUAL EXPLANATION
        # ----------------------------------------------------

        st.write("")

        st.markdown(
            "### 👁️ Why did the model flag this image?"
        )

        st.caption(
            "Grad-CAM is a model-interpretation aid. "
            "It does not identify disease or prove that a highlighted "
            "area is abnormal."
        )

        image_col, heatmap_col = st.columns(
            2,
            gap="large",
        )

        with image_col:

            st.image(
                result["image"],
                caption="Input image",
                use_container_width=True,
            )

        with heatmap_col:

            st.image(
                result["gradcam"],
                caption="Grad-CAM visualization",
                use_container_width=True,
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
            "What does the 0.25 threshold mean?"
        ):

            st.write(
                """
                A threshold is simply the decision point used by
                this prototype to generate a review flag.

                A score below 0.25 produces no automatic flag.

                A score of 0.25 or higher produces a review signal.

                This value is specific to this research prototype.
                It is not a universal clinical cutoff.
                """
            )

        st.warning(
            """
            **Clinical safety:** This prototype is not clinically validated.
            Do not use the model output alone to diagnose disease,
            reject a clinical examination, repeat a scan, or change
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

    st.title(
        "How the tool works"
    )

    st.caption(
        "A short guide to the terms you may see in the results."
    )

    with st.expander(
        "🧠 What is the quality-risk score?"
    ):

        st.write(
            """
            The model produces a number between 0 and 1.

            A higher score means the image more strongly resembles
            the pattern the model was trained to flag.

            It is a model signal, not a clinical measurement.
            """
        )

    with st.expander(
        "👁️ What is Grad-CAM?"
    ):

        st.write(
            """
            Grad-CAM creates a heatmap showing image regions that
            contributed to the model's prediction.

            Think of it as:

            "Which parts of the image influenced the model?"

            It does not prove that the highlighted region is abnormal.
            """
        )

    with st.expander(
        "🔄 Why does the app use several image views?"
    ):

        st.write(
            """
            The prototype uses Test-Time Augmentation (TTA).

            The same image is evaluated in five slightly different
            forms and the predictions are averaged.

            This is intended to make the model less sensitive to
            small presentation changes.
            """
        )

    with st.expander(
        "🩻 Why DICOM?"
    ):

        st.write(
            """
            DICOM is the standard format used for medical imaging.

            Besides the image, it can contain acquisition information
            such as tube voltage, tube current and slice thickness.
            """
        )

    with st.expander(
        "📊 How was the model evaluated?"
    ):

        st.write(
            """
            In the project evaluation, testing was performed on
            fully held-out patients.

            Reported results were:

            • Recall: 85%
            • Precision: 54%
            • F1-score: 0.66
            • ROC-AUC: 0.839
            • Full-dose false-positive rate: 3.3%

            These results describe this prototype's evaluation.
            They do not establish clinical effectiveness.
            """
        )

    st.write("")

    st.info(
        """
        **Project limitation:** The training data contained
        paired full-dose and low-dose CT images with substantial
        quality differences. Real clinical dose reductions can be
        more subtle. Further validation on larger, diverse,
        real-world datasets would therefore be necessary.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CT Image Quality Flagger • Zainab Fatima • "
    "Medical Imaging Technology • Educational/research prototype only"
)
