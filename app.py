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
# PAGE CONFIG — MUST BE BEFORE ANY STREAMLIT UI
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
# PROFESSIONAL CSS
# IMPORTANT: CSS ONLY — NO VISIBLE HTML CONTENT
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background-color: #f6f8fb;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ---------- TOP BRAND ---------- */

    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #102a43;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 0.15rem;
    }

    /* ---------- NAVIGATION ---------- */

    div.stButton > button {
        border-radius: 10px;
        min-height: 2.55rem;
        font-weight: 650;
        border: 1px solid #d8e2ea;
        background: white;
        color: #17324d;
    }

    div.stButton > button:hover {
        border-color: #247ba0;
        color: #247ba0;
    }

    /* ---------- HEADINGS ---------- */

    h1 {
        color: #102a43 !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em;
    }

    h2, h3 {
        color: #17324d !important;
    }

    /* ---------- CARDS ---------- */

    .small-muted {
        color: #64748b;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    .score-number {
        font-size: 3.2rem;
        font-weight: 800;
        color: #102a43;
        line-height: 1;
    }

    .score-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .result-box {
        padding: 1rem;
        border-radius: 12px;
        background: #eef7fa;
        border-left: 4px solid #247ba0;
        line-height: 1.6;
        color: #334e68;
    }

    /* ---------- UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #9bb9c8;
        border-radius: 14px;
        padding: 0.5rem;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 700px) {

        .main .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
            padding-top: 1rem;
        }

        .brand-title {
            font-size: 1rem;
        }

        .brand-subtitle {
            font-size: 0.7rem;
        }

        h1 {
            font-size: 2rem !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


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
# TOP NAVIGATION
# ============================================================

brand, home_btn, analyze_btn, results_btn, learn_btn = st.columns(
    [2.6, 1, 1, 1, 1],
    vertical_alignment="center",
)

with brand:
    st.markdown(
        '<div class="brand-title">🩻 CT Image Quality Flagger</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="brand-subtitle">AI-assisted CT image-quality research prototype</div>',
        unsafe_allow_html=True,
    )

with home_btn:
    if st.button("Home", use_container_width=True):
        navigate("Home")

with analyze_btn:
    if st.button("Analyze", use_container_width=True):
        navigate("Analyze")

with results_btn:
    if st.button("Results", use_container_width=True):
        navigate("Results")

with learn_btn:
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


try:
    model = load_model()
    MODEL_READY = True
    MODEL_ERROR = None

except Exception as exc:
    model = None
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
# MODEL PREDICTION + TTA
# ============================================================

def predict_single(model_input):

    prediction = model.predict(
        model_input,
        verbose=0,
    )

    return float(prediction[0][0])


def get_score(model_input):

    predictions = []

    # Original
    predictions.append(
        predict_single(model_input)
    )

    # Horizontal flip
    flipped = np.flip(
        model_input,
        axis=2,
    )

    predictions.append(
        predict_single(flipped)
    )

    # Small rotations
    image = model_input[0]

    center = (
        IMAGE_SIZE[0] // 2,
        IMAGE_SIZE[1] // 2,
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
            predict_single(rotated)
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
        predict_single(zoomed)
    )

    return float(
        np.mean(predictions)
    )


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam(image_array, model_input):

    try:

        base_model = model.layers[0]

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                base_model.get_layer(
                    GRADCAM_LAYER
                ).output,
                model.output,
            ],
        )

        with tf.GradientTape() as tape:

            conv_output, prediction = grad_model(
                model_input,
                training=False,
            )

            loss = prediction[:, 0]

        gradients = tape.gradient(
            loss,
            conv_output,
        )

        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(1, 2),
        )

        conv_output = conv_output[0]

        pooled_gradients = pooled_gradients[0]

        heatmap = tf.reduce_sum(
            conv_output
            * pooled_gradients[tf.newaxis, tf.newaxis, :],
            axis=-1,
        )

        heatmap = tf.maximum(
            heatmap,
            0,
        )

        maximum = tf.reduce_max(
            heatmap
        )

        heatmap = heatmap / (
            maximum + 1e-8
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

        return overlay

    except Exception:

        # If Grad-CAM cannot be generated,
        # return the original image instead of crashing the app.
        return image_array


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
# RESULT INTERPRETATION
# ============================================================

def interpretation(score):

    if score >= 0.50:

        return {
            "title": "Higher review priority",
            "icon": "🔴",
            "action": "Closer review recommended",
            "message": (
                "The model produced a relatively high "
                "quality-risk score. The image deserves "
                "closer technical review."
            ),
        }

    elif score >= THRESHOLD:

        return {
            "title": "Review signal",
            "icon": "🟠",
            "action": "Review the image",
            "message": (
                "The score reached the project's review "
                "threshold. The technologist should take "
                "a closer look at image quality before "
                "interpreting the result."
            ),
        }

    elif score >= 0.15:

        return {
            "title": "Lower-risk pattern",
            "icon": "🟡",
            "action": "No automatic flag",
            "message": (
                "The score is below the project's review "
                "threshold. The model did not trigger a "
                "review signal, but this does not prove "
                "that image quality is clinically acceptable."
            ),
        }

    else:

        return {
            "title": "Low-risk pattern",
            "icon": "🟢",
            "action": "No automatic flag",
            "message": (
                "The model produced a relatively low "
                "quality-risk score."
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
        item["name"]
        for item in st.session_state.results
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

    st.write("")

    st.markdown(
        "### MEDICAL IMAGING TECHNOLOGY × AI"
    )

    st.title(
        "CT Image Quality Flagger"
    )

    st.write(
        "An AI-assisted research prototype that flags CT images "
        "showing patterns associated with reduced image quality, "
        "helping the technologist decide which images deserve "
        "a closer review."
    )

    st.write("")

    # --------------------------------------------------------
    # INTRO
    # --------------------------------------------------------

    intro_left, intro_right = st.columns(
        [1.6, 1],
        gap="large",
    )

    with intro_left:

        with st.container(border=True):

            st.markdown(
                "### Hi, I'm Zainab 👋"
            )

            st.write(
                "I'm a Medical Imaging Technology student interested "
                "in how AI can support safer and more consistent "
                "medical-imaging workflows."
            )

            st.write(
                "This project began with a simple question:"
            )

            st.markdown(
                "**When CT dose is reduced, how can we identify "
                "images that may need a closer quality review?**"
            )

            st.write(
                "The goal is not to replace the technologist or "
                "radiologist. It is to provide an additional signal "
                "that can support image-quality review."
            )

    with intro_right:

        with st.container(border=True):

            st.markdown(
                "### What this prototype does"
            )

            st.write(
                "🩻 Analyzes a CT image"
            )

            st.write(
                "📊 Produces a quality-risk score"
            )

            st.write(
                "👁️ Shows a Grad-CAM explanation"
            )

            st.write(
                "📋 Reads selected DICOM information"
            )

    st.write("")

    # --------------------------------------------------------
    # START OPTIONS
    # --------------------------------------------------------

    st.markdown(
        "### Start here"
    )

    option1, option2 = st.columns(
        2,
        gap="large",
    )

    with option1:

        with st.container(border=True):

            st.markdown(
                "### 📤 Analyze your CT"
            )

            st.write(
                "Upload a DICOM, PNG or JPG image and "
                "get the model's result."
            )

            if st.button(
                "Upload & Analyze →",
                type="primary",
                use_container_width=True,
            ):
                navigate("Analyze")

    with option2:

        with st.container(border=True):

            st.markdown(
                "### 🧪 Try a Demo Case"
            )

            st.write(
                "Explore prepared cases to see how the "
                "prototype works before uploading your own image."
            )

            if st.button(
                "View Demo Cases →",
                use_container_width=True,
            ):
                navigate("Demos")

    st.write("")

    # --------------------------------------------------------
    # WHY IT MATTERS
    # --------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### Why this matters"
        )

        st.write(
            "CT dose optimization is not simply about using less "
            "radiation. Reducing dose can increase image noise and "
            "may eventually affect image quality."
        )

        st.write(
            "A future version of this concept could help identify "
            "quality patterns consistently across large numbers of "
            "images and support protocol-level quality audits."
        )

    st.write("")

    st.info(
        "Research prototype only. This tool is not clinically "
        "validated and must not be used to diagnose patients, "
        "reject clinical scans, or independently change CT protocols."
    )


# ============================================================
# ANALYZE PAGE
# ============================================================

elif st.session_state.page == "Analyze":

    st.markdown(
        "### CT IMAGE ANALYSIS"
    )

    st.title(
        "Upload a CT Image"
    )

    st.write(
        "Upload a CT image to see the model's quality-risk score "
        "and visual explanation."
    )

    if not MODEL_READY:

        st.error(
            "The AI model could not be loaded."
        )

        st.code(
            MODEL_ERROR
        )

    else:

        uploaded_files = st.file_uploader(
            "Choose CT image",
            type=[
                "dcm",
                "png",
                "jpg",
                "jpeg",
            ],
            accept_multiple_files=True,
            help=(
                "DICOM is preferred because it can contain "
                "additional acquisition information."
            ),
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

                    st.exception(exc)

            if st.session_state.selected_result:

                if st.button(
                    "View Result →",
                    type="primary",
                ):
                    navigate("Results")

    st.divider()

    with st.expander(
        "What does the score mean?"
    ):

        st.write(
            "The model produces a quality-risk score from 0 to 1. "
            "A higher score means the image more strongly resembles "
            "patterns the model was trained to flag."
        )

        st.write(
            f"The current project threshold is **{THRESHOLD:.2f}**."
        )

        st.write(
            "This threshold is specific to this research prototype. "
            "It is not a universal clinical cutoff."
        )


# ============================================================
# DEMO CASES
# ============================================================

elif st.session_state.page == "Demos":

    st.markdown(
        "### DEMONSTRATION"
    )

    st.title(
        "Explore Demo Cases"
    )

    st.write(
        "Try the prepared cases to understand how the "
        "prototype produces and explains its output."
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

        cols = st.columns(
            4,
            gap="medium",
        )

        for i, (title, filename) in enumerate(
            demos
        ):

            path = os.path.join(
                demo_folder,
                filename,
            )

            with cols[i]:

                with st.container(
                    border=True
                ):

                    if os.path.exists(path):

                        demo_image = (
                            Image.open(path)
                            .convert("RGB")
                        )

                        st.image(
                            demo_image,
                            use_container_width=True,
                        )

                        st.markdown(
                            f"**{title}**"
                        )

                        if st.button(
                            "Analyze",
                            key=f"demo_{i}",
                            use_container_width=True,
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
                            f"{title} is unavailable."
                        )

    st.write("")

    st.info(
        "The demonstration cases are included to show how "
        "the application works. They should not be interpreted "
        "as clinical validation cases."
    )


# ============================================================
# RESULTS PAGE
# ============================================================

elif st.session_state.page == "Results":

    st.markdown(
        "### ANALYSIS RESULT"
    )

    st.title(
        "CT Image Quality Result"
    )

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
            result["name"]
            for result in st.session_state.results
        ]

        selected = st.selectbox(
            "Select an analyzed image",
            names,
        )

        result = next(
            item
            for item in st.session_state.results
            if item["name"] == selected
        )

        score = result["score"]

        result_info = interpretation(
            score
        )

        # ----------------------------------------------------
        # MAIN RESULT
        # ----------------------------------------------------

        left, right = st.columns(
            [1, 1.6],
            gap="large",
        )

        with left:

            with st.container(
                border=True
            ):

                st.markdown(
                    "##### MODEL OUTPUT"
                )

                st.markdown(
                    f"## {result_info['icon']} "
                    f"{result_info['title']}"
                )

                st.markdown(
                    '<div class="score-label">'
                    'QUALITY-RISK SCORE'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="score-number">'
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
                    f"**{result_info['action']}**"
                )

        with right:

            st.markdown(
                "### What should the technologist do?"
            )

            if score >= THRESHOLD:

                st.warning(
                    "Review signal triggered"
                )

                st.write(
                    "Take a closer look at the image for "
                    "possible quality concerns such as increased "
                    "noise or loss of useful anatomical detail."
                )

                st.write(
                    "**Important:** This flag does not mean "
                    "the scan must automatically be repeated."
                )

                st.write(
                    "If image quality is genuinely non-diagnostic, "
                    "the appropriate action should follow the "
                    "department's normal clinical and technical "
                    "workflow."
                )

            else:

                st.success(
                    "No automatic review flag"
                )

                st.write(
                    "The model did not cross the project's "
                    "review threshold."
                )

                st.write(
                    "Continue normal image-quality assessment. "
                    "A low model score does not guarantee "
                    "clinical acceptability."
                )

        st.divider()

        # ----------------------------------------------------
        # SIMPLE INTERPRETABILITY
        # ----------------------------------------------------

        st.markdown(
            "### 👁️ Why did the model give this result?"
        )

        st.write(
            "Grad-CAM provides a visual explanation of where "
            "the model's prediction was influenced."
        )

        image_col, heatmap_col = st.columns(
            2,
            gap="large",
        )

        with image_col:

            st.image(
                result["image"],
                caption="Input CT image",
                use_container_width=True,
            )

        with heatmap_col:

            st.image(
                result["gradcam"],
                caption="Grad-CAM — model attention",
                use_container_width=True,
            )

        st.caption(
            "Important: Grad-CAM shows model attention, "
            "not disease, pathology, or a clinically validated "
            "quality map."
        )

        # ----------------------------------------------------
        # TECHNICIAN GUIDE
        # ----------------------------------------------------

        st.divider()

        with st.container(
            border=True
        ):

            st.markdown(
                "### Quick review guide"
            )

            st.write(
                "When a review signal appears, the technologist "
                "can use the normal image-quality workflow to "
                "check the scan."
            )

            guide1, guide2, guide3 = st.columns(
                3
            )

            with guide1:

                st.markdown(
                    "**1 · Check noise**"
                )

                st.caption(
                    "Is image noise noticeably limiting "
                    "visualization of anatomy?"
                )

            with guide2:

                st.markdown(
                    "**2 · Check diagnostic detail**"
                )

                st.caption(
                    "Can the relevant anatomy and required "
                    "structures still be adequately assessed?"
                )

            with guide3:

                st.markdown(
                    "**3 · Follow clinical workflow**"
                )

                st.caption(
                    "If quality is inadequate, follow your "
                    "department's established protocol."
                )

        # ----------------------------------------------------
        # DICOM
        # ----------------------------------------------------

        if result["metadata"]:

            st.divider()

            with st.expander(
                "📋 DICOM acquisition information"
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
        # SCORE EXPLANATION
        # ----------------------------------------------------

        with st.expander(
            "What is the 0.25 threshold?"
        ):

            st.write(
                "For this prototype, scores of 0.25 or above "
                "generate a review signal."
            )

            st.write(
                "The threshold was selected for this project's "
                "research evaluation. It is not a universal "
                "clinical cutoff."
            )

        st.warning(
            "Research prototype only. Do not use this output "
            "alone to diagnose disease, reject clinical scans, "
            "or change CT radiation-dose protocols."
        )


# ============================================================
# LEARN PAGE
# ============================================================

elif st.session_state.page == "Learn":

    st.markdown(
        "### PROJECT GUIDE"
    )

    st.title(
        "How the Prototype Works"
    )

    st.write(
        "A few short explanations for the technical terms "
        "you may see in the results."
    )

    with st.expander(
        "🧠 What is VGG16?"
    ):

        st.write(
            "VGG16 is a deep-learning model that learns visual "
            "patterns from images. This project uses transfer "
            "learning, adapting learned visual features to "
            "the CT image-quality task."
        )

    with st.expander(
        "🔄 What is Test-Time Augmentation?"
    ):

        st.write(
            "The same image is presented to the model in several "
            "slightly modified forms, such as a flip, small "
            "rotations and a small crop. The predictions are "
            "then averaged to produce the final score."
        )

    with st.expander(
        "👁️ What is Grad-CAM?"
    ):

        st.write(
            "Grad-CAM is an explainability method. It highlights "
            "regions that contributed to a model prediction."
        )

        st.write(
            "Think of it as asking: "
            "**'Which parts of the image influenced the model?'**"
        )

        st.write(
            "It does not prove that the highlighted region is "
            "abnormal or clinically important."
        )

    with st.expander(
        "🩻 What is DICOM?"
    ):

        st.write(
            "DICOM is a standard format used for medical images. "
            "Unlike an ordinary JPG, a DICOM file can also contain "
            "information about image acquisition."
        )

    with st.expander(
        "📊 What is the quality-risk score?"
    ):

        st.write(
            "The score is the model's numerical output. "
            "A higher value means the image more strongly "
            "resembles the pattern the model was trained to flag."
        )

        st.write(
            "It is not a radiation-dose measurement and it is "
            "not a clinical diagnosis."
        )

    with st.expander(
        "🎯 What does the review threshold mean?"
    ):

        st.write(
            f"This project uses **{THRESHOLD:.2f}** as the "
            "review threshold."
        )

        st.write(
            "At or above the threshold → review signal."
        )

        st.write(
            "Below the threshold → no automatic flag."
        )

        st.write(
            "This rule belongs only to this research prototype."
        )

    with st.expander(
        "📈 How was the model evaluated?"
    ):

        st.write(
            "The project evaluation used patient-level held-out "
            "testing."
        )

        st.write(
            "**Reported evaluation:**"
        )

        st.write(
            "• Recall: 85%\n"
            "• Precision: 54%\n"
            "• F1-score: 0.66\n"
            "• ROC-AUC: 0.839\n"
            "• Full-dose false-positive rate: 3.3%"
        )

        st.caption(
            "These are research evaluation metrics and do not "
            "establish clinical effectiveness."
        )

    st.divider()

    st.markdown(
        "### A limitation worth knowing"
    )

    st.write(
        "The prototype was developed using paired full-dose and "
        "low-dose chest CT data in which the quality difference "
        "could be relatively noticeable."
    )

    st.write(
        "Real clinical dose reductions can be much subtler. "
        "Therefore, future validation should test the approach "
        "on more subtle dose variations, different patients, "
        "scanners and institutions."
    )

    st.info(
        "The purpose of this prototype is to explore the concept "
        "of AI-assisted image-quality review — not to replace "
        "professional judgment."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CT Image Quality Flagger  •  Zainab Fatima  •  "
    "Medical Imaging Technology  •  Educational / research prototype"
)
