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

brand, home_btn, analyze_btn, demos_btn, results_btn, learn_btn = st.columns(
    [2.2, 1, 1, 1, 1, 1],
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

with demos_btn:
    if st.button("Demos", use_container_width=True):
        navigate("Demos")

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

def find_base_model(full_model, target_layer_name):
    """
    Locate the nested sub-model (e.g. VGG16) inside `full_model`
    that actually contains `target_layer_name`, instead of
    assuming it is always full_model.layers[0]. Many transfer-
    learning models put an InputLayer first, so hardcoding
    index 0 silently grabs the wrong layer and Grad-CAM ends up
    producing nothing useful.
    Returns (base_model, index_of_base_model_in_full_model.layers).
    """

    for index, layer in enumerate(full_model.layers):

        # A nested model (VGG16 base) exposes its own .layers list.
        if hasattr(layer, "layers"):

            try:
                layer.get_layer(target_layer_name)
                return layer, index

            except (ValueError, Exception):
                continue

    # Fallback: maybe the target layer sits directly on the
    # outer model (no nesting at all).
    try:
        full_model.get_layer(target_layer_name)
        return full_model, -1
    except Exception:
        pass

    return None, None


def _finish_heatmap(heatmap, image_array):

    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = heatmap / (maximum + 1e-8)

    heatmap = cv2.resize(
        heatmap.numpy(),
        IMAGE_SIZE,
        interpolation=cv2.INTER_CUBIC,
    )

    heatmap = np.clip(heatmap, 0, 1)

    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB,
    )

    image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    overlay = cv2.addWeighted(
        image_array,
        0.60,
        heatmap_color,
        0.40,
        0,
    )

    return overlay


def _gradcam_direct(image_array, model_input, base_model, base_index):
    """
    Approach 1: build one Model from the outer model's inputs to
    [target conv layer output, final prediction]. Works when the
    base model is nested cleanly inside a Functional model.
    """

    if base_index == -1:
        target_output = base_model.get_layer(GRADCAM_LAYER).output
    else:
        target_output = base_model.get_layer(GRADCAM_LAYER).output

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_output, model.output],
    )

    with tf.GradientTape() as tape:

        conv_output, prediction = grad_model(
            model_input,
            training=False,
        )

        tape.watch(conv_output)

        loss = prediction[:, 0]

    gradients = tape.gradient(loss, conv_output)

    if gradients is None:
        raise ValueError(
            "Gradient computation returned None — the conv layer "
            "output is disconnected from the model's output graph."
        )

    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))[0]
    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(
        conv_output * pooled_gradients[tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    return _finish_heatmap(heatmap, image_array)


def _gradcam_two_stage(image_array, model_input, base_model, base_index):
    """
    Approach 2 (fallback): some transfer-learning models raise a
    'graph disconnected' error with the direct approach because the
    nested base model's layer.output tensor was created in a
    different call context than the outer model. This rebuilds the
    forward pass explicitly in two stages so gradients can flow:
    conv features -> rest of base model -> head layers -> prediction.
    """

    target_layer = base_model.get_layer(GRADCAM_LAYER)

    dual_extractor = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[target_layer.output, base_model.output],
    )

    head_layers = model.layers[base_index + 1:] if base_index >= 0 else []

    inputs = tf.convert_to_tensor(model_input, dtype=tf.float32)

    with tf.GradientTape() as tape:

        conv_output, base_features = dual_extractor(
            inputs,
            training=False,
        )

        tape.watch(conv_output)

        x = base_features

        for layer in head_layers:
            x = layer(x, training=False)

        loss = x[:, 0]

    gradients = tape.gradient(loss, conv_output)

    if gradients is None:
        raise ValueError(
            "Gradient computation returned None in the two-stage "
            "fallback as well."
        )

    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))[0]
    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(
        conv_output * pooled_gradients[tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    return _finish_heatmap(heatmap, image_array)


def make_gradcam(image_array, model_input):
    """
    Returns (overlay_image, error_message_or_None). The caller
    decides how to surface a failure — we no longer swallow the
    error silently, since a silent fallback to the plain image
    looked identical to a working-but-uninformative Grad-CAM and
    made the bug impossible to diagnose from the UI.
    """

    base_model, base_index = find_base_model(model, GRADCAM_LAYER)

    if base_model is None:
        return image_array, (
            f"Could not find a layer named '{GRADCAM_LAYER}' "
            "anywhere in the model (checked nested sub-models too). "
            "Check GRADCAM_LAYER against the actual base model's "
            "layer names."
        )

    try:
        return _gradcam_direct(
            image_array, model_input, base_model, base_index
        ), None

    except Exception as direct_error:

        try:
            return _gradcam_two_stage(
                image_array, model_input, base_model, base_index
            ), None

        except Exception as fallback_error:

            error_message = (
                "Direct approach failed: "
                f"{direct_error}\n\n"
                "Two-stage fallback also failed: "
                f"{fallback_error}"
            )

            return image_array, error_message


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

    gradcam, gradcam_error = make_gradcam(
        arr,
        model_input,
    )

    return (
        arr,
        gradcam,
        score,
        gradcam_error,
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
            "title": "Borderline — below threshold",
            "icon": "🟡",
            "action": "No automatic flag",
            "message": (
                "The score is below the project's review "
                "threshold, but close to it. The model did "
                "not trigger a review signal, but this does "
                "not prove that image quality is clinically "
                "acceptable."
            ),
        }

    else:

        return {
            "title": "Low quality-risk",
            "icon": "🟢",
            "action": "No automatic flag",
            "message": (
                "The model produced a relatively low "
                "quality-risk score, well below the review "
                "threshold."
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
    gradcam_error=None,
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
                "gradcam_error": gradcam_error,
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
    # KEY RESULTS STRIP
    # --------------------------------------------------------

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric("Recall (sensitivity)", "85%")

    with kpi2:
        st.metric("ROC-AUC", "0.839")

    with kpi3:
        st.metric("Full-dose false-positive rate", "3.3%")

    with kpi4:
        st.metric("Chest CT patients used", "21")

    st.caption(
        "Patient-level held-out evaluation, with Test-Time "
        "Augmentation. See the Learn page for full methodology."
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

            st.write(
                "📈 Summarizes patterns across a batch of images"
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
            "This question is not only technical. Studies on CT "
            "dose practice in Pakistan (e.g. Yaseen et al., 2024) "
            "have documented wide dose variation across institutions "
            "and the absence of established diagnostic reference "
            "levels — the kind of setting where a lightweight, "
            "low-cost image-quality signal could plausibly matter."
        )

        st.write(
            "This prototype does not decide the final dose. Dose "
            "decisions belong to physicists and radiologists, "
            "weighing image quality against radiation-safety "
            "reference levels. Used at the protocol level, patterns "
            "flagged across many scans could help an institution "
            "notice when a dose-reduction protocol may be too "
            "aggressive — rather than each flag triggering an "
            "individual re-scan."
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
        "Upload one or more CT images to see the model's "
        "quality-risk score and visual explanation for each."
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
            "Choose CT image(s)",
            type=[
                "dcm",
                "png",
                "jpg",
                "jpeg",
            ],
            accept_multiple_files=True,
            help=(
                "DICOM is preferred because it can contain "
                "additional acquisition information. Upload "
                "several images at once to see a batch summary "
                "on the Results page."
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

                        arr, gradcam, score, gradcam_error = analyze(
                            image
                        )

                    add_result(
                        file.name,
                        arr,
                        gradcam,
                        score,
                        metadata,
                        gradcam_error,
                    )

                    if gradcam_error:
                        st.success(
                            f"{file.name} analyzed successfully "
                            "(score computed normally)."
                        )
                        st.warning(
                            "Grad-CAM explanation could not be "
                            "generated for this image — see the "
                            "Results page for details."
                        )
                    else:
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

                                arr, gradcam, score, gradcam_error = (
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
                                gradcam_error,
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

        # ----------------------------------------------------
        # BATCH SUMMARY (protocol-level view)
        # ----------------------------------------------------
        # Shown only when more than one image has been analyzed.
        # A single image's score is a per-image quality signal;
        # a pattern across many images is closer to how this
        # concept would actually be used — to notice whether a
        # dose-reduction protocol looks too aggressive overall.

        if len(st.session_state.results) > 1:

            with st.container(border=True):

                st.markdown(
                    "### 📈 Batch summary — protocol-level view"
                )

                st.caption(
                    "A single flagged image is a per-image signal. "
                    "A pattern across a batch is closer to how this "
                    "tool is meant to be used: noticing whether a "
                    "dose-reduction protocol looks too aggressive "
                    "overall, not deciding on any one patient."
                )

                batch_rows = [
                    {
                        "Image": item["name"],
                        "Score": round(item["score"], 3),
                        "Flagged": (
                            "Yes"
                            if item["score"] >= THRESHOLD
                            else "No"
                        ),
                    }
                    for item in st.session_state.results
                ]

                batch_df = pd.DataFrame(batch_rows)

                n_total = len(batch_df)
                n_flagged = int((batch_df["Flagged"] == "Yes").sum())
                flag_rate = n_flagged / n_total if n_total else 0
                avg_score = batch_df["Score"].mean()

                b1, b2, b3 = st.columns(3)

                with b1:
                    st.metric("Images analyzed", n_total)

                with b2:
                    st.metric(
                        "Flagged for review",
                        f"{n_flagged} ({flag_rate:.0%})",
                    )

                with b3:
                    st.metric("Average score", f"{avg_score:.3f}")

                if flag_rate >= 0.5:
                    st.warning(
                        "More than half of this batch was flagged. "
                        "If these images share a scanner or "
                        "protocol, that protocol's dose settings "
                        "may be worth reviewing."
                    )

                st.dataframe(
                    batch_df,
                    use_container_width=True,
                    hide_index=True,
                )

                csv_data = batch_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇️ Download batch summary (CSV)",
                    data=csv_data,
                    file_name="ct_quality_batch_summary.csv",
                    mime="text/csv",
                    use_container_width=False,
                )

            st.divider()

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

            st.caption(
                "🏥 **Intended future direction:** on its own, one "
                "flagged image is only a per-image signal. The "
                "actual goal for this concept is protocol-level "
                "review — if flagged patterns keep appearing across "
                "many scans on the same dose-reduction protocol, "
                "that is a signal for a hospital's physicists and "
                "radiologists to reconsider whether that protocol "
                "is calibrated correctly, not a trigger to rescan "
                "any single patient. See the batch summary above "
                "(when analyzing multiple images) for this view."
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

        if result.get("gradcam_error"):

            st.warning(
                "Grad-CAM could not be generated for this image, "
                "so the plain input image is shown on the right "
                "instead of a heatmap."
            )

            with st.expander("Debug details"):
                st.code(result["gradcam_error"])

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
                caption=(
                    "Grad-CAM — model attention"
                    if not result.get("gradcam_error")
                    else "Grad-CAM unavailable (see warning above)"
                ),
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
            "testing, so no slices from a test patient were ever "
            "seen during training."
        )

        st.write(
            "**Reported evaluation:**"
        )

        eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)

        with eval_col1:
            st.metric("Recall", "85%")

        with eval_col2:
            st.metric("Precision", "54%")

        with eval_col3:
            st.metric("F1-score", "0.66")

        with eval_col4:
            st.metric("ROC-AUC", "0.839")

        st.caption(
            "Full-dose false-positive rate: 3.3% — measured by "
            "running the model on undegraded full-dose images "
            "from the held-out test patients, none of which "
            "should trigger a review flag."
        )

        st.caption(
            "These are research evaluation metrics and do not "
            "establish clinical effectiveness."
        )

    with st.expander(
        "🧭 How is this meant to be used in practice?"
    ):

        st.write(
            "A single flagged image is a per-image signal for the "
            "technologist to take a closer look — it does not mean "
            "the scan must be repeated."
        )

        st.write(
            "The primary intended use is at the protocol level: if "
            "flagged patterns keep appearing across many scans on "
            "the same dose-reduction protocol, that is a signal for "
            "physicists and radiologists to reconsider whether the "
            "protocol is calibrated correctly — the same balance "
            "concept as an Acceptable Quality Dose (AQD)."
        )

        st.write(
            "This tool only supplies the image-quality signal. It "
            "does not decide the final radiation dose — that "
            "decision stays with physicists and radiologists, who "
            "weigh it against separately established Diagnostic "
            "Reference Levels (DRLs)."
        )

    st.divider()

    st.markdown(
        "### A limitation worth knowing"
    )

    st.write(
        "The prototype was developed using paired full-dose and "
        "low-dose chest CT data in which the quality difference "
        "could be relatively noticeable to the eye. This makes it "
        "a proof-of-concept on a clearer case, not evidence that "
        "the model outperforms manual review on subtler, more "
        "realistic dose reductions."
    )

    st.write(
        "Real clinical dose reductions can be much subtler. "
        "Therefore, future validation should test the approach "
        "on more subtle dose variations, different patients, "
        "scanners and institutions — where a consistent, "
        "scalable signal would matter more than it does on "
        "obvious cases."
    )

    st.info(
        "The purpose of this prototype is to explore the concept "
        "of AI-assisted image-quality review — not to replace "
        "professional judgment."
    )

    st.divider()

    st.markdown(
        "### References"
    )

    st.write(
        "This project's framing — balancing image quality against "
        "dose, rather than minimizing dose alone — draws on the "
        "Acceptable Quality Dose (AQD) concept as applied to CT "
        "practice in Pakistan:"
    )

    st.markdown(
        "- Yaseen, M., Nishtar, T., Kharita, M.H., et al. "
        "*Development of Acceptable Quality Dose (AQD) and "
        "image quality-related diagnostic reference levels for "
        "common computed tomography investigations in a tertiary "
        "care public sector hospital of Khyber Pakhtunkhwa, "
        "Pakistan.* Japanese Journal of Radiology, 42, 1479–1492 "
        "(2024). [doi.org/10.1007/s11604-024-01627-y]"
        "(https://doi.org/10.1007/s11604-024-01627-y)"
    )

    st.caption(
        "Cited for its dose-optimization framing (AQD/DRL), not "
        "as a source of training data — this prototype's dataset "
        "is the separate, public LDCT-and-Projection-data archive "
        "referenced below."
    )

    st.divider()

    st.markdown(
        "### About this project"
    )

    st.write(
        "Built independently by a Medical Imaging Technology "
        "student, without institutional data access, using the "
        "public LDCT-and-Projection-data dataset (The Cancer "
        "Imaging Archive)."
    )

    st.link_button(
        "🤗 Model on Hugging Face",
        f"https://huggingface.co/{HF_REPO}",
        use_container_width=False,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CT Image Quality Flagger  •  Zainab Fatima  •  "
    "Medical Imaging Technology  •  Educational / research prototype"
)
