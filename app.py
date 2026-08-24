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
# PAGE
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
# No raw HTML is used for visible interface text.
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
    }

    .stApp {
        background: #f7fafc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.1rem;
        padding-bottom: 3rem;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Top brand */
    .brand-name {
        font-size: 1.25rem;
        font-weight: 800;
        color: #102a43;
        margin-bottom: 0;
    }

    .brand-subtitle {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 0.15rem;
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0f2942 0%, #174e73 55%, #2b82a8 100%);
        border-radius: 24px;
        padding: 3.3rem 3.2rem;
        color: white;
        margin: 0.8rem 0 1.6rem 0;
        box-shadow: 0 16px 38px rgba(16,42,67,.15);
    }

    .hero-kicker {
        font-size: .74rem;
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
        max-width: 760px;
        font-size: 1.03rem;
        line-height: 1.65;
        margin: 0;
        color: rgba(255,255,255,.91);
    }

    /* Section cards */
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

    .explain {
        background: #eef7fa;
        border-left: 4px solid #247ba0;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        color: #334e68;
        line-height: 1.65;
    }

    .demo-label {
        color: #526174;
        font-size: .82rem;
        margin-top: -.3rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        min-height: 2.65rem;
        font-weight: 650;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border: 1.5px dashed #8db9ca;
        border-radius: 16px;
        padding: .6rem;
    }

    /* Mobile */
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
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TOP BAR
# ============================================================

brand, nav = st.columns([1.25, 2.75], vertical_alignment="center")

with brand:
    st.markdown("**🩻 CT Image Quality Flagger**")
    st.caption("AI-assisted CT image-quality research prototype")

with nav:
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
# IMAGE FUNCTIONS
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
    hu = ((hu - low) / (high - low) * 255)
    hu = np.clip(hu, 0, 255).astype(np.uint8)

    image = Image.fromarray(hu).convert("RGB").resize(IMAGE_SIZE)

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
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(image).astype(np.uint8)

    model_input = np.expand_dims(arr.astype(np.float32), axis=0)
    model_input = preprocess_input(model_input)

    return arr, model_input


def get_score(model_input):
    preds = []

    preds.append(float(model.predict(model_input, verbose=0)[0][0]))

    flipped = np.flip(model_input, axis=2)
    preds.append(float(model.predict(flipped, verbose=0)[0][0]))

    image = model_input[0]
    center = (112, 112)

    for angle in (-5, 5):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            matrix,
            IMAGE_SIZE,
            borderMode=cv2.BORDER_REFLECT,
        )
        rotated = np.expand_dims(rotated, axis=0)
        preds.append(float(model.predict(rotated, verbose=0)[0][0]))

    crop = image[11:213, 11:213]
    zoomed = cv2.resize(crop, IMAGE_SIZE)
    zoomed = np.expand_dims(zoomed, axis=0)
    preds.append(float(model.predict(zoomed, verbose=0)[0][0]))

    return float(np.mean(preds))


def make_gradcam(image_array, model_input):
    with tf.GradientTape() as tape:
        conv_output = conv_layer_model(model_input)
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

    grads = tape.gradient(loss, conv_output)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv = conv_output[0]

    heatmap = conv @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

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
        .60,
        heatmap_color,
        .40,
        0,
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def analyze(image):
    arr, model_input = prepare_image(image)
    score = get_score(model_input)
    gradcam = make_gradcam(arr, model_input)
    return arr, gradcam, score


def interpretation(score):
    if score >= .50:
        return (
            "Higher-risk pattern",
            "🔴",
            "Review signal triggered",
            "The model produced a relatively high quality-risk score.",
        )
    if score >= THRESHOLD:
        return (
            "Borderline risk",
            "🟠",
            "Review signal triggered",
            "The score has reached this project's review threshold.",
        )
    if score >= .15:
        return (
            "Lower-risk pattern",
            "🟡",
            "No automatic flag",
            "The score is below the project's review threshold, although some uncertainty remains.",
        )
    return (
        "Low-risk pattern",
        "🟢",
        "No automatic flag",
        "The model produced a relatively low quality-risk score.",
    )


def add_result(name, arr, gradcam, score, metadata):
    names = [r["name"] for r in st.session_state.results]
    if name not in names:
        st.session_state.results.append({
            "name": name,
            "image": arr,
            "gradcam": gradcam,
            "score": score,
            "metadata": metadata,
        })
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
            <h1>CT Image Quality Flagger</h1>
            <p>
                An AI-assisted research prototype exploring whether
                CT image-quality assessment can provide an additional
                signal during dose-optimization research.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.45, 1], gap="large")

    with left:
        st.markdown("### Hi, I'm Zainab 👋")
        st.write(
            """
            I'm a Medical Imaging Technology student interested in
            how artificial intelligence can support medical-imaging
            workflows.

            This project started with a practical question:

            **What happens when CT radiation dose is reduced, and
            how can we notice when image quality may need another look?**

            The aim is not to diagnose disease. The model provides a
            supportive research signal that can be explored alongside
            professional image assessment.
            """
        )

        if st.button(
            "🔬 Start CT Analysis",
            type="primary",
            use_container_width=True,
        ):
            navigate("Analyze")

    with right:
        st.markdown("### What can you do here?")
        st.markdown(
            """
            **🩻 Analyze a CT image**  
            Try a demonstration case or upload an image.

            **📊 Understand the score**  
            See what the model's output means in simple language.

            **👁️ Explore Grad-CAM**  
            See a heatmap of regions that influenced the prediction.

            **📋 Review DICOM information**  
            When available, view selected acquisition details.
            """
        )

    st.write("")
    st.markdown("### The idea in four steps")

    a, b, c, d = st.columns(4)

    with a:
        st.markdown("**01 — Upload**")
        st.caption("Provide a CT image in DICOM, PNG or JPG format.")

    with b:
        st.markdown("**02 — Analyze**")
        st.caption("VGG16 produces a quality-risk score.")

    with c:
        st.markdown("**03 — Understand**")
        st.caption("Grad-CAM helps visualize model attention.")

    with d:
        st.markdown("**04 — Review**")
        st.caption("The output is a research signal, not a clinical decision.")

    st.write("")
    st.warning(
        """
        **Important:** This is an educational/research prototype.
        It has not been clinically validated and should not be used
        to diagnose patients, accept/reject clinical scans, or change
        CT acquisition protocols.
        """
    )

# ============================================================
# ANALYZE
# ============================================================

elif st.session_state.page == "Analyze":

    st.markdown('<div class="eyebrow">CT ANALYSIS</div>', unsafe_allow_html=True)
    st.title("Analyze a CT Image")
    st.write(
        "Choose a demonstration case or upload your own image. "
        "Nothing is analyzed until you select an analysis action."
    )

    if not MODEL_READY:
        st.error("The AI model could not be loaded.")
        st.caption(MODEL_ERROR)

    st.markdown("### 🧪 Demonstration cases")
    st.caption(
        "These are included only to demonstrate how the application works."
    )

    demo_folder = "sample_images"
    demos = [
        ("Demo Case 1", "sample_acceptable_1.png"),
        ("Demo Case 2", "sample_acceptable_2.png"),
        ("Demo Case 3", "sample_flagged_1.png"),
        ("Demo Case 4", "sample_flagged_2.png"),
    ]

    cols = st.columns(4)

    for i, (title, filename) in enumerate(demos):
        path = os.path.join(demo_folder, filename)

        with cols[i]:
            if os.path.exists(path):
                demo_image = Image.open(path).convert("RGB")
                st.image(
                    demo_image,
                    caption=title,
                    use_container_width=True,
                )

                if st.button(
                    f"Try {title}",
                    key=f"demo_{i}",
                    use_container_width=True,
                    disabled=not MODEL_READY,
                ):
                    with st.spinner("Analyzing demonstration case..."):
                        arr, gradcam, score = analyze(demo_image)

                    add_result(title, arr, gradcam, score, {})
                    navigate("Results")
            else:
                st.info(f"{title} is not available.")

    st.divider()

    st.markdown("### 📤 Upload your CT image")

    st.write(
        """
        **DICOM (.dcm)** is preferred for CT because it can contain
        the image together with acquisition information.

        **PNG / JPG / JPEG** are also supported for demonstration
        and image-based testing.
        """
    )

    files = st.file_uploader(
        "Choose CT image(s)",
        type=["dcm", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="For large CT/DICOM files, the Streamlit server configuration controls the upload limit.",
    )

    if files and MODEL_READY:
        for file in files:
            if any(r["name"] == file.name for r in st.session_state.results):
                continue

            try:
                with st.spinner(f"Analyzing {file.name}..."):
                    if file.name.lower().endswith(".dcm"):
                        image, metadata = dicom_to_image(file.read())
                    else:
                        image = Image.open(file).convert("RGB")
                        metadata = {}

                    arr, gradcam, score = analyze(image)

                add_result(file.name, arr, gradcam, score, metadata)

                st.success(f"{file.name} analyzed.")
            except Exception as exc:
                st.error(f"Could not analyze {file.name}.")
                st.caption(str(exc))

        if files:
            st.button(
                "View latest result →",
                type="primary",
                on_click=navigate,
                args=("Results",),
            )

    st.divider()

    st.markdown("### Before you interpret the result")

    st.markdown(
        """
        <div class="explain">
        <b>Quality-risk score</b> is the model's numerical output.
        A higher score means the image more strongly resembles the
        pattern the model was trained to flag.

        The project's review threshold is <b>0.25</b>. This is a
        project-specific research threshold, not a universal clinical cutoff.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# RESULTS
# ============================================================

elif st.session_state.page == "Results":

    st.markdown('<div class="eyebrow">ANALYSIS RESULT</div>', unsafe_allow_html=True)
    st.title("Your Result")

    if not st.session_state.results:
        st.info("No images have been analyzed yet.")
        if st.button("Go to CT Analysis", type="primary"):
            navigate("Analyze")
    else:
        names = [r["name"] for r in st.session_state.results]

        selected = st.selectbox(
            "Select an analyzed image",
            names,
            index=max(
                0,
                names.index(st.session_state.selected_result)
                if st.session_state.selected_result in names else 0,
            ),
        )

        result = next(r for r in st.session_state.results if r["name"] == selected)
        score = result["score"]

        category, emoji, recommendation, explanation = interpretation(score)

        top1, top2 = st.columns([1.1, 1.9], gap="large")

        with top1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="eyebrow">MODEL ASSESSMENT</div>', unsafe_allow_html=True)
            st.markdown(f"## {emoji} {category}")
            st.markdown('<div class="score-label">Quality-risk score</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="score">{score:.3f}</div>', unsafe_allow_html=True)
            st.progress(min(max(score, 0), 1))
            st.markdown(f"**{recommendation}**")
            st.markdown("</div>", unsafe_allow_html=True)

        with top2:
            st.markdown("### What does this mean?")
            st.markdown(
                f"""
                <div class="explain">
                {explanation}<br><br>
                <b>Simple explanation:</b><br>
                The model is giving a signal about how strongly this
                image resembles the pattern it was trained to flag.
                It is <b>not</b> saying that the patient has a disease,
                and it does not establish whether the image is clinically
                acceptable.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if score >= THRESHOLD:
                st.warning(
                    f"The score is at or above the project's review threshold ({THRESHOLD:.2f})."
                )
            else:
                st.info(
                    f"The score is below the project's review threshold ({THRESHOLD:.2f})."
                )

        st.write("")
        st.markdown("### 👁️ What did the model focus on?")

        st.caption(
            "Grad-CAM is a visualization aid that highlights image regions "
            "that contributed to the model prediction. It is not a diagnostic heatmap."
        )

        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.image(
                np.clip(result["image"], 0, 255).astype(np.uint8),
                caption="Original image",
                use_container_width=True,
            )

        with c2:
            st.image(
                np.clip(result["gradcam"], 0, 255).astype(np.uint8),
                caption="Grad-CAM visualization",
                use_container_width=True,
            )

        if result["metadata"]:
            st.markdown("### 📋 DICOM information")
            st.caption(
                "These values come from the DICOM metadata when they are present in the uploaded file."
            )

            metadata_df = pd.DataFrame(
                list(result["metadata"].items()),
                columns=["Parameter", "Value"],
            )

            st.dataframe(
                metadata_df,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Why is 0.25 the threshold?"):
            st.write(
                """
                A threshold is simply a decision point. In this project,
                a score of 0.25 or higher produces a review flag.

                This threshold was selected for this research prototype.
                It is not a universal clinical cutoff and should not be
                used to make patient-care decisions.
                """
            )

        st.warning(
            """
            **Clinical safety:** This prototype is not clinically validated.
            Do not use its output to diagnose disease, accept/reject a clinical
            scan, or change CT radiation-dose protocols.
            """
        )

# ============================================================
# LEARN
# ============================================================

elif st.session_state.page == "Learn":

    st.markdown('<div class="eyebrow">PROJECT GUIDE</div>', unsafe_allow_html=True)
    st.title("Understand the Technology")

    st.write(
        "Technical terms are explained in simple language so that "
        "students and non-AI users can understand what the application is doing."
    )

    topics = [
        (
            "🧠 VGG16",
            """
            VGG16 is a deep-learning model designed to learn visual
            patterns in images.

            This project uses transfer learning: instead of training
            a visual model entirely from zero, previously learned image
            features are adapted to the CT image-quality task.
            """
        ),
        (
            "🔄 Test-Time Augmentation (TTA)",
            """
            TTA means showing the model several slightly modified
            versions of the same image before producing the final score.

            Here, five views are averaged: the original image, a flipped
            version, two small rotations, and a small crop/zoom.

            The simple idea is to reduce sensitivity to small changes
            in how the image is presented.
            """
        ),
        (
            "👁️ Grad-CAM",
            """
            Grad-CAM is a visualization method used to inspect which
            regions of an image contributed to a model prediction.

            It produces a heatmap. Think of it as asking:
            "Where was the model looking?"

            It does not prove that the highlighted region is abnormal
            or clinically important.
            """
        ),
        (
            "🩻 DICOM",
            """
            DICOM is a standard format used for medical imaging.

            A DICOM file can contain the image plus information about
            how it was acquired, such as tube voltage, tube current,
            exposure and slice thickness.
            """
        ),
        (
            "📈 Quality-risk score",
            """
            The model produces a numerical score rather than a simple
            yes/no answer.

            A higher score means the image more strongly resembles the
            pattern the model was trained to flag.

            This score is a model output, not a clinical measurement.
            """
        ),
        (
            "🎯 Threshold",
            """
            A threshold is a decision point.

            In this project, 0.25 is the review threshold:

            Below 0.25 → no automatic review flag.
            0.25 or above → review signal.

            This is specific to this project and is not a universal
            clinical rule.
            """
        ),
        (
            "📊 Recall and precision",
            """
            Recall asks: "Of all cases that really belonged to the
            flagged group, how many did the model find?"

            Precision asks: "Of all cases the model flagged, how many
            actually belonged to the flagged group?"

            The reported project results are:
            Recall 85%, Precision 54%, F1-score 0.66,
            ROC-AUC 0.839, and full-dose false-positive rate 3.3%.

            These metrics describe the project's evaluation and do not
            establish clinical effectiveness.
            """
        ),
    ]

    for title, explanation in topics:
        with st.expander(title):
            st.write(explanation)

    st.divider()

    st.markdown("### 📚 Project context")

    st.write(
        """
        The model was developed as a research prototype using paired
        full-dose and low-dose chest CT data. Quality labels were based
        on a noise-related image-quality proxy rather than
        radiologist-confirmed diagnostic ground truth.

        The reported evaluation used patient-level held-out testing.
        External validation across different patients, scanners and
        institutions would be needed before any clinical application.
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
    "Medical Imaging Technology • Educational/research prototype only"
)
