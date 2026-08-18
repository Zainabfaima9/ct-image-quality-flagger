import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import pydicom
import cv2
import pandas as pd
import io
import os
from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CT Image Quality Flagger",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SETTINGS
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


# ============================================================
# NAVIGATION
# ============================================================

def go_to(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# CLEAN APP CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 3.2rem;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            #0b2138,
            #174d73,
            #2b82a8
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero-small {
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        opacity: 0.85;
        margin-bottom: 0.8rem;
    }

    .hero h1 {
        font-size: 3rem;
        margin: 0;
        line-height: 1.1;
    }

    .hero p {
        font-size: 1.08rem;
        max-width: 760px;
        line-height: 1.7;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.3rem;
        height: 100%;
    }

    .card h3 {
        color: #0f2540;
        margin-top: 0;
    }

    .card p {
        color: #475569;
        line-height: 1.6;
    }

    .simple-box {
        background: #f1f8fb;
        border-left: 4px solid #2a7ba8;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        line-height: 1.65;
    }

    .warning-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        line-height: 1.6;
    }

    .danger-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        line-height: 1.6;
    }

    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .score-number {
        font-size: 3rem;
        font-weight: 750;
        color: #0f2540;
    }

    .small-label {
        color: #64748b;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0f2540;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #64748b;
    }

    footer {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TOP NAVIGATION
# ============================================================

nav1, nav2, nav3, nav4, nav5 = st.columns(
    [2.5, 1.2, 1.3, 1.4, 1.5]
)

with nav1:
    st.markdown("### 🩻 CT Image Quality Flagger")

with nav2:
    if st.button("Home", use_container_width=True):
        go_to("Home")

with nav3:
    if st.button("Analyze", use_container_width=True):
        go_to("Analyze")

with nav4:
    if st.button("Report", use_container_width=True):
        go_to("Report")

with nav5:
    if st.button("Learn", use_container_width=True):
        go_to("Learn")


st.markdown("---")


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILENAME
    )

    return tf.keras.models.load_model(model_path)


@st.cache_resource
def build_gradcam_extractor(model):

    base_model = model.layers[0]

    extractor = tf.keras.Model(
        base_model.input,
        base_model.get_layer(
            GRADCAM_LAYER
        ).output
    )

    return extractor, base_model


try:

    model = load_model()

    conv_layer_model, base_model = (
        build_gradcam_extractor(model)
    )

    MODEL_READY = True

except Exception as e:

    MODEL_READY = False
    MODEL_ERROR = str(e)


# ============================================================
# DICOM
# ============================================================

def dicom_to_image(dicom_bytes):

    ds = pydicom.dcmread(
        io.BytesIO(dicom_bytes)
    )

    pixels = ds.pixel_array.astype(
        np.float32
    )

    slope = float(
        getattr(ds, "RescaleSlope", 1)
    )

    intercept = float(
        getattr(ds, "RescaleIntercept", 0)
    )

    hu = pixels * slope + intercept

    lower = (
        WINDOW_CENTER
        - WINDOW_WIDTH / 2
    )

    upper = (
        WINDOW_CENTER
        + WINDOW_WIDTH / 2
    )

    hu = np.clip(
        hu,
        lower,
        upper
    )

    hu = (
        (hu - lower)
        / (upper - lower)
        * 255
    )

    # VERY IMPORTANT:
    # Always make displayed image uint8.
    hu = np.clip(
        hu,
        0,
        255
    ).astype(
        np.uint8
    )

    image = Image.fromarray(
        hu
    ).convert(
        "RGB"
    )

    image = image.resize(
        IMAGE_SIZE
    )

    metadata = {}

    dicom_fields = [
        ("KVP", "Tube voltage"),
        ("XRayTubeCurrent", "Tube current"),
        ("Exposure", "Exposure"),
        ("SliceThickness", "Slice thickness"),
        ("BodyPartExamined", "Body part"),
        ("Manufacturer", "Scanner manufacturer"),
        ("ManufacturerModelName", "Scanner model"),
        ("SeriesDescription", "Series description")
    ]

    for tag, label in dicom_fields:

        if hasattr(ds, tag):

            value = getattr(
                ds,
                tag
            )

            metadata[label] = str(
                value
            )

    return image, metadata


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image):

    image = image.convert(
        "RGB"
    ).resize(
        IMAGE_SIZE
    )

    # IMPORTANT:
    # uint8 image for display
    image_array = np.array(
        image
    ).astype(
        np.uint8
    )

    model_input = (
        image_array
        .astype(np.float32)
    )

    model_input = np.expand_dims(
        model_input,
        axis=0
    )

    model_input = preprocess_input(
        model_input
    )

    return (
        image_array,
        model_input
    )


# ============================================================
# MODEL SCORE + TTA
# ============================================================

def get_score(model_input):

    predictions = []

    # Original
    pred = model.predict(
        model_input,
        verbose=0
    )[0][0]

    predictions.append(
        float(pred)
    )

    # Horizontal flip
    flipped = np.flip(
        model_input,
        axis=2
    )

    pred = model.predict(
        flipped,
        verbose=0
    )[0][0]

    predictions.append(
        float(pred)
    )

    # Rotations
    image = model_input[0]

    center = (
        112,
        112
    )

    for angle in [-5, 5]:

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

        pred = model.predict(
            rotated,
            verbose=0
        )[0][0]

        predictions.append(
            float(pred)
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
        axis=0
    )

    pred = model.predict(
        zoomed,
        verbose=0
    )[0][0]

    predictions.append(
        float(pred)
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

    gradients = tape.gradient(
        loss,
        conv_output
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_output = conv_output[0]

    heatmap = (
        conv_output
        @ pooled_gradients[
            ..., tf.newaxis
        ]
    )

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = (
        heatmap
        /
        (
            tf.reduce_max(
                heatmap
            )
            + 1e-8
        )
    )

    heatmap = cv2.resize(
        heatmap.numpy(),
        IMAGE_SIZE
    )

    heatmap = np.clip(
        heatmap,
        0,
        1
    )

    heatmap_color = cv2.applyColorMap(
        (
            heatmap * 255
        ).astype(
            np.uint8
        ),
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Make absolutely sure both arrays are uint8
    image_array = np.clip(
        image_array,
        0,
        255
    ).astype(
        np.uint8
    )

    heatmap_color = np.clip(
        heatmap_color,
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

    # Final safety conversion
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

    image_array, model_input = (
        prepare_image(image)
    )

    score = get_score(
        model_input
    )

    gradcam = make_gradcam(
        image_array,
        model_input
    )

    return (
        image_array,
        gradcam,
        score
    )


# ============================================================
# INTERPRETATION
# ============================================================

def interpret(score):

    if score >= 0.50:

        return (
            "Higher-risk pattern",
            "🔴",
            "Review recommended",
            """
            The model produced a relatively high
            quality-risk score.
            """
        )

    elif score >= THRESHOLD:

        return (
            "Borderline risk",
            "🟠",
            "Review recommended",
            """
            The score is at or above the project's
            review threshold.
            """
        )

    elif score >= 0.15:

        return (
            "Lower-risk pattern",
            "🟡",
            "No automatic flag",
            """
            The score is below the project's review
            threshold, although some uncertainty remains.
            """
        )

    else:

        return (
            "Low-risk pattern",
            "🟢",
            "No automatic flag",
            """
            The model produced a relatively low
            quality-risk score.
            """
        )


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    name,
    image,
    gradcam,
    score,
    metadata
):

    # Avoid duplicates
    names = [
        r["name"]
        for r in st.session_state.results
    ]

    if name not in names:

        st.session_state.results.append(
            {
                "name": name,
                "image": image,
                "gradcam": gradcam,
                "score": score,
                "metadata": metadata
            }
        )


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    # HERO

    st.markdown(
        """
        <div class="hero">

            <div class="hero-small">
                MEDICAL IMAGING TECHNOLOGY × ARTIFICIAL INTELLIGENCE
            </div>

            <h1>
                CT Image Quality Flagger
            </h1>

            <p>
                An AI-assisted research prototype exploring
                how CT image-quality assessment can support
                dose-optimization research.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # INTRO

    left, right = st.columns(
        [1.4, 1]
    )

    with left:

        st.header(
            "Hi, I'm Zainab 👋"
        )

        st.write(
            """
            I'm a Medical Imaging Technology student interested
            in how artificial intelligence can become a useful
            part of medical-imaging workflows.

            I built this prototype around a practical CT problem:

            **How can we reduce radiation exposure while still
            paying attention to image quality?**

            Instead of trying to diagnose disease, this project
            explores whether AI can identify image patterns that
            may deserve additional quality review.
            """
        )

        st.markdown(
            """
            <div class="simple-box">

            <b>The idea is simple:</b><br><br>

            CT dose should be optimized carefully. Too much
            radiation increases exposure, while excessive dose
            reduction can affect image quality.

            This prototype explores AI as a
            <b>supportive tool</b> — not as a replacement for
            radiologists or Medical Imaging Technologists.

            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="card">

            <h3>What can this app do?</h3>

            <p>
            🩻 Analyze a CT image
            </p>

            <p>
            🧠 Generate a quality-risk score
            </p>

            <p>
            👁️ Show a Grad-CAM explanation
            </p>

            <p>
            📋 Read selected DICOM information
            </p>

            <p>
            📊 Generate a simple analysis report
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    if st.button(
        "🔬 Start CT Analysis",
        type="primary",
        use_container_width=True
    ):

        go_to("Analyze")

    st.markdown("---")

    st.subheader(
        "Before you start"
    )

    st.write(
        """
        You can test the application using the built-in
        demonstration cases, or upload your own CT image.
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="card">

            <h3>🧠 VGG16</h3>

            <p>
            A deep-learning image model used here to
            recognize visual patterns associated with
            image-quality risk.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="card">

            <h3>👁️ Grad-CAM</h3>

            <p>
            Creates a heatmap to show which parts of the
            image contributed to the model's prediction.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="card">

            <h3>📊 Test-Time Augmentation</h3>

            <p>
            The model looks at several slightly modified
            versions of the same image and combines the
            predictions.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(
        """
        <div class="danger-box">

        <b>Research & educational prototype</b><br><br>

        This application is not clinically validated and
        should not be used to make patient-care decisions,
        repeat examinations, change CT protocols, or replace
        professional judgment.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ANALYZE PAGE
# ============================================================

elif st.session_state.page == "Analyze":

    st.title(
        "🔬 Analyze a CT Image"
    )

    st.write(
        """
        Choose a demonstration case or upload a CT image.
        The app will generate a quality-risk score and a
        visual explanation.
        """
    )

    # ========================================================
    # DEMO CASES
    # ========================================================

    st.subheader(
        "🧪 Try a demonstration case"
    )

    st.caption(
        "Demo images are included so you can test the app "
        "without uploading your own file."
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
        )
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
                    use_container_width=True
                )

                if st.button(
                    "Analyze",
                    key=f"demo_{i}",
                    use_container_width=True
                ):

                    if not MODEL_READY:

                        st.error(
                            "The AI model could not be loaded."
                        )

                    else:

                        with st.spinner(
                            "Analyzing image..."
                        ):

                            image, gradcam, score = (
                                analyze(
                                    demo_image
                                )
                            )

                        save_result(
                            title,
                            image,
                            gradcam,
                            score,
                            {}
                        )

                        st.success(
                            "Analysis complete."
                        )

                        st.rerun()

            else:

                st.warning(
                    f"{title} image is missing."
                )

    st.markdown("---")

    # ========================================================
    # UPLOAD
    # ========================================================

    st.subheader(
        "📤 Upload your CT image"
    )

    st.write(
        """
        **DICOM (.dcm):** CT's standard medical-imaging
        format. It can contain both image information and
        technical acquisition information.

        **PNG / JPG / JPEG:** standard image formats.
        """
    )

    uploaded = st.file_uploader(
        "Choose CT image(s)",
        type=[
            "dcm",
            "png",
            "jpg",
            "jpeg"
        ],
        accept_multiple_files=True
    )

    if uploaded:

        for file in uploaded:

            if any(
                r["name"] == file.name
                for r in st.session_state.results
            ):
                continue

            try:

                if not MODEL_READY:

                    st.error(
                        "The AI model could not be loaded."
                    )

                    break

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

                    image_array, gradcam, score = (
                        analyze(image)
                    )

                save_result(
                    file.name,
                    image_array,
                    gradcam,
                    score,
                    metadata
                )

                st.success(
                    f"{file.name} analyzed successfully."
                )

            except Exception as error:

                st.error(
                    f"Could not analyze {file.name}."
                )

                st.caption(
                    str(error)
                )

    # ========================================================
    # MOST RECENT RESULT
    # ========================================================

    if st.session_state.results:

        result = (
            st.session_state.results[-1]
        )

        score = result["score"]

        (
            category,
            emoji,
            recommendation,
            explanation
        ) = interpret(score)

        st.markdown("---")

        st.subheader(
            "📊 Analysis Result"
        )

        # Result using Streamlit components,
        # NOT raw HTML.

        st.markdown(
            f"## {emoji} {category}"
        )

        st.markdown(
            '<div class="small-label">Quality-risk score</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="score-number">{score:.3f}</div>',
            unsafe_allow_html=True
        )

        st.progress(
            min(
                max(score, 0.0),
                1.0
            )
        )

        st.write(
            f"**Project threshold: {THRESHOLD:.2f}**"
        )

        if score >= THRESHOLD:

            st.warning(
                "Review recommended: the model score "
                "has reached the project's review threshold."
            )

        else:

            st.success(
                "No automatic review flag: the score is "
                "below the project's threshold."
            )

        # Simple explanation

        st.markdown(
            "### 💡 What does this mean?"
        )

        st.info(
            explanation
        )

        st.markdown(
            """
            **In simple words:** the score tells us how
            strongly this image resembles the pattern that
            the model was trained to flag.

            It does **not** mean that the patient has a disease,
            and it does **not** determine whether the scan is
            clinically diagnostic.
            """
        )

        # Threshold explanation

        with st.expander(
            "What is the 0.25 threshold?"
        ):

            st.write(
                f"""
                The model produces a numerical score.

                In this project, **0.25** was selected as
                the review threshold.

                A score of **0.25 or above** produces a
                review flag.

                A threshold is simply a decision point used
                to convert a continuous model score into
                a practical signal.

                **Important:** 0.25 is a project-specific
                threshold. It is **not a universal clinical
                cutoff**.
                """
            )

        # Images

        st.markdown(
            "### 👁️ Model explanation"
        )

        st.write(
            """
            The image below shows the original CT beside
            a Grad-CAM heatmap. The heatmap is an explanation
            aid showing regions that contributed to the model's
            prediction.
            """
        )

        img1, img2 = st.columns(2)

        with img1:

            # Explicit uint8 safety
            original = np.clip(
                result["image"],
                0,
                255
            ).astype(
                np.uint8
            )

            st.image(
                original,
                caption="Original CT image",
                use_container_width=True
            )

        with img2:

            # Explicit uint8 safety
            gradcam = np.clip(
                result["gradcam"],
                0,
                255
            ).astype(
                np.uint8
            )

            st.image(
                gradcam,
                caption="Grad-CAM visualization",
                use_container_width=True
            )

        with st.expander(
            "What is Grad-CAM?"
        ):

            st.write(
                """
                Grad-CAM stands for Gradient-weighted
                Class Activation Mapping.

                It produces a heatmap showing areas that
                contributed to the model's prediction.

                In simple terms:

                **It helps us see where the model was looking.**

                However, a highlighted area should not be
                interpreted as a diagnosis or proof of an
                abnormality.
                """
            )

        # DICOM

        if result["metadata"]:

            with st.expander(
                "📋 DICOM information"
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

        st.markdown(
            """
            <div class="warning-box">

            ⚠️ <b>Important:</b>

            This is an experimental research prototype.
            The result should not be used to accept or reject
            a clinical CT examination or to modify patient
            imaging protocols.

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# REPORT
# ============================================================

elif st.session_state.page == "Report":

    st.title(
        "📋 Analysis Report"
    )

    if not st.session_state.results:

        st.info(
            "No CT images have been analyzed yet."
        )

        if st.button(
            "🔬 Analyze a CT"
        ):

            go_to("Analyze")

    else:

        results = (
            st.session_state.results
        )

        scores = [
            r["score"]
            for r in results
        ]

        flagged = sum(
            score >= THRESHOLD
            for score in scores
        )

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "Images analyzed",
                len(results)
            )

        with b:
            st.metric(
                "Review flags",
                flagged
            )

        with c:
            st.metric(
                "Average score",
                f"{np.mean(scores):.3f}"
            )

        rows = []

        for r in results:

            category, _, recommendation, _ = (
                interpret(
                    r["score"]
                )
            )

            rows.append(
                {
                    "File": r["name"],
                    "Risk score": round(
                        r["score"],
                        3
                    ),
                    "Assessment": category,
                    "Recommendation": recommendation
                }
            )

        df = pd.DataFrame(
            rows
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv = df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "📥 Download CSV report",
            csv,
            "ct_quality_report.csv",
            "text/csv"
        )

        if st.button(
            "Clear report"
        ):

            st.session_state.results = []

            st.rerun()


# ============================================================
# LEARN
# ============================================================

elif st.session_state.page == "Learn":

    st.title(
        "🧠 Learn About the Project"
    )

    st.write(
        """
        Technical terms are explained here in simple language
        so that the project can be understood by both medical
        imaging students and people from a non-AI background.
        """
    )

    with st.expander(
        "🧠 What is VGG16?"
    ):

        st.write(
            """
            VGG16 is a deep-learning model that can learn
            visual patterns from images.

            This project uses VGG16 through **transfer learning**.

            Transfer learning means taking a model that has
            already learned useful visual features and adapting
            it to a different task.

            **Simple idea:** instead of teaching the model
            everything from zero, we reuse useful visual
            knowledge and adapt it for CT image-quality
            assessment.
            """
        )

    with st.expander(
        "🔄 What is Test-Time Augmentation (TTA)?"
    ):

        st.write(
            """
            Test-Time Augmentation means giving the model
            several slightly modified versions of the same
            image and combining the predictions.

            In this project, five versions are used:

            • Original image
            • Horizontal flip
            • Small negative rotation
            • Small positive rotation
            • Small crop/zoom

            **Simple idea:** the model gets several slightly
            different looks at the same image before producing
            the final score.
            """
        )

    with st.expander(
        "👁️ What is Grad-CAM?"
    ):

        st.write(
            """
            Grad-CAM is a visualization technique used to
            understand which parts of an image contributed
            to a model's prediction.

            It creates a heatmap.

            **Simple idea:** it gives us a visual clue about
            where the AI was looking.

            It does not prove that the highlighted area is
            clinically abnormal.
            """
        )

    with st.expander(
        "🩻 What is DICOM?"
    ):

        st.write(
            """
            DICOM is the standard format commonly used for
            medical images.

            Unlike an ordinary JPG, a DICOM file can contain
            both the image and information about how the image
            was acquired.

            For example, it may contain:

            • Tube voltage (kVp)
            • Tube current
            • Exposure
            • Slice thickness
            • Body part
            • Scanner information
            """
        )

    with st.expander(
        "📈 What is a quality-risk score?"
    ):

        st.write(
            """
            The model does not simply produce "good" or "bad."

            It produces a numerical score.

            A higher score means the image is more strongly
            associated with the pattern the model was trained
            to flag.

            This project uses 0.25 as its review threshold.

            **Important:** this is a research-model score,
            not a clinical diagnostic measurement.
            """
        )

    with st.expander(
        "🎯 What is a threshold?"
    ):

        st.write(
            """
            A threshold is a decision point.

            Here:

            **Score < 0.25**
            → no automatic review flag

            **Score ≥ 0.25**
            → review flag

            This threshold belongs to this project and should
            not be treated as a universal clinical standard.
            """
        )

    with st.expander(
        "🧪 How was the model evaluated?"
    ):

        st.write(
            """
            The final evaluation used three fully held-out
            patients containing 1,058 CT slices.

            "Held-out" means these patients were kept separate
            from model development.

            This is important because slices from the same
            patient can be highly similar.

            The reported evaluation results are:

            • Recall: 85%
            • Precision: 54%
            • F1-score: 0.66
            • ROC-AUC: 0.839
            • Full-dose false-positive rate: 3.3%

            These numbers describe this project's evaluation
            dataset. They do not establish clinical effectiveness.
            """
        )

    st.markdown("---")

    st.subheader(
        "⚠️ Project limitations"
    )

    st.write(
        """
        This is a research prototype.

        The model was developed using a limited number of
        patients from one public dataset. The quality labels
        were based on noise-related measurements rather than
        radiologist-confirmed diagnostic quality.

        External validation using different patients, scanners,
        institutions and clinically meaningful reference
        standards would be required before considering clinical
        deployment.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "CT Image Quality Flagger • "
    "Zainab Fatima • Medical Imaging Technology • "
    "Educational & research prototype — not for clinical use."
)
