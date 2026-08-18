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


def go_to(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------------- GLOBAL ---------------- */

    .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    /* ---------------- TOP APP BAR ---------------- */

    .app-title {
        font-size: 1.25rem;
        font-weight: 750;
        color: #0f2540;
        line-height: 1.2;
        margin: 0;
    }

    .app-subtitle {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.15rem;
    }

    /* ---------------- HERO ---------------- */

    .hero-box {
        background: linear-gradient(
            135deg,
            #0b2138 0%,
            #174d73 55%,
            #2b82a8 100%
        );

        border-radius: 24px;
        padding: 3rem 3rem;
        margin-top: 0.8rem;
        margin-bottom: 1.5rem;

        color: white;

        box-shadow:
            0 12px 30px rgba(15, 37, 64, 0.15);
    }

    .hero-tag {
        font-size: 0.78rem;
        font-weight: 650;
        letter-spacing: 0.08em;
        opacity: 0.85;
        margin-bottom: 0.9rem;
    }

    .hero-title {
        font-size: clamp(2rem, 5vw, 3.2rem);
        font-weight: 800;
        line-height: 1.08;
        margin-bottom: 0.8rem;
    }

    .hero-description {
        font-size: clamp(0.95rem, 2vw, 1.1rem);
        line-height: 1.65;
        max-width: 760px;
        opacity: 0.93;
    }

    /* ---------------- CARDS ---------------- */

    .info-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.35rem;
        height: 100%;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    .info-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f2540;
        margin-bottom: 0.5rem;
    }

    .info-card-text {
        font-size: 0.92rem;
        line-height: 1.6;
        color: #475569;
    }

    /* ---------------- SIMPLE EXPLANATION ---------------- */

    .simple-box {
        background: #f1f8fb;
        border-left: 4px solid #2a7ba8;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        line-height: 1.65;
        color: #334155;
    }

    .warning-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        line-height: 1.6;
        color: #7c2d12;
    }

    .danger-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        line-height: 1.6;
        color: #7f1d1d;
    }

    /* ---------------- RESULT ---------------- */

    .score-number {
        font-size: clamp(2.3rem, 6vw, 3.2rem);
        font-weight: 800;
        color: #0f2540;
        line-height: 1;
    }

    .small-label {
        color: #64748b;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }

    /* ---------------- MOBILE ---------------- */

    @media (max-width: 700px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 0.8rem;
        }

        .hero-box {
            padding: 2rem 1.35rem;
            border-radius: 18px;
        }

        .hero-title {
            font-size: 2rem;
        }

        .hero-description {
            font-size: 0.92rem;
        }

        .hero-tag {
            font-size: 0.68rem;
        }

        .app-title {
            font-size: 1rem;
        }

        .app-subtitle {
            font-size: 0.65rem;
        }

    }

    /* Remove Streamlit default footer/menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TOP NAVIGATION
# ============================================================

# Brand row
brand_col, space_col = st.columns(
    [3, 1]
)

with brand_col:

    st.markdown(
        """
        <div class="app-title">
            🩻 CT Image Quality Flagger
        </div>

        <div class="app-subtitle">
            AI-assisted CT image-quality research prototype
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# Navigation buttons
nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:

    if st.button(
        "🏠 Home",
        use_container_width=True,
        type="secondary"
    ):
        go_to("Home")

with nav2:

    if st.button(
        "🔬 Analyze",
        use_container_width=True,
        type="secondary"
    ):
        go_to("Analyze")

with nav3:

    if st.button(
        "📋 Report",
        use_container_width=True,
        type="secondary"
    ):
        go_to("Report")

with nav4:

    if st.button(
        "🧠 Learn",
        use_container_width=True,
        type="secondary"
    ):
        go_to("Learn")


st.divider()


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILENAME
    )

    return tf.keras.models.load_model(
        model_path
    )


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
# DICOM PROCESSING
# ============================================================

def dicom_to_image(dicom_bytes):

    ds = pydicom.dcmread(
        io.BytesIO(dicom_bytes)
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

    hu = (
        pixels * slope
        + intercept
    )

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
        /
        (upper - lower)
        * 255
    )

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

    fields = [
        ("KVP", "Tube voltage"),
        ("XRayTubeCurrent", "Tube current"),
        ("Exposure", "Exposure"),
        ("SliceThickness", "Slice thickness"),
        ("BodyPartExamined", "Body part"),
        ("Manufacturer", "Scanner manufacturer"),
        ("ManufacturerModelName", "Scanner model"),
        ("SeriesDescription", "Series description")
    ]

    for tag, label in fields:

        if hasattr(
            ds,
            tag
        ):

            metadata[label] = str(
                getattr(
                    ds,
                    tag
                )
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
# MODEL PREDICTION
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

        predictions.append(
            float(
                model.predict(
                    rotated,
                    verbose=0
                )[0][0]
            )
        )

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
        np.mean(
            predictions
        )
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

    return np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# ANALYSIS
# ============================================================

def analyze(image):

    image_array, model_input = (
        prepare_image(
            image
        )
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
# RESULT INTERPRETATION
# ============================================================

def interpret(score):

    if score >= 0.50:

        return (
            "Higher-risk pattern",
            "🔴",
            "Review recommended",
            "The model produced a relatively high quality-risk score."
        )

    elif score >= THRESHOLD:

        return (
            "Borderline risk",
            "🟠",
            "Review recommended",
            "The score has reached the project's review threshold."
        )

    elif score >= 0.15:

        return (
            "Lower-risk pattern",
            "🟡",
            "No automatic flag",
            "The score is below the project's review threshold, although some uncertainty remains."
        )

    else:

        return (
            "Low-risk pattern",
            "🟢",
            "No automatic flag",
            "The model produced a relatively low quality-risk score."
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

    existing = [
        r["name"]
        for r in st.session_state.results
    ]

    if name not in existing:

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
# HOME
# ============================================================

if st.session_state.page == "Home":

    # ========================================================
    # HERO
    # ========================================================

    st.markdown(
        '<div class="hero-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        "MEDICAL IMAGING TECHNOLOGY × ARTIFICIAL INTELLIGENCE"
    )

    st.markdown(
        '<div class="hero-title">CT Image Quality Flagger</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        An AI-assisted research prototype exploring
        CT image-quality assessment as a supportive layer
        in dose-optimization research.
        """
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # ========================================================
    # INTRO
    # ========================================================

    left, right = st.columns(
        [1.4, 1],
        gap="large"
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

            I built this prototype around a practical CT question:
            **how can we think about radiation dose and image
            quality together?**

            The goal is not to diagnose disease. Instead, the
            model explores whether certain image patterns may
            deserve an additional quality review.
            """
        )

        st.info(
            """
            **The idea in one sentence:**  
            use AI as a supportive signal for image-quality
            assessment — not as a replacement for professional
            judgment.
            """
        )

    with right:

        st.markdown(
            """
            <div class="info-card">

            <div class="info-card-title">
                What can you explore?
            </div>

            <div class="info-card-text">

            🩻 Analyze CT images<br><br>

            📊 Generate a quality-risk score<br><br>

            👁️ Visualize model attention with Grad-CAM<br><br>

            📋 Read selected DICOM information<br><br>

            📥 Create a simple analysis report

            </div>

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
        "How the application works"
    )

    c1, c2, c3 = st.columns(
        3,
        gap="medium"
    )

    with c1:

        st.markdown(
            """
            <div class="info-card">

            <div class="info-card-title">
                🧠 1. VGG16
            </div>

            <div class="info-card-text">

            A deep-learning image model that learns
            visual patterns from images.

            Here it is adapted using
            <b>transfer learning</b> for CT
            image-quality assessment.

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="info-card">

            <div class="info-card-title">
                🔄 2. Multiple Views
            </div>

            <div class="info-card-text">

            The model examines several slightly
            modified versions of the same image.

            Their predictions are combined to make
            the final score more stable.

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="info-card">

            <div class="info-card-title">
                👁️ 3. Grad-CAM
            </div>

            <div class="info-card-text">

            A heatmap shows which image regions
            contributed to the model's prediction.

            This makes the model's output easier
            to inspect.

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.warning(
        """
        **Research prototype:** This application has not been
        clinically validated. Its output should not be used to
        make patient-care decisions or modify CT protocols.
        """
    )


# ============================================================
# ANALYZE
# ============================================================

elif st.session_state.page == "Analyze":

    st.title(
        "🔬 Analyze a CT Image"
    )

    st.write(
        """
        Test the model using a demonstration image or upload
        your own CT image.
        """
    )

    # ========================================================
    # DEMO
    # ========================================================

    st.subheader(
        "🧪 Demonstration cases"
    )

    st.caption(
        "These images are included only to demonstrate how the application works."
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

    cols = st.columns(
        4
    )

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
                    Image.open(
                        path
                    ).convert(
                        "RGB"
                    )
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
                            "The model could not be loaded."
                        )

                    else:

                        with st.spinner(
                            "Analyzing..."
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

                        st.rerun()

            else:

                st.info(
                    f"{title} is not available yet."
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
        **DICOM (.dcm)** is the preferred format for medical
        imaging because it can contain both the CT image and
        acquisition information.

        **PNG / JPG / JPEG** can also be used for demonstration.
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
                            Image.open(
                                file
                            ).convert(
                                "RGB"
                            )
                        )

                        metadata = {}

                    image_array, gradcam, score = (
                        analyze(
                            image
                        )
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
    # RESULT
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
        ) = interpret(
            score
        )

        st.markdown("---")

        st.subheader(
            "📊 Your result"
        )

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
                max(score, 0),
                1
            )
        )

        st.write(
            f"**Project threshold: {THRESHOLD:.2f}**"
        )

        if score >= THRESHOLD:

            st.warning(
                "Review signal triggered. "
                "The model score has reached the project's threshold."
            )

        else:

            st.success(
                "No automatic review flag. "
                "The score is below the project's threshold."
            )

        st.markdown(
            "### 💡 What does this mean?"
        )

        st.info(
            explanation
        )

        st.write(
            """
            **Simple explanation:**  
            the model is giving us a signal about how strongly
            the image resembles the pattern it was trained to flag.

            It is **not** saying that the patient has a disease,
            and it does **not** replace professional image review.
            """
        )

        with st.expander(
            "Why is 0.25 important?"
        ):

            st.write(
                """
                A threshold is simply a decision point.

                In this project, a score of **0.25 or higher**
                produces a review flag.

                This value belongs to this research project.
                It is **not a universal clinical cutoff**.
                """
            )

        st.markdown(
            "### 👁️ What did the model focus on?"
        )

        st.write(
            """
            Grad-CAM creates a heatmap showing regions that
            contributed to the model's prediction.

            It is an explanation aid, not a diagnostic image.
            """
        )

        col1, col2 = st.columns(
            2,
            gap="large"
        )

        with col1:

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

        with col2:

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

        st.warning(
            """
            This is an educational and research prototype.
            It should not be used to accept/reject clinical scans,
            diagnose patients, or change CT acquisition protocols.
            """
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
            "No images have been analyzed yet."
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
            s >= THRESHOLD
            for s in scores
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
        Technical terms are explained here in simple language,
        so the project is understandable even if you are new
        to artificial intelligence.
        """
    )

    with st.expander(
        "🧠 What is VGG16?"
    ):

        st.write(
            """
            VGG16 is a deep-learning model designed to learn
            visual patterns from images.

            This project uses **transfer learning**, meaning
            previously learned visual features are reused and
            adapted for the CT image-quality task.

            **Simple idea:** instead of teaching an AI model
            everything from zero, we start with an existing
            visual model and adapt it to our problem.
            """
        )

    with st.expander(
        "🔄 What is Test-Time Augmentation?"
    ):

        st.write(
            """
            Test-Time Augmentation, or TTA, means asking the
            model to examine several slightly modified versions
            of the same image.

            This project uses five views:

            • Original image
            • Horizontal flip
            • Small negative rotation
            • Small positive rotation
            • Small crop/zoom

            The predictions are then averaged.

            **Simple idea:** give the model several slightly
            different looks at the same image before deciding
            on the final score.
            """
        )

    with st.expander(
        "👁️ What is Grad-CAM?"
    ):

        st.write(
            """
            Grad-CAM is a visualization technique that helps
            us understand which areas of an image contributed
            to the model's prediction.

            It produces a heatmap.

            **Simple idea:** it helps answer,
            "Where was the AI looking?"

            It does not prove that the highlighted area is
            abnormal or clinically important.
            """
        )

    with st.expander(
        "🩻 What is DICOM?"
    ):

        st.write(
            """
            DICOM is a standard format used for medical images.

            Unlike an ordinary JPG, a DICOM file can contain
            both the image and information about how the image
            was acquired.

            Examples include:

            • Tube voltage
            • Tube current
            • Exposure
            • Slice thickness
            • Body part
            • Scanner information
            """
        )

    with st.expander(
        "📈 What is the quality-risk score?"
    ):

        st.write(
            """
            The model produces a numerical score rather than
            simply saying "good" or "bad."

            A higher score means the image is more strongly
            associated with the pattern that the model was
            trained to flag.

            This project uses **0.25** as its review threshold.

            The score is a model output — it is not a clinical
            diagnostic measurement.
            """
        )

    with st.expander(
        "🎯 What is a threshold?"
    ):

        st.write(
            """
            A threshold is a decision point.

            In this project:

            **Below 0.25**
            → no automatic review flag

            **0.25 or above**
            → review flag

            This threshold is specific to this project and
            should not be treated as a universal clinical rule.
            """
        )

    with st.expander(
        "📊 What do recall and precision mean?"
    ):

        st.write(
            """
            These are ways of evaluating how well a model
            identifies the cases it was designed to flag.

            **Recall** asks:

            "Of all the cases that really belonged to the
            flagged group, how many did the model find?"

            **Precision** asks:

            "Of all the cases the model flagged, how many
            actually belonged to the flagged group?"

            For this project, the reported evaluation results
            were:

            • Recall: 85%
            • Precision: 54%
            • F1-score: 0.66
            • ROC-AUC: 0.839
            • Full-dose false-positive rate: 3.3%

            These results describe the project's evaluation
            dataset and do not establish clinical effectiveness.
            """
        )

    st.markdown("---")

    st.subheader(
        "⚠️ Important limitations"
    )

    st.write(
        """
        This is a research prototype developed using a limited
        public dataset.

        The model's quality labels were based on
        noise-related measurements rather than
        radiologist-confirmed diagnostic quality.

        Testing on different patients, scanners and institutions,
        together with clinically meaningful reference standards,
        would be needed before considering real clinical use.
        """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CT Image Quality Flagger • "
    "Zainab Fatima • Medical Imaging Technology • "
    "Educational & research prototype — not for clinical use."
)
