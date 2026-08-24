# ============================================================
# PROFESSIONAL HEADER
# ============================================================

st.markdown(
    """
    <style>

    .app-header {
        background: linear-gradient(
            135deg,
            #102a43 0%,
            #174e73 55%,
            #247ba0 100%
        );
        border-radius: 22px;
        padding: 2.6rem 2.8rem;
        margin: 0.5rem 0 1.5rem 0;
        box-shadow: 0 12px 30px rgba(16,42,67,0.12);
    }

    .app-kicker {
        color: rgba(255,255,255,0.78);
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        margin-bottom: 0.7rem;
    }

    .app-title {
        color: #ffffff;
        font-size: 3rem;
        line-height: 1.05;
        font-weight: 800;
        margin: 0;
    }

    .app-description {
        color: rgba(255,255,255,0.90);
        font-size: 1rem;
        line-height: 1.6;
        max-width: 720px;
        margin-top: 0.85rem;
    }

    .intro-card {
        background: #ffffff;
        border: 1px solid #e1e8ef;
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
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
        font-size: 0.9rem;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.markdown(
    """
    <div class="app-header">
        <div class="app-kicker">
            MEDICAL IMAGING TECHNOLOGY × ARTIFICIAL INTELLIGENCE
        </div>

        <div class="app-title">
            CT Image Quality Flagger
        </div>

        <div class="app-description">
            An AI-assisted research prototype that flags CT images
            showing patterns associated with reduced image quality,
            helping the technologist identify images that may deserve
            a closer review.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# INTRO
# ------------------------------------------------------------

st.markdown(
    """
    <div class="intro-card">

        <div class="intro-title">
            Hi, I'm Zainab 👋
        </div>

        <div class="intro-text">
            I'm a Medical Imaging Technology student interested in how
            AI can support safer and more consistent medical-imaging
            workflows. I built this prototype around one practical
            question: can AI help flag CT images that may deserve
            a closer quality review?
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)
