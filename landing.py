import streamlit as st

def show_landing_page():
    st.markdown("""
    <style>
    .hero {
        text-align: center;
        padding: 80px 20px;
        background: linear-gradient(135deg, #0E0E11, #1A1A1E);
        border-radius: 20px;
        box-shadow: 0 4px 25px rgba(0,0,0,0.4);
        margin-bottom: 40px;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        color: #FF4B4B;
        margin-bottom: 10px;
    }
    .hero p {
        font-size: 1.2rem;
        color: #DDD;
        max-width: 700px;
        margin: 0 auto;
    }
    .feature-box {
        background: #1A1A1E;
        padding: 25px;
        border-radius: 12px;
        border-left: 4px solid #FF4B4B;
        margin-bottom: 20px;
    }
    .feature-box h3 {
        margin: 0;
        color: #FFF;
    }
    .feature-box p {
        color: #AAA;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <h1>SignStudio</h1>
        <p>The fastest, cleanest, and most customizable ASL + Sign Language translator ever built in Streamlit.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("✨ What You Can Do Here")

    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown("""
        <div class="feature-box">
            <h3>🔮 Live Translator</h3>
            <p>Translate signs instantly using your webcam with real-time confidence scoring.</p>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("""
        <div class="feature-box">
            <h3>📘 ASL Dictionary</h3>
            <p>Browse, preview, and manage pre-built ASL blueprint words.</p>
        </div>
        """, unsafe_allow_html=True)

    with colC:
        st.markdown("""
        <div class="feature-box">
            <h3>🛠️ Creator Studio</h3>
            <p>Create your own custom sign-language words using 60-frame angle averaging.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 🚀 Choose a workspace to begin:")
    mode = st.radio("", ["Landing Page", "Translator", "ASL Dictionary", "Creator Studio"])

    return mode
