import streamlit as st

# =========================
# PAGE CONFIG (HARUS PALING ATAS)
# =========================
st.set_page_config(
    page_title="Image Emotion Classification Dashboard",
    page_icon="assets/logo.png",
    layout="centered"
)

# =========================
# PIXEL FONT + GRADIENT + BUTTON CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
* {
    font-family: 'Press Start 2P', monospace !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}
.stApp {
    background-image: linear-gradient(to bottom, #0D0D17, #181829);
    min-height: 100vh;
}

/* -------------------------
   TEXT ELEMENTS
------------------------- */
p, span, label, div, small,
h1, h2, h3, h4, h5, h6 {
    color: #A1A1FF;
}

/* -------------------------
   BUTTONS - like file_uploader
------------------------- */
.stButton > button {
    width: 195px !important;
    height: 60px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background-color: #131d34 !important; /* default tombol */
    border: none !important;               /* border hilang normal */
    border-radius: 8px !important;
    padding: 14px 0px !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 14px !important;
    color: #A1A1FF !important;            /* teks default */
    transition: all 0.15s ease-in-out !important;
}
/* Hover */
.stButton > button:hover,
.stButton > button:hover span {
    border: 2px solid #A1A1FF !important;  /* border muncul saat hover */
    background-color: #181829 !important;  /* background tetap gelap */
    color: #A1A1FF !important;             /* teks tetap ungu */
}
/* Focus */
.stButton > button:focus,
.stButton > button:focus span {
    border: 2px solid #A1A1FF !important;  /* border muncul saat fokus */
    outline: none !important;
}

/* -------------------------
   INPUTS
------------------------- */
input, textarea, select {
    background-color: rgba(24,24,41,0.8) !important;
    color: #A1A1FF !important;
    border: 1px solid #A1A1FF !important;
}

/* -------------------------
   CENTER COLUMNS EVENLY
------------------------- */
[data-testid="stVerticalBlock"] > div[role="list"] {
    justify-content: center !important;
    gap: 40px !important;
}
</style>
""", unsafe_allow_html=True)

# Preload semua model
from modules.base_cnn_app import load_model
from modules.effnet_app import load_model
from modules.vgg16_app import load_model

# Loading model saat aplikasi dijalankan
with st.spinner("Loading all models, please wait..."):
    base_cnn_model = load_model()
    vgg16_model = load_model()
    effnetb0_model = load_model()
    
# =========================
# ROUTER
# =========================
params = st.query_params
page = params.get("page", "home")

def go(p=None):
    if p is None:
        st.query_params.clear()
    else:
        st.query_params.update({"page": p})

# =========================
# HOME
# =========================
if page == "home":
    st.title("Emotion Classification (FER2013)")

    st.markdown("""
    Pilih model **Image Classification**:
    - CNN Base (From Scratch)
    - VGG16 (Transfer Learning)
    - EfficientNetB0 (Transfer Learning)
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("CNN Base", on_click=lambda: go("cnn"))
    with col2:
        st.button("VGG16", on_click=lambda: go("vgg"))
    with col3:
        st.button("EfficientNet", on_click=lambda: go("effnet"))

# =========================
# CNN PAGE
# =========================
elif page == "cnn":
    from modules.base_cnn_app import render
    st.button("⬅️ Home", on_click=lambda: go())
    render()

# =========================
# VGG PAGE
# =========================
elif page == "vgg":
    from modules.vgg16_app import render
    st.button("⬅️ Home", on_click=lambda: go())
    render()

# =========================
# EFFNET PAGE
# =========================
elif page == "effnet":
    from modules.effnet_app import render
    st.button("⬅️ Home", on_click=lambda: go())
    render()
