from PIL import Image, ImageOps, ImageFilter
import os
import re
import sqlite3
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import cv2
import base64
import keras

# ==========================================
# 1. KONFIGURASI UTAMA & DATABASE
# ==========================================
st.set_page_config(page_title="NailSense App", page_icon="💅", layout="centered")

MODEL_PATH = r"C:\Users\WN11\OneDrive\Documents\NAILSENSE\nailsense_hibrida_softmaxfixx.keras"
DB_PATH    = "nailsense_data.db"

# =================================================================
# PREPROCESSING
# =================================================================
TARGET_SIZE = (224, 224)
NORM_MODE   = "simple"

MIN_CONFIDENCE        = 55.0   

MIN_IMAGE_SIZE       = 80     
BLUR_THRESHOLD       = 50     
OVEREXPOSE_THRESHOLD = 248    
UNDEREXPOSE_THRESHOLD = 15    

NAIL_EDGE_DENSITY_MIN  = 0.04  
NAIL_EDGE_DENSITY_MAX  = 0.40  
NAIL_COLOR_STD_MAX     = 75.0  
NAIL_SAT_MAX           = 65.0  
NAIL_MIN_R_OVER_B      = 1.02  

# Kuesioner Data Gejala Klinis
PERTANYAAN_GEJALA = [
    {
        "id": "lelah",
        "tanya": "Apakah akhir-akhir ini Anda sering mengalami kelelahan atau merasa lemas (5L) tanpa alasan yang jelas?",
    },
    {
        "id": "pusing",
        "tanya": "Apakah Anda sering merasa pusing, sakit kepala, atau terasa berputar terutama saat mendadak berdiri?",
    },
    {
        "id": "pucat",
        "tanya": "Apakah kelopak mata bagian dalam (konjungtiva), bibir, atau telapak tangan Anda terlihat lebih pucat dari biasanya?",
    },
    {
        "id": "jantung",
        "tanya": "Apakah Anda sering merasakan jantung berdebar-debar lebih cepat padahal sedang tidak melakukan aktivitas berat?",
    },
    {
        "id": "sesak",
        "tanya": "Apakah Anda merasa mudah terengah-engah atau sesak napas saat melakukan aktivitas ringan (misal: naik tangga)?",
    }
]

OPSI_JAWABAN = {
    "Sangat Sering": 3,
    "Sering": 2,
    "Kadang-kadang": 1,
    "Tidak Pernah": 0
}  

# ==========================================
# DATABASE
# ==========================================
def init_db():
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            nama TEXT,
            usia INTEGER,
            gender TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            tanggal TEXT,
            hasil TEXT,
            confidence REAL,
            catatan TEXT,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@st.cache_resource
def load_nailsense_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"File model tidak ditemukan di:\n{MODEL_PATH}"
    try:
        m        = keras.models.load_model(MODEL_PATH)
        n_inputs = len(m.inputs)
        if n_inputs not in (1, 2):
            return None, (
                f"Model memiliki {n_inputs} input. Hanya 1 atau 2 input yang didukung."
            )
        return m, None
    except Exception as e:
        return None, f"Gagal memuat model: {str(e)}"

model, model_error = load_nailsense_model()
MODEL_N_INPUTS     = len(model.inputs) if model is not None else 0

# ==========================================
# 2. CSS & TAMPILAN 
# ==========================================
LOGO_LOGIN = "logo_nailsense.png"
LOGO_APP   = "logo_nailsense.png"

def get_base64_of_image(image_path: str) -> str | None:
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None


def get_logo_html(logo_file: str, scale: float = 1.0) -> str:
    base64_data = get_base64_of_image(logo_file)
    if base64_data is None:
        return '<span style="color:var(--cream);font-weight:bold;">NS</span>'

    ext = logo_file.rsplit(".", 1)[-1].lower()
    mime_type = "jpeg" if ext in ["jpg", "jpeg"] else "png"

    return (
        f'<img src="data:image/{mime_type};base64,{base64_data}" '
        f'style="width:{int(scale*100)}%;height:{int(scale*100)}%;'
        f'object-fit:contain;flex-shrink:0;">'  # ← cover → contain
    )


def render_logo(large: bool = False):
    """Logo header halaman dalam aplikasi — ukuran kecil (95px)."""
    if large:
        return

    logo_content = get_logo_html(LOGO_APP, scale=1.0)

    st.markdown(f"""
        <style>
        .ns-logo-circle.small {{
            width: 95px !important;
            height: 95px !important;
            border-radius: 50% !important;
            overflow: hidden !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin-left: 8px !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            background: var(--cream) !important;
        }}
        .ns-logo-circle.small img {{
            width: 100% !important;
            height: 100% !important;
            object-fit: contain !important;  /* ← cover → contain */
            flex-shrink: 0 !important;
        }}
        </style>
        <div class="ns-logo-wrap">
            <div class="ns-logo-circle small">
                {logo_content}
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_page_title(title: str, subtitle: str = None):
    """Judul halaman satu baris, bold, tengah, warna maroon — sesuai mockup."""
    st.markdown(f'<div class="ns-page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="ns-page-subtitle">{subtitle}</div>', unsafe_allow_html=True)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Dancing+Script:wght@600;700&display=swap');

    :root {
        --maroon: #7B2A3D;
        --maroon-dark: #5E2030;
        --pink-accent: #C2185B;
        --pink-soft: #FBEAF0;
        --pink-border: #F0C8D6;
        --blue-btn: #ADCBDB;
        --blue-btn-hover: #93B7CB;
        --cream: #FBF7F1;
        --frame-border: #000000;
        --text-muted: #8A8A8A;
        --lavender-fill: #F1EEF6;
    }

    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif !important; }

    .block-container {
        max-width: 415px !important;
        height: 760px !important;
        min-height: 770px !important;
        max-height: 770px !important;
        padding: 0rem 0rem 0rem 0rem !important;
        border: 7px solid var(--frame-border);
        border-radius: 40px;
        box-shadow: 0px 12px 28px rgba(0,0,0,0.18);
        background-color: var(--cream);
        margin: auto;
        position: relative !important;
        overflow: hidden !important;
        display: flex;
        flex-direction: column;
    }
    .stMainBlockContainer, [data-testid="stVerticalBlock"] {
        overflow-y: auto !important;
        max-height: 660px !important;
        padding-left: 1.3rem !important;
        padding-right: 1.3rem !important;
        padding-bottom: 5rem !important;
    }
    .stMainBlockContainer::-webkit-scrollbar,
    [data-testid="stVerticalBlock"]::-webkit-scrollbar { width: 4px; }
    .stMainBlockContainer::-webkit-scrollbar-thumb,
    [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
        background: #E5D6E0; border-radius: 10px;
    }
    header, footer { visibility: hidden !important; }

    /* ===== LOGO BULAT ===== */
    .ns-logo-wrap { display:flex; justify-content:center; align-items:center; margin: 14px 0 8px 0; }
    .ns-logo-circle {
        width: 60px; height: 60px; border-radius: 50%;
        background: var(--cream);
        border: 2.3px solid var(--maroon);
        display:flex; align-items:center; justify-content:center;
    }
    .ns-logo-circle.large {
        width: 108px; height: 108px; flex-direction: column; border-width: 3px;
        background: var(--cream);
    }
    .ns-logo-icon { width: 24px; height: 24px; }
    .ns-logo-circle.large .ns-logo-icon { width: 32px; height: 32px; margin-bottom: 3px; }
    .ns-logo-word { font-family: 'Dancing Script', cursive; color: var(--maroon); font-weight: 700; font-size: 15px; line-height: 1; }

    /* ===== HEADER MAROON (halaman login) — mentok ke tepi atas & samping, ikon center vertikal ===== */
    .stMainBlockContainer { padding-top: 0 !important; }
    .ns-auth-header {
        background: var(--maroon);
        /* Menarik paksa margin agar benar-benar mentok ke frame hitam terluar */
        margin: 0rem -1.3rem 0rem -1.3rem !important; 
        padding: 40px 0px 60px 0px !important;
        min-height: 260px;
        width: calc(100% + 2.6rem) !important;
        display: flex;
        align-items: center;
        justify-content: center;
        /* Membuat lengkungan estetik di bagian bawah background maroon */
        border-radius: 0px 0px 40px 40px !important; 
        position: relative;
        z-index: 1;
    }
    .ns-auth-header .ns-logo-wrap { margin: 0; }
    .ns-auth-header .ns-logo-circle { background: var(--cream); }

    /* ===== KARTU FORM MENUMPUK RAPI DI ATAS MAROON (TANPA JALUR GESER) ===== */
    .st-key-auth_card {
        background: var(--cream) !important;
        /* Membuat lengkungan di atas kartu putih/krem agar kontras dengan lengkungan maroon */
        border-radius: 35px 35px 0px 0px !important; 
        /* Menarik kartu ke atas agar menumpuk di atas background maroon */
        margin: -45px -1.3rem 0px -1.3rem !important; 
        padding: 30px 1.5rem 20px 1.5rem !important;
        position: relative !important;
        z-index: 2 !important;
        width: calc(100% + 2.6rem) !important;
        box-sizing: border-box !important;
        transform: translateX(22px) !important; 
    }

    /* ===== JUDUL HALAMAN ===== */
    .ns-page-title {
        text-align: center; color: var(--maroon); font-weight: 700;
        font-size: 20px; line-height: 1.3; margin: 4px 0 18px 0;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .ns-page-subtitle {
        text-align: center; color: var(--text-muted); font-size: 12.5px;
        margin-top: -12px; margin-bottom: 16px; padding: 0 8px;
    }

    /* ===== TABS (Masuk / Daftar) ===== */
    .stTabs [data-baseweb="tab-list"] { gap: 22px; justify-content: center; border-bottom: 1px solid var(--pink-border); }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); font-weight: 600; font-size: 14.5px; padding-bottom: 9px; }
    .stTabs [aria-selected="true"] { color: var(--maroon) !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--maroon) !important; height: 2.5px !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 18px; }

    /* ===== LABEL INPUT ===== */
    [data-testid="stWidgetLabel"] p, label {
        color: var(--maroon) !important; font-weight: 600 !important; font-size: 13px !important;
    }

    /* ===== TEXT INPUT / SELECT (perbaikan border ganda) ===== */
    [data-testid="stTextInput"] div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1.4px solid var(--pink-border) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        overflow: hidden;
    }
    [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
        border-color: var(--maroon) !important;
        box-shadow: none !important;
    }
    [data-testid="stTextInput"] div[data-baseweb="base-input"],
    [data-testid="stTextInputRootElement"],
    [data-testid="stTextInput"] div[data-baseweb="input"] > div {
        background-color: transparent !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stTextInput"] input {
        background-color: transparent !important;
        color: #4A3B40 !important;
        padding: 10px 12px !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stTextInput"] button {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1.4px solid var(--pink-border) !important;
        border-radius: 12px !important;
    }
    ::placeholder { color: #C5AEB7 !important; opacity: 1 !important; }

    /* ===== TOMBOL UMUM — maroon dengan teks krem (selector ganda agar kompatibel semua versi Streamlit) ===== */
    div.stButton > button,
    div[data-testid="stButton"] button,
    .stButton button,
    button[kind="primary"],
    button[kind="secondary"],
    button[kind="primaryFormSubmit"],
    button[kind="secondaryFormSubmit"],
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-secondary"],
    button[data-testid^="stBaseButton"] {
        width: 100% !important;
        background-color: var(--maroon) !important;
        color: var(--cream) !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 12px 8px !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        white-space: nowrap !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, color 0.15s ease;
    }
    div.stButton > button p,
    div[data-testid="stButton"] button p,
    .stButton button p,
    button[kind="primary"] p,
    button[kind="secondary"] p,
    button[data-testid^="stBaseButton"] p {
        color: var(--cream) !important;
        font-weight: 700 !important;
    }
    div.stButton > button:hover,
    div[data-testid="stButton"] button:hover,
    .stButton button:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover,
    button[data-testid^="stBaseButton"]:hover {
        background-color: var(--maroon-dark) !important;
        color: var(--cream) !important;
    }
    div.stButton > button:hover p,
    div[data-testid="stButton"] button:hover p,
    .stButton button:hover p {
        color: var(--cream) !important;
    }

    /* ===== ALERT / INFO BOX (tips) ===== */
    .stAlert { border-radius: 16px !important; border: none !important; padding: 14px 16px !important; }
    .stAlert p { font-size: 13px !important; line-height: 1.6 !important; }
    .stAlert svg { display: none !important; }

    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--pink-soft) !important;
        border: 1.6px dashed var(--pink-border) !important;
        border-radius: 18px !important;
    }
    [data-testid="stFileUploaderDropzone"] svg { fill: var(--pink-accent) !important; color: var(--pink-accent) !important; }
    [data-testid="stFileUploaderDropzone"] span { color: var(--maroon) !important; font-weight: 600 !important; }
    [data-testid="stFileUploaderDropzone"] small { color: var(--text-muted) !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: var(--maroon) !important;
        color: var(--cream) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: var(--maroon-dark) !important;
        color: var(--cream) !important;
    }
    [data-testid="stFileUploaderDropzone"] button p,
    [data-testid="stFileUploaderDropzone"] button span {
        color: var(--cream) !important;
    }
    [data-testid="stFileUploaderDropzone"] button svg {
        display: none !important;
    }
    [data-testid="stFileUploader"] [data-testid="stBaseButton-minimal"],
    [data-testid="stFileUploader"] button[kind="minimal"] {
        display: none !important;
    }

    /* ===== EXPANDER (Edukasi) ===== */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1.4px solid var(--pink-border) !important;
        border-radius: 16px !important;
        margin-bottom: 12px !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        color: var(--pink-accent) !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        padding: 13px 14px !important;
    }
    [data-testid="stExpander"] summary svg { color: var(--pink-accent) !important; fill: var(--pink-accent) !important; }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 0 14px 14px 14px !important; font-size: 13px !important; color: #4A3B40 !important;
    }

    /* ===== METRIC (Riwayat) ===== */
    [data-testid="stMetric"] {
        background-color: #FFFFFF; border: 1.2px solid var(--pink-border);
        border-radius: 14px; padding: 10px 6px;
    }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 11px !important; }
    [data-testid="stMetricValue"] { color: var(--maroon) !important; font-size: 19px !important; }

    /* ===== DATAFRAME ===== */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1.2px solid var(--pink-border); }

    /* ===== NAVIGASI BAWAH (discope ke container key="bottom_nav" agar tidak bentrok kolom lain) ===== */
    .st-key-bottom_nav {
        position: absolute !important;
        bottom: 0px !important;
        left: 0px !important;
        right: 0px !important;
        width: 100% !important;
        height: 74px !important;
        background-color: var(--maroon) !important;
        border-radius: 0px 0px 33px 33px !important;
        overflow: hidden !important;
        z-index: 99999 !important;
        box-sizing: border-box !important;
    }
    .st-key-bottom_nav [data-testid="stHorizontalBlock"] {
        width: 100% !important;
        height: 100% !important;
        background-color: transparent !important;
        border-radius: 0px !important;
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        justify-content: space-evenly !important;
        align-items: center !important;
        padding: 0px 0px 0px 0px !important;
        margin: 0px !important;
        gap: 0px !important;
        box-sizing: border-box !important;
        position: relative !important;
        top: 16% !important;
        transform: translateY(35%) !important;
    }
    .st-key-bottom_nav [data-testid="stHorizontalBlock"] > div,
    .st-key-bottom_nav [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        width: 25% !important; min-width: 25% !important; max-width: 25% !important;
        flex: 1 1 25% !important; padding: 0px !important; margin: 0px !important;
        display: flex !important; justify-content: center !important;
        align-items: center !important; height: 100% !important;
    }
    .st-key-bottom_nav [data-testid="stVerticalBlock"] {
        padding-bottom: 0 !important;
        padding-top: 0 !important;
        max-height: none !important;
        overflow: visible !important;
    }
    .st-key-bottom_nav div.stButton {
        width: auto !important; height: auto !important;
        display: flex !important; justify-content: center !important;
        align-items: center !important; margin: 0px !important; padding: 0px !important;
    }
    .st-key-bottom_nav div.stButton > button {
        background-color: transparent !important;
        border: none !important;
        line-height: 1 !important;
        padding: 0px !important;
        height: 42px !important;
        width: 42px !important;
        box-shadow: none !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 0 auto !important;
        border-radius: 50% !important;
        filter: none !important;
        transition: all 0.15s ease;
    }
    .st-key-bottom_nav div.stButton > button:active { transform: scale(0.9); }
    .st-key-bottom_nav div.stButton > button *,
    .st-key-bottom_nav div.stButton > button p,
    .st-key-bottom_nav div.stButton > button span {
        font-size: 20px !important;
        filter: none !important;
    }

    #logout-box {
        background-color: #FFFDE7 !important;
        border: 1px solid #FFF59D !important;
        border-radius: 14px !important;
        padding: 14px !important;
        margin-top: 10px !important;
        margin-bottom: 20px !important;
    }
    </style>
""", unsafe_allow_html=True)

# =================================================================
# 1. FUNGSI BANTU STATISTIK GAMBAR
# =================================================================
def _rgb_stats(img_rgb: Image.Image) -> dict:
    arr = np.array(img_rgb.convert('RGB'), dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return {
        'r_mean': float(np.mean(r)),
        'g_mean': float(np.mean(g)),
        'b_mean': float(np.mean(b)),
        'r_std' : float(np.std(r)),
        'g_std' : float(np.std(g)),
        'b_std' : float(np.std(b)),
        'color_std': float(np.std(arr)),
    }

def _hsv_saturation_mean(img_rgb: Image.Image) -> float:
    arr  = np.array(img_rgb.convert('RGB'), dtype=np.float32) / 255.0
    r, g, b  = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    sat  = np.where(cmax > 0, (cmax - cmin) / cmax, 0.0)
    return float(np.mean(sat) * 100)

def _laplacian_var(img_gray: Image.Image) -> float:
    edges = img_gray.filter(ImageFilter.FIND_EDGES)
    return float(np.var(np.array(edges)))

def _edge_density(img_gray: Image.Image) -> float:
    edges = np.array(img_gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    threshold = 255 * 0.10
    return float(np.mean(edges > threshold))

# =================================================================
# 2. DETEKSI & CROP AREA KUKU (PENTING — mengatasi train-serve skew)
# =================================================================
def detect_and_crop_nail(image: Image.Image) -> Image.Image:
    img_rgb = np.array(image.convert('RGB'))
    h, w = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Rect awal: asumsikan kuku berada di 80% area tengah foto
    rx, ry = int(w * 0.10), int(h * 0.10)
    rw, rh = int(w * 0.80), int(h * 0.80)
    rect = (rx, ry, rw, rh)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model,
                     5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return image.crop((rx, ry, rx + rw, ry + rh))

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image.crop((rx, ry, rx + rw, ry + rh))

    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)

    # Validasi: tolak hasil crop yang terlalu kecil (gagal deteksi)
    # atau terlalu besar (GrabCut tidak menemukan objek untuk dipotong)
    area_ratio = (cw * ch) / float(w * h)
    if area_ratio < 0.05 or area_ratio > 0.95:
        return image.crop((rx, ry, rx + rw, ry + rh))

    pad = int(0.05 * max(cw, ch))
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + cw + pad, w), min(y + ch + pad, h)

    return image.crop((x0, y0, x1, y1))

# =================================================================
# 3. VALIDASI KUALITAS FOTO
# =================================================================
def validate_image_quality(image: Image.Image) -> tuple[bool, str]:
    img_rgb  = image.convert('RGB')
    img_gray = img_rgb.convert('L')
    w, h     = img_rgb.size

    if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
        return False, "📷 Foto tidak dapat dianalisis — resolusi terlalu kecil."

    lap_var = _laplacian_var(img_gray)
    if lap_var < BLUR_THRESHOLD:
        return False, f"📷 Foto terlalu blur (skor: {lap_var:.0f}, min: {BLUR_THRESHOLD}). Silakan ambil foto ulang yang lebih fokus."

    gray_np     = np.array(img_gray, dtype=np.float32)
    mean_bright = float(np.mean(gray_np))
    if mean_bright < UNDEREXPOSE_THRESHOLD or mean_bright > OVEREXPOSE_THRESHOLD:
        return False, "📷 Foto terlalu gelap atau terlalu terang (pencahayaan tidak ideal). Coba gunakan pencahayaan merata."

    margin_x = w // 6
    margin_y = h // 6
    roi_rgb  = img_rgb.crop((margin_x, margin_y, w - margin_x, h - margin_y))
    roi_gray = roi_rgb.convert('L')

    stats       = _rgb_stats(roi_rgb)
    saturation  = _hsv_saturation_mean(roi_rgb)
    edge_den    = _edge_density(roi_gray)

    fail_reasons = []
    if stats['r_mean'] <= 0 or (stats['b_mean'] / stats['r_mean']) >= NAIL_MIN_R_OVER_B:
        fail_reasons.append("warna biru/hijau dominan")
    if stats['color_std'] > NAIL_COLOR_STD_MAX:
        fail_reasons.append("variasi warna terlalu tinggi")
    if saturation > NAIL_SAT_MAX:
        fail_reasons.append("saturasi terlalu tinggi")
    if edge_den > NAIL_EDGE_DENSITY_MAX:
        fail_reasons.append("terlalu banyak garis/tekstur non-kuku")

    if len(fail_reasons) >= 3:
        return False, f"📷 Objek tidak terdeteksi sebagai kuku. Alasan: {'; '.join(fail_reasons)}. Pastikan posisi jari tegak lurus dan memenuhi area foto."

    return True, ""

# =================================================================
# 4. PREPROCESSING & LOGIKA PREDIKSI PURE SOFTMAX
# =================================================================
def preprocess_image(image: Image.Image) -> tuple:
    """
    Mengembalikan (img_array, tabular_arr, cropped_image).
    cropped_image dikembalikan agar bisa dipakai ulang untuk
    validate_image_quality() dan ditampilkan ke user, sehingga
    validasi & prediksi konsisten memakai gambar yang SAMA.
    """
    # Langkah krusial: crop ke region kuku agar sesuai distribusi data latih
    image_cropped = detect_and_crop_nail(image)

    img_rgb = np.array(image_cropped.convert('RGB'))
    img_resized = cv2.resize(img_rgb, (TARGET_SIZE[0], TARGET_SIZE[1]))
    img_np = img_resized.astype(np.float32)

    tabular_arr = None
    if MODEL_N_INPUTS == 2:
        R = img_np[:, :, 0].flatten()
        G = img_np[:, :, 1].flatten()
        B = img_np[:, :, 2].flatten()

        r5,  r95 = np.percentile(R, 5),  np.percentile(R, 95)
        g5,  g95 = np.percentile(G, 5),  np.percentile(G, 95)
        b5,  b95 = np.percentile(B, 5),  np.percentile(B, 95)

        tabular_arr = np.array([[r5, r95, g5, g95, b5, b95]], dtype=np.float32) / 255.0

    img_norm = img_np / 255.0
    img_array = np.expand_dims(img_norm, axis=0)

    return img_array, tabular_arr, image_cropped

def predict_anemia(img_array, tabular_array) -> dict:
    if model is None:
        return {"hasil": "Error", "confidence": 0.0, "probabilitas": -1.0, "catatan": model_error}
    try:
        img_in = img_array.astype(np.float32)

        if MODEL_N_INPUTS == 2 and tabular_array is not None:
            pred = model.predict([img_in, tabular_array.astype(np.float32)], verbose=0)
        else:
            pred = model.predict(img_in, verbose=0)

        skor_normal = float(pred[0][0])
        skor_anemia = float(pred[0][1])

        indeks_tertinggi = np.argmax(pred[0])

        if indeks_tertinggi == 1:
            hasil = "Anemia"
            confidence = skor_anemia * 100
        else:
            hasil = "Normal"
            confidence = skor_normal * 100

        catatan = ""
        if confidence < MIN_CONFIDENCE:
            catatan = f"⚠️ Keyakinan model cukup rendah ({confidence:.1f}%). Kondisi kuku berada di batas samar. Disarankan foto ulang dengan pencahayaan merata."

        return {
            "hasil"       : hasil,
            "confidence"  : round(confidence, 2),
            "probabilitas": round(skor_anemia, 4),
            "catatan"     : catatan,
        }
    except Exception as e:
        return {"hasil": "Error", "confidence": 0.0, "probabilitas": -1.0, "catatan": f"Error saat prediksi: {str(e)}"}
    
# ==========================================
# 5. VALIDASI PASSWORD
# ==========================================
def validasi_password(pw: str) -> bool:
    return (len(pw) >= 8
            and bool(re.search("[a-zA-Z]", pw))
            and bool(re.search("[0-9]", pw)))

# ==========================================
# 6. STATE MANAGEMENT
# ==========================================
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'user' not in st.session_state: st.session_state['user'] = None
if 'current_page' not in st.session_state: st.session_state['current_page'] = "Skrining"
if 'logout_confirm' not in st.session_state: st.session_state['logout_confirm'] = False

# ==========================================
# 7. HALAMAN AUTENTIKASI
# ==========================================
def show_auth_page():
    logo_content = get_logo_html(LOGO_LOGIN, scale=1.0)

    st.markdown("""
        <style>
        .stMainBlockContainer, [data-testid="stVerticalBlock"] {
            padding-top: 0px !important;
        }
        .ns-auth-header {
            background: var(--maroon);
            margin: -1rem -1.3rem 0rem -1.3rem !important;
            padding: 45px 0px 65px 0px !important;
            min-height: 260px;
            width: calc(100% + 2.6rem) !important;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 0px 0px 40px 40px !important;
            position: relative;
            z-index: 10 !important;
        }
        .ns-logo-wrap.auth-layout {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            text-align: center !important;
        }
        .ns-logo-circle.large {
            width: 140px !important;
            height: 140px !important;
            border-radius: 50% !important;
            overflow: hidden !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background: var(--cream) !important;
            border: 4px solid var(--maroon) !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            margin: 0 auto !important;
        }
        .ns-logo-circle.large img {
            width: 100% !important;
            height: 100% !important;
            object-fit: contain !important;  
        }
        .ns-logo-word {
            margin-top: 12px !important;
            font-size: 34px !important;
            font-weight: bold !important;
            color: var(--cream) !important;
        }
        .st-key-auth_card {
            background: var(--cream) !important;
            border-radius: 35px 35px 0px 0px !important;
            margin: -40px -1.3rem 0px -1.3rem !important;
            padding: 35px 1.5rem 20px 1.5rem !important;
            position: relative !important;
            z-index: 99 !important;
            width: calc(100% + 2.6rem) !important;
            box-sizing: border-box !important;
            transform: translateX(22px) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="ns-auth-header">
            <div class="ns-logo-wrap auth-layout">
                <div class="ns-logo-circle large">
                    {logo_content}
                </div>
                <div class="ns-logo-word">NailSense</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with st.container(key="auth_card"):
        tab1, tab2 = st.tabs(["Masuk", "Daftar"])

        with tab1:
            username = st.text_input("Username", placeholder="Masukkan username", key="login_user")
            password = st.text_input("Password", type="password", placeholder="Masukkan password", key="login_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Masuk", type="primary", key="btn_masuk"):
                if not username or not password:
                    st.error("Semua kolom harus diisi!")
                else:
                    conn   = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
                    if not cursor.fetchone():
                        st.error("❌ Akun belum terdaftar. Silakan registrasi di tab Daftar!")
                    else:
                        cursor.execute("SELECT * FROM users WHERE username=? AND password=?",
                                       (username, password))
                        if cursor.fetchone():
                            st.session_state['logged_in'] = True
                            st.session_state['user']      = username
                            st.rerun()
                        else:
                            st.error("❌ Username atau password salah.")
                    conn.close()

        with tab2:
            new_user     = st.text_input("Buat Username", placeholder="Masukkan username", key="reg_user")
            new_pass     = st.text_input("Buat Password (min. 8 karakter, huruf & angka)", placeholder="Masukkan password",
                                         type="password", key="reg_pass")
            nama         = st.text_input("Nama Lengkap", placeholder="Masukkan nama lengkap", key="reg_nama")
            usia_input   = st.text_input("Usia", placeholder="Masukkan usia Anda", key="reg_usia")
            gender       = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], key="reg_gender")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Registrasi", type="primary", key="btn_registrasi"):
                if not usia_input:
                    st.error("Kolom usia wajib diisi!")
                    return
                try:
                    usia = int(usia_input)
                except ValueError:
                    st.error("Usia harus berupa angka!")
                    return
                if not all([new_user, new_pass, nama]):
                    st.error("Semua data wajib diisi!")
                elif not validasi_password(new_pass):
                    st.error("Password minimal 8 karakter dengan kombinasi huruf & angka.")
                else:
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        conn.cursor().execute(
                            "INSERT INTO users VALUES (?,?,?,?,?)",
                            (new_user, new_pass, nama, usia, gender))
                        conn.commit(); conn.close()
                        st.success("Registrasi sukses! Silakan masuk di tab Masuk.")
                    except sqlite3.IntegrityError:
                        st.error("Username sudah digunakan.")

# ==========================================
# 8. HALAMAN SKRINING
# ==========================================
def page_skrining():
    render_logo()
    render_page_title("🔍 Skrining Anemia")

    if model is None:
        st.error(f"❌ **Model AI tidak tersedia.**\n\n{model_error}\n\n"
                 "Pastikan file `nailsense_hibrida_softmaxfixx.keras` berada di path yang benar.")
        return

    st.info(
        "💡 **Tips foto terbaik:**\n"
        "- Foto close-up 1–2 jari, fokus pada bantalan kuku\n"
        "- Disarankan menggunakan flash kamera\n"
        "- Tanpa cat kuku, kuku bersih\n"
        "- Kamera sejajar dengan kuku, tidak miring"
    )

    if 'foto_terkirim' not in st.session_state: st.session_state['foto_terkirim'] = False
    if 'analisis_selesai' not in st.session_state: st.session_state['analisis_selesai'] = False
    if 'uploader_key' not in st.session_state: st.session_state['uploader_key'] = 0
    if 'step_tanya' not in st.session_state: st.session_state['step_tanya'] = 0
    if 'jawaban_user' not in st.session_state: st.session_state['jawaban_user'] = {}

    uploaded_file = st.file_uploader(
        "Unggah Foto Bantalan Kuku",
        type=["jpg", "jpeg", "png"],
        key=f"kuku_uploader_{st.session_state['uploader_key']}"
    )

    if uploaded_file is None:
        st.session_state['foto_terkirim'] = False
        st.session_state['analisis_selesai'] = False
        st.session_state['step_tanya'] = 0
        st.session_state['jawaban_user'] = {}
        if 'hasil_terakhir' in st.session_state: del st.session_state['hasil_terakhir']

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        if not st.session_state['analisis_selesai']:
            st.image(image, caption="Foto yang diunggah", use_container_width=True)
            st.divider()

        if not st.session_state['foto_terkirim']:
            if st.button("Kirim Foto", type="primary"):
                st.session_state['foto_terkirim'] = True
                st.rerun()

        elif st.session_state['foto_terkirim'] and not st.session_state['analisis_selesai']:
            current_step = st.session_state['step_tanya']
            total_steps = len(PERTANYAAN_GEJALA)

            if current_step < total_steps:
                st.markdown(f"##### Kuesioner Gejala Anemia ({current_step + 1}/{total_steps})")
                q_now = PERTANYAAN_GEJALA[current_step]
                
                list_opsi = ["-- Pilih Jawaban --"] + list(OPSI_JAWABAN.keys())
                saved_ans = st.session_state['jawaban_user'].get(q_now['id'], "-- Pilih Jawaban --")
                idx_default = list_opsi.index(saved_ans) if saved_ans in list_opsi else 0

                pilihan = st.radio(q_now['tanya'], options=list_opsi, index=idx_default, key=f"q_{q_now['id']}")
                st.markdown("<br>", unsafe_allow_html=True)
                
                col_nav1, col_nav2 = st.columns(2)
                with col_nav1:
                    if current_step > 0:
                        if st.button("⬅️ Kembali", type="secondary", use_container_width=True):
                            st.session_state['step_tanya'] -= 1
                            st.rerun()
                with col_nav2:
                    label_tombol_nav = "Mulai Analisis" if current_step == total_steps - 1 else "Selanjutnya ➡️"
                    if st.button(label_tombol_nav, type="primary", use_container_width=True):
                        if pilihan == "-- Pilih Jawaban --":
                            st.error("⚠️ Silakan pilih salah satu jawaban terlebih dahulu sebelum melanjutkan!")
                        else:
                            st.session_state['jawaban_user'][q_now['id']] = pilihan
                            st.session_state['step_tanya'] += 1
                            st.rerun()

        if st.session_state['foto_terkirim'] and st.session_state['step_tanya'] >= len(PERTANYAAN_GEJALA) and not st.session_state['analisis_selesai']:
            with st.spinner("Mendeteksi area kuku..."):
                img_arr, tab_arr, cropped_img = preprocess_image(image)
            with st.spinner("Memeriksa kualitas foto..."):
                valid, err_msg = validate_image_quality(cropped_img)

            if not valid:
                st.session_state['hasil_terakhir'] = {"status": "gagal_kualitas", "pesan": err_msg}
                st.session_state['analisis_selesai'] = True
                st.rerun()

            with st.spinner("Menganalisis kuku & gejala..."):
                result = predict_anemia(img_arr, tab_arr)

            if result["hasil"] == "Error":
                st.session_state['hasil_terakhir'] = {"status": "gagal_sistem", "pesan": result['catatan']}
                st.session_state['analisis_selesai'] = True
                st.rerun()

            skor_gejala = sum([OPSI_JAWABAN[val] for val in st.session_state['jawaban_user'].values()])
            persentase_gejala = (skor_gejala / 15.0)
            prob_anemia_ai = result["probabilitas"]
            
            prob_hibrida = (0.65 * prob_anemia_ai) + (0.35 * persentase_gejala)

            if prob_hibrida >= 0.50:
                result["hasil"] = "Anemia"
                result["confidence"] = round(prob_hibrida * 100, 2)
            else:
                result["hasil"] = "Normal"
                result["confidence"] = round((1.0 - prob_hibrida) * 100, 2)

            result["catatan"] = result["catatan"] + f" Skor keparahan gejala klinis Anda adalah {skor_gejala}/15 poin."

            st.session_state['hasil_terakhir'] = {
                "status": "sukses",
                "data": result,
                "cropped_preview": cropped_img,
            }
            st.session_state['analisis_selesai'] = True

            try:
                tanggal = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                conn = sqlite3.connect(DB_PATH)
                conn.cursor().execute(
                    "INSERT INTO history (username,tanggal,hasil,confidence,catatan) VALUES (?,?,?,?,?)",
                    (st.session_state['user'], tanggal, result["hasil"], result["confidence"], result["catatan"]))
                conn.commit(); conn.close()
            except Exception as e:
                st.warning(f"Hasil tidak dapat disimpan ke riwayat: {e}")
            st.rerun()

        if st.session_state['analisis_selesai']:
            res_state = st.session_state['hasil_terakhir']
            
            if res_state["status"] in ["gagal_kualitas", "gagal_sistem"]:
                st.error(res_state["pesan"])
                
            elif res_state["status"] == "sukses":
                result = res_state["data"]
                st.image(res_state["cropped_preview"], caption="Area kuku yang dianalisis", use_container_width=True)
                st.markdown(f"**Tingkat Keyakinan (Hybrid Confidence):** `{result['confidence']:.2f}%`")

                if result["hasil"] == "Anemia":
                    st.error("🔴 **Hasil: Terindikasi Anemia**")
                    st.markdown("""
**Apa artinya?**
Bantalan kuku Anda terdeteksi lebih pucat dari normal, yang bisa menjadi tanda anemia (kekurangan hemoglobin/zat besi).

**Rekomendasi segera:**
- 🥩 Konsumsi makanan kaya **zat besi heme**: daging merah, hati ayam/sapi, ikan tuna
- 🍊 Perbanyak **Vitamin C** (jeruk, tomat, paprika) untuk bantu penyerapan zat besi
- 🥬 Tambah **sayuran hijau**: bayam, brokoli, kacang-kacangan
- ❌ Hindari **teh/kopi** saat atau setelah makan — tanin hambat penyerapan besi
- 🏥 **Periksa ke dokter** untuk cek darah lengkap (Hb, ferritin, serum iron)

> ⚠️ Aplikasi ini bukan pengganti diagnosis dokter. Hasil skrining ini hanya sebagai deteksi awal.
""")
                else:
                    st.success("🟢 **Hasil: Kondisi Kuku Normal**")
                    st.markdown("""
**Apa artinya?**
Warna bantalan kuku Anda dalam rentang normal — tidak terdeteksi tanda-tanda anemia dari foto ini.

**Pertahankan gaya hidup sehat:**
- 🥗 Konsumsi makanan bergizi seimbang setiap hari
- 💧 Minum air putih minimal 2 liter per hari
- 🏃 Olahraga rutin 30 menit/hari, minimal 3x seminggu
- 😴 Istirahat cukup 7–8 jam per malam
- 🩺 Lakukan pemeriksaan darah rutin setiap 6–12 bulan

> ℹ️ Lakukan skrining ulang jika Anda merasa mudah lelah, pusing, atau pucat.
""")
                    st.markdown("---")
            if st.button("Upload Ulang", type="secondary"):
                st.session_state['foto_terkirim'] = False
                st.session_state['analisis_selesai'] = False
                st.session_state['step_tanya'] = 0
                st.session_state['jawaban_user'] = {}
                st.session_state['uploader_key'] += 1
                if 'hasil_terakhir' in st.session_state: del st.session_state['hasil_terakhir']
                st.rerun()

# ==========================================
# 9. HALAMAN EDUKASI 
# ==========================================
def page_edukasi():
    render_logo()
    render_page_title("📚 Edukasi Anemia", subtitle="Klik salah satu topik di bawah untuk membaca informasinya.")

    with st.expander("📖 1. Apa itu Anemia?", expanded=False):
        st.markdown("""
**Anemia** adalah kondisi medis di mana kadar hemoglobin (Hb) atau jumlah sel darah merah (eritrosit) dalam darah berada di bawah nilai normal. Hemoglobin adalah protein yang mengandung zat besi dan berfungsi sebagai "kendaraan" oksigen dari paru-paru ke seluruh jaringan tubuh.

**Nilai normal kadar Hemoglobin:**
| Kelompok | Nilai Normal Hb |
|---|---|
| Pria dewasa | ≥ 13,0 g/dL |
| Wanita dewasa | ≥ 12,0 g/dL |
| Ibu hamil | ≥ 11,0 g/dL |
| Anak usia 6–14 tahun | ≥ 12,0 g/dL |
| Anak usia 6 bulan–5 tahun | ≥ 11,0 g/dL |

Ketika hemoglobin rendah, tubuh kekurangan oksigen sehingga organ-organ tidak dapat bekerja optimal. Indonesia termasuk negara dengan prevalensi anemia tinggi — data Kemenkes 2018 menunjukkan sekitar **48,9% remaja putri** dan **38,5% ibu hamil** di Indonesia mengalami anemia.

**Kenapa kuku bisa mendeteksi anemia?**
Bantalan kuku (nail bed) adalah jaringan di bawah lempeng kuku yang kaya pembuluh darah kapiler. Pada orang sehat, kuku terlihat merah-muda karena darah kaya hemoglobin di bawahnya. Pada penderita anemia, hemoglobin rendah menyebabkan darah lebih pucat, sehingga bantalan kuku tampak **pucat, putih, atau kekuningan**.
""")

    with st.expander("🔍 2. Gejala Umum Anemia", expanded=False):
        st.markdown("""
Gejala anemia bervariasi dari ringan hingga berat, tergantung seberapa rendah kadar hemoglobin dan seberapa cepat penurunannya.

**Gejala klasik (5L):**
- **Lemah** — tubuh terasa tidak bertenaga meski tidak banyak aktivitas
- **Letih** — cepat lelah bahkan setelah istirahat
- **Lesu** — tidak bersemangat menjalani aktivitas sehari-hari
- **Lelah** — stamina menurun signifikan
- **Lalai (susah konsentrasi)** — otak kekurangan oksigen menyebabkan sulit fokus

**Gejala fisik yang bisa dilihat:**
- 🫧 **Pucat** pada bantalan kuku, telapak tangan, wajah, bibir, gusi, dan konjungtiva (kelopak mata dalam)
- 💨 **Sesak napas** saat aktivitas ringan seperti naik tangga
- 💓 **Jantung berdebar** (palpitasi) karena jantung bekerja lebih keras
- 🤕 **Sakit kepala** dan pusing, terutama saat berdiri tiba-tiba (ortostatik)
- 🧊 **Tangan dan kaki dingin** karena sirkulasi darah ke ekstremitas berkurang
- 👅 **Lidah terasa nyeri** atau permukaan lidah tampak halus/licin (glossitis)
- 🦴 **Kuku rapuh, berlekuk, atau berbentuk sendok** (koilonychia) pada anemia berat

**Kapan harus ke dokter segera?**
Segera periksakan diri jika Anda mengalami sesak napas berat, nyeri dada, pingsan, atau denyut jantung sangat cepat dan tidak teratur.
""")

    with st.expander("🧪 3. Jenis & Penyebab Anemia", expanded=False):
        st.markdown("""
Ada lebih dari 400 jenis anemia. Berikut yang paling sering ditemukan di Indonesia:

---
**A. Anemia Defisiensi Zat Besi** *(Paling umum — 50% dari semua kasus)*
- **Penyebab:** Asupan zat besi kurang, penyerapan terganggu, atau kehilangan darah (menstruasi berat, perdarahan saluran cerna, cacing tambang)
- **Ciri khas:** Kuku berbentuk sendok (koilonychia), lidah nyeri, rambut rontok
- **Risiko tinggi:** Remaja putri, ibu hamil, vegetarian, anak-anak

**B. Anemia Defisiensi Vitamin B12 & Asam Folat** *(Anemia Megaloblastik)*
- **Penyebab:** Kurang konsumsi daging/produk hewani (B12), kurang sayuran hijau (folat), gangguan penyerapan usus
- **Ciri khas:** Sel darah merah berukuran besar tapi tidak berfungsi normal, kesemutan di tangan/kaki, gangguan keseimbangan
- **Risiko tinggi:** Vegetarian ketat, ibu hamil, lansia, penderita penyakit Crohn

**C. Anemia Aplastik**
- **Penyebab:** Sumsum tulang gagal memproduksi sel darah — bisa akibat penyakit autoimun, paparan racun, obat-obatan tertentu, atau radiasi
- **Serius:** Perlu penanganan medis segera

**D. Anemia Hemolitik**
- **Penyebab:** Sel darah merah hancur lebih cepat dari produksinya — bisa karena kelainan genetik (talasemia, sickle cell) atau penyakit autoimun
- **Indonesia:** Talasemia cukup banyak ditemukan, terutama di Sulawesi dan Kalimantan

**E. Anemia akibat Penyakit Kronik**
- **Penyebab:** Penyakit ginjal kronik, infeksi kronis (TBC, HIV), peradangan (rheumatoid arthritis, lupus), atau kanker menghambat produksi eritropoietin
- **Ciri khas:** Anemia biasanya ringan hingga sedang, tidak merespons suplemen besi biasa

**F. Anemia pada Ibu Hamil**
- Kebutuhan zat besi meningkat drastis selama kehamilan untuk mendukung pertumbuhan janin dan plasenta
- Anemia pada kehamilan meningkatkan risiko kelahiran prematur, berat badan lahir rendah, dan kematian ibu
""")

    with st.expander("🛡️ 4. Pencegahan & Pengobatan", expanded=False):
        st.markdown("""
**Pencegahan melalui pola makan:**

| Sumber Zat Besi Heme (mudah diserap) | Kandungan Zat Besi |
|---|---|
| Hati sapi/ayam (100g) | 6–7 mg |
| Daging sapi merah (100g) | 2,5–3 mg |
| Ikan tuna/salmon (100g) | 1–1,5 mg |
| Telur ayam (1 butir) | 0,9 mg |

| Sumber Zat Besi Non-Heme (penyerapan 2–20%) | Kandungan Zat Besi |
|---|---|
| Bayam (100g, dimasak) | 3,6 mg |
| Tahu (100g) | 2,7 mg |
| Kacang merah (100g, matang) | 2,6 mg |
| Tempe (100g) | 2,0 mg |
| Biji labu (30g) | 2,5 mg |

**Tips meningkatkan penyerapan zat besi:**
- ✅ Konsumsi **Vitamin C** (jeruk, tomat, paprika, jambu biji) bersamaan dengan makanan kaya zat besi
- ✅ Masak di **wajan besi** — sedikit zat besi dari wajan bisa masuk ke makanan
- ❌ Hindari **teh, kopi, susu** saat makan — tanin & kalsium hambat penyerapan besi
- ❌ Hindari **obat antasida** bersamaan dengan suplemen besi

**Pengobatan medis:**
- 💊 **Suplemen zat besi** (ferrous sulfate/gluconate) — umumnya 60–120 mg/hari selama 3–6 bulan
- 💉 **Suntikan zat besi** — bila tidak bisa minum oral atau penyerapan sangat buruk
- 🩸 **Transfusi darah** — hanya pada anemia berat dengan gejala mengancam jiwa
- 💉 **Eritropoietin** — untuk anemia akibat gagal ginjal
- **Atasi penyebab dasar:** cacing → obat cacing; perdarahan → hentikan sumber perdarahan

**Kebutuhan zat besi harian (RDA):**
- Wanita dewasa (19–50 tahun): **18 mg/hari**
- Pria dewasa: **8 mg/hari**
- Ibu hamil: **27 mg/hari**
- Remaja putri (14–18 tahun): **15 mg/hari**
""")

    with st.expander("🔬 5. Anemia & Warna Kuku", expanded=False):
        st.markdown("""
**Mengapa kuku bisa mencerminkan kondisi darah?**

Bantalan kuku (nail bed) adalah jaringan vaskular padat yang terletak tepat di bawah lempeng kuku transparan. Kapiler-kapiler kecil di sini memberi warna merah-muda khas pada kuku yang sehat.

**Perbandingan warna kuku:**
| Kondisi | Warna Bantalan Kuku |
|---|---|
| Normal | Merah-muda cerah dan merata |
| Anemia ringan | Merah-muda pucat |
| Anemia sedang | Pucat keputihan |
| Anemia berat | Putih/hampir tanpa warna |
| Koilonychia | Kuku berbentuk sendok (cekung) |

**Tes capillary refill (sederhana):**
Tekan ujung kuku hingga memutih, lalu lepas. Normal: warna kembali dalam < 2 detik. Pada anemia atau gangguan sirkulasi, pengisian kembali > 2 detik.

**Bagaimana NailSense bekerja:**
Model AI NailSense dilatih menggunakan dataset foto bantalan kuku yang telah diverifikasi dengan pemeriksaan Hb laboratorium. Model belajar mengenali pola warna dan tekstur halus di bantalan kuku yang berkorelasi dengan kadar hemoglobin, mencakup:
- Distribusi warna merah-muda vs pucat
- Keseragaman warna (anemia sering tidak merata)
- Tekstur permukaan bantalan kuku

**Batasan penting:**
Skrining visual kuku adalah deteksi *awal*. Diagnosis definitif anemia hanya dapat dilakukan melalui pemeriksaan darah lengkap di laboratorium klinik.
""")
        
    with st.expander("🏥 6. Kapan Harus ke Dokter?", expanded=False):
        st.markdown("""
NailSense adalah alat bantu skrining awal — **bukan pengganti pemeriksaan medis**.

**Segera ke dokter / IGD jika:**
- 🚨 Sesak napas berat saat istirahat
- 🚨 Nyeri dada atau jantung berdebar sangat kencang
- 🚨 Pingsan atau hampir pingsan
- 🚨 Pucat ekstrem (wajah, bibir, lidah putih)
- 🚨 Hb < 7 g/dL (jika punya hasil lab)

**Periksakan ke dokter dalam beberapa hari jika:**
- Hasil NailSense menunjukkan terindikasi anemia
- Anda sering lelah padahal cukup tidur
- Mudah pusing, terutama saat berdiri
- Menstruasi sangat berat atau berkepanjangan
- Sedang hamil dan belum periksa darah

**Pemeriksaan lab yang disarankan:**
| Tes | Fungsi |
|---|---|
| Hemoglobin (Hb) | Konfirmasi anemia |
| Hematokrit | Persentase sel darah merah |
| MCV, MCH, MCHC | Menentukan jenis anemia |
| Serum ferritin | Cadangan zat besi tubuh |
| Serum iron & TIBC | Status zat besi |
| Retikulosit | Aktivitas produksi sel darah merah |
| Vitamin B12 & Asam Folat | Cek defisiensi vitamin |

**Faskes yang bisa dituju:**
- Puskesmas terdekat (gratis dengan BPJS)
- Klinik pratama
- Rumah sakit — poli penyakit dalam atau poli anak
""")

# ==========================================
# 10. HALAMAN RIWAYAT
# ==========================================
def page_riwayat():
    render_logo()
    render_page_title("📊 Riwayat Skrining")

    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        "SELECT tanggal, hasil, confidence FROM history WHERE username=? ORDER BY id DESC",
        conn, params=(st.session_state['user'],)
    )
    conn.close()

    if df.empty:
        st.info("Belum ada riwayat skrining. Lakukan skrining pertama Anda di menu 🔍.")
        return

    avg_conf     = df['confidence'].mean()
    st.caption(f"Rata-rata confidence: **{avg_conf:.1f}%**")
    st.divider()

    df_rev = df.copy()
    df_rev['Tren'] = df_rev['hasil'].apply(lambda x: 1 if x == 'Anemia' else 0)
    df_rev = df_rev.iloc[::-1].reset_index(drop=True)
    st.markdown("**Tren Hasil Skrining** (1 = Anemia, 0 = Normal)")
    st.line_chart(data=df_rev, y='Tren', height=150)

    st.dataframe(
        df[['tanggal','hasil','confidence']].rename(columns={
            'tanggal':'Tanggal', 'hasil':'Hasil', 'confidence':'Confidence (%)'}),
        use_container_width=True
    )

# ==========================================
# 11. HALAMAN PROFIL
# ==========================================
def page_profil():
    render_logo()
    render_page_title("👤 Profil Pengguna")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT nama, usia, gender, password FROM users WHERE username = ?", (st.session_state['user'],))
    user_info = cursor.fetchone()
    conn.close()

    if user_info:
        nama, usia, gender, password = user_info
        edit_nama = st.text_input("Nama Lengkap", value=nama)
        edit_usia_input = st.text_input("Usia", value=str(usia))
        edit_gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], index=0 if gender == "Laki-laki" else 1)
        edit_pass = st.text_input("Ubah Password Baru", value=password, type="password")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Simpan Perubahan", type="primary"):
            try: 
                edit_usia = int(edit_usia_input)
            except ValueError:
                st.error("Usia harus berupa angka!")
                return

            if not validasi_password(edit_pass):
                st.error("Password minimal 8 karakter kombinasi huruf & angka.")
            else:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET nama=?, usia=?, gender=?, password=? WHERE username=?",
                               (edit_nama, edit_usia, edit_gender, edit_pass, st.session_state['user']))
                conn.commit()
                conn.close()
                st.success("Profil diperbarui!")
                st.rerun()

        st.divider()
        if not st.session_state['logout_confirm']:
            if st.button("Keluar", type="secondary"):
                st.session_state['logout_confirm'] = True
                st.rerun()
        else:
            st.markdown("""
                <div style="background-color: #FFFDE7; border: 1px solid #FFF59D; border-radius: 12px; padding: 14px; text-align: center; margin-bottom: 15px;">
                    <p style="margin: 0 0 12px 0; font-weight: bold; color: #555555; font-size: 15px;">Yakin ingin logout?</p>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("Iya, Keluar", type="primary", key="logout_iya_fix", use_container_width=True):
                st.session_state['logged_in'] = False
                st.session_state['user'] = None
                st.session_state['logout_confirm'] = False
                st.session_state['current_page'] = "Skrining"
                st.rerun()
                
            if st.button("Tidak, Batal", type="secondary", key="logout_tidak_fix", use_container_width=True):
                st.session_state['logout_confirm'] = False
                st.rerun()
                
# ==========================================
# 12. ROUTER & NAVIGASI BAWAH
# ==========================================
if not st.session_state['logged_in']:
    show_auth_page()
else:
    page = st.session_state['current_page']

    nav_position = {"Skrining": 1, "Edukasi": 2, "Riwayat": 3, "Profil": 4}.get(page, 1)
    st.markdown(f"""
        <style>
        .st-key-bottom_nav [data-testid="stHorizontalBlock"] > div:nth-child({nav_position}) div.stButton > button {{
            background-color: #FFFFFF !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    if   page == "Skrining": page_skrining()
    elif page == "Edukasi" : page_edukasi()
    elif page == "Riwayat" : page_riwayat()
    elif page == "Profil"  : page_profil()

    with st.container(key="bottom_nav"):
        n1, n2, n3, n4 = st.columns(4)
        with n1:
            if st.button("🔍"): st.session_state['current_page']="Skrining"; st.rerun()
        with n2:
            if st.button("📚"): st.session_state['current_page']="Edukasi";  st.rerun()
        with n3:
            if st.button("📊"): st.session_state['current_page']="Riwayat";  st.rerun()
        with n4:
            if st.button("👤"): st.session_state['current_page']="Profil";  st.rerun()
