import streamlit as st
import numpy as np
from PIL import Image
import os

from tensorflow.keras.applications.efficientnet import preprocess_input

# Menggunakan TFLite runtime agar web ringan
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Deteksi Coral Bleaching | Kelompok 5",
    page_icon="🪸",
    layout="centered"
)

# ==========================================
# 2. LOAD MODEL TFLITE
# ==========================================
def load_tflite_model():
    if os.path.exists('model_coral_efficientnet.tflite'):
        try:
            interpreter = tflite.Interpreter(
                model_path='model_coral_efficientnet.tflite'
            )
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    return None

interpreter = load_tflite_model()

# ==========================================
# 3. PREPROCESSING (SAMA DENGAN TRAIN.PY)
# ==========================================
def preprocess_efficientnet(img_array):
    img_array = img_array.astype(np.float32)
    img_array = preprocess_input(img_array)
    return img_array

# ==========================================
# 4. HEADER
# ==========================================
st.title("🪸 Aplikasi Deteksi Dini Pemutihan Terumbu Karang")
st.subheader("Metode Convolutional Neural Network (EfficientNetB0) Berbasis Web")
st.caption("Proyek Tugas Besar Mata Kuliah Pengolahan Citra Digital — Teknik Informatika UMRAH")
st.markdown("---")

# ==========================================
# 5. INPUT GAMBAR
# ==========================================
st.markdown("### 📸 Pilih Metode Input Citra")

tab1, tab2 = st.tabs([
    "📁 Unggah Berkas Gambar",
    "📷 Ambil Foto via Kamera (Webcam)"
])

uploaded_file = None

with tab1:
    file_input = st.file_uploader(
        "Pilih file gambar terumbu karang",
        type=["jpg", "jpeg", "png"]
    )

    if file_input is not None:
        uploaded_file = file_input

with tab2:
    camera_input = st.camera_input(
        "Posisikan objek tepat di depan kamera"
    )

    if camera_input is not None:
        uploaded_file = camera_input

# ==========================================
# 6. PROSES PREDIKSI
# ==========================================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Citra Terumbu Karang",
        use_container_width=True
    )

    st.success("✔ Berkas citra berhasil dimuat.")

    if st.button("Jalankan Klasifikasi Citra", type="primary"):

        st.markdown("---")
        st.markdown("### 📊 Hasil Analisis")

        if interpreter is None:

            st.warning(
                "⚠️ File model_coral_efficientnet.tflite tidak ditemukan."
            )

        else:

            with st.spinner("Menganalisis gambar..."):

                # =====================================
                # PREPROCESSING
                # =====================================
                img_resized = image.convert("RGB").resize((300, 300))

                img_array = np.array(img_resized)

                img_tensor = np.expand_dims(
                    img_array,
                    axis=0
                )

                img_tensor = preprocess_efficientnet(
                    img_tensor
                )

                # =====================================
                # INFERENSI TFLITE
                # =====================================
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                interpreter.set_tensor(
                    input_details[0]['index'],
                    img_tensor
                )

                interpreter.invoke()

                prediction = interpreter.get_tensor(
                    output_details[0]['index']
                )

                raw_score = float(prediction[0][0])

                # Debug score model
                st.info(
                    f"Raw Score Model : {raw_score:.4f}"
                )

                # =====================================
                # THRESHOLD
                # =====================================

                THRESHOLD = 0.40

                if raw_score < THRESHOLD:

                    hasil_prediksi = "Bleached Coral (Memutih/Sakit)"
                    score = (1 - raw_score) * 100

                    st.error(
                        f"### KONDISI KRITIS: {hasil_prediksi}"
                    )

                    st.markdown(
                        """
                        <div style="
                        background-color:#ffe6e6;
                        padding:15px;
                        border-radius:10px;
                        border-left:5px solid #ff4b4b;
                        color:#1e1e1e;
                        ">
                        <strong>Hasil Analisis:</strong>
                        Terumbu karang terdeteksi mengalami bleaching.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    hasil_prediksi = "Healthy Coral (Sehat)"
                    score = raw_score * 100

                    st.success(
                        f"### KONDISI AMAN: {hasil_prediksi}"
                    )

                    st.markdown(
                        """
                        <div style="
                        background-color:#e6f4ea;
                        padding:15px;
                        border-radius:10px;
                        border-left:5px solid #137333;
                        color:#1e1e1e;
                        ">
                        <strong>Hasil Analisis:</strong>
                        Terumbu karang dinilai sehat.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Status Klasifikasi",
                        "Selesai ✔"
                    )

                with col2:
                    st.metric(
                        "Confidence Score",
                        f"{score:.2f}%"
                    )

# ==========================================
# 7. FOOTER
# ==========================================
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#888888;font-size:0.85em;">
        <strong>Dibuat oleh Kelompok 5 - Teknik Informatika UMRAH</strong><br>
        Anggota: Syawal Rizal Utama | Meyza Zaharanie |
        Putri Ramadhanti | Zony Fatma Mulia |
        Tommy Susanto | Rusydi Ardani |
        Rani Nadia Sihombing | Luvita Septiana Putri
    </div>
    """,
    unsafe_allow_html=True
)