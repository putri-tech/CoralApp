import streamlit as st
import numpy as np
from PIL import Image
import os
import tflite_runtime.interpreter as tflite

tflite = tf.lite

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
@st.cache_resource
def load_tflite_model():

    model_path = "model_coral_efficientnet.tflite"

    if not os.path.exists(model_path):
        return None

    interpreter = tflite.Interpreter(
        model_path=model_path
    )

    interpreter.allocate_tensors()

    return interpreter


interpreter = load_tflite_model()

# ==========================================
# 3. PREPROCESSING
# ==========================================
def preprocess_efficientnet(img_array):

    img_array = img_array.astype(np.float32)

    # setara preprocess_input EfficientNet
    img_array = img_array / 127.5 - 1.0

    return img_array

# ==========================================
# 4. HEADER
# ==========================================
st.title("🪸 Aplikasi Deteksi Dini Pemutihan Terumbu Karang")

st.subheader(
    "Metode Convolutional Neural Network (EfficientNetB0) Berbasis Web"
)

st.caption(
    "Proyek Tugas Besar Mata Kuliah Pengolahan Citra Digital — Teknik Informatika UMRAH"
)

st.markdown("---")

# ==========================================
# 5. INPUT GAMBAR
# ==========================================
st.markdown("### 📸 Pilih Metode Input Citra")

tab1, tab2 = st.tabs([
    "📁 Unggah Berkas Gambar",
    "📷 Ambil Foto via Kamera"
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
        "Ambil gambar menggunakan kamera"
    )

    if camera_input is not None:
        uploaded_file = camera_input

# ==========================================
# 6. PREDIKSI
# ==========================================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Citra Terumbu Karang",
        use_container_width=True
    )

    st.success("✔ Berkas citra berhasil dimuat")

    if st.button(
        "Jalankan Klasifikasi Citra",
        type="primary"
    ):

        if interpreter is None:

            st.error(
                "❌ File model_coral_efficientnet.tflite tidak ditemukan."
            )

        else:

            try:

                with st.spinner("Menganalisis gambar..."):

                    # Resize sesuai model training
                    img_resized = image.convert("RGB").resize((300, 300))

                    img_array = np.array(img_resized)

                    img_tensor = np.expand_dims(
                        img_array,
                        axis=0
                    )

                    img_tensor = preprocess_efficientnet(
                        img_tensor
                    )

                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()

                    interpreter.set_tensor(
                        input_details[0]["index"],
                        img_tensor
                    )

                    interpreter.invoke()

                    prediction = interpreter.get_tensor(
                        output_details[0]["index"]
                    )

                    raw_score = float(prediction[0][0])

                    st.info(
                        f"Raw Score Model : {raw_score:.4f}"
                    )

                    # Threshold
                    THRESHOLD = 0.40

                    if raw_score < THRESHOLD:

                        hasil = "Bleached Coral"
                        confidence = (1 - raw_score) * 100

                        st.error(
                            "### 🚨 KONDISI KRITIS: Bleached Coral"
                        )

                        st.markdown(
                            """
                            Terumbu karang terdeteksi mengalami
                            **coral bleaching (pemutihan)**.
                            """
                        )

                    else:

                        hasil = "Healthy Coral"
                        confidence = raw_score * 100

                        st.success(
                            "### ✅ KONDISI AMAN: Healthy Coral"
                        )

                        st.markdown(
                            """
                            Terumbu karang terdeteksi dalam
                            kondisi **sehat**.
                            """
                        )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Hasil Klasifikasi",
                            hasil
                        )

                    with col2:
                        st.metric(
                            "Confidence",
                            f"{confidence:.2f}%"
                        )

            except Exception as e:

                st.error(
                    f"Terjadi kesalahan saat inferensi: {e}"
                )

# ==========================================
# 7. FOOTER
# ==========================================
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#888888;font-size:0.85em;">
        <strong>Dibuat oleh Kelompok 5 - Teknik Informatika UMRAH</strong>
    </div>
    """,
    unsafe_allow_html=True
)
