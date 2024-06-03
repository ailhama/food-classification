# Import libraries
from dotenv import load_dotenv
import os
import google.generativeai as genai
from PIL import Image
import streamlit as st
import tensorflow as tf
import numpy as np

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load model CNN
model_path = "./my_model"
cnn_model = tf.keras.models.load_model(model_path)

# Fungsi untuk memuat model Gemini AI dan mendapatkan respons
def get_response_nutrition(image, prompt):
    try:
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content([image[0], prompt])
        return response.text
    except Exception as e:
        st.error(f"Error during API call: {e}")
        return None

# Preprocess data gambar
def prep_image(uploaded_file):
    if uploaded_file is not None:
        # Read the file as bytes
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No File is uploaded!")

# Preprocess gambar untuk model CNN
def prep_image_cnn(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Konfigurasi Streamlit App
st.header("Kalkulator Nutrisi dan Klasifikasi Makanan")

# Upload file gambar
upload_file = st.file_uploader("Pilih Gambar...", type=["jpg", "jpeg", "png"])
if upload_file is not None:
    # Menampilkan gambar yang diupload
    max_width = 200
    max_height = 200
    image = Image.open(upload_file)
    image.thumbnail((max_width, max_height), Image.LANCZOS)
    st.image(image, caption="Gambar berhasil diunggah")

    # Menyiapkan gambar untuk model CNN
    image_for_cnn = prep_image_cnn(image)

    # Mengklasifikasikan gambar menggunakan model CNN
    predictions = cnn_model.predict(image_for_cnn)
    class_idx = np.argmax(predictions, axis=1)[0]
    class_labels = ['Ayam Goreng', 'Bakso', 'Bubur Ayam', 'Ikan Lele Goreng', 'Mi Goreng', 'Nasi', 'Sate', 'Soto', 'Telur dadar', 'Telur mata sapi', 'Ikan mujahir goreng', 'Lontong', 'Pempek telur', 'Singkong Goreng', 'Tempe kedelai murni, goreng']
    class_label = class_labels[class_idx]
    st.write(f"Klasifikasi Makanan: {class_label}")

    # Menyiapkan gambar untuk integrasi dengan model Gemini AI
    image_data = prep_image(upload_file)

    # Prompt Template
    input_prompt_nutrition = f"""
    Anda adalah seorang Ahli Gizi yang ahli. Sebagai ahli gizi yang terampil, Anda diharuskan untuk menganalisis makanan dalam gambar dan menentukan nilai gizi total.
    Gambar ini memperlihatkan {class_label}.
    Silakan berikan rincian dari jenis makanan yang ada dalam {class_label} beserta kandungan gizinya.
    Makanan, Ukuran porsi, Kalori, Protein (g), Lemak,
    Karbohidrat (g), Serat (g)
    Tolong nilai dalam hal akurasi dan presisi pada nutrisi yang Anda prediksi.
    Tampilkan berapa akurasi dan presisinya.
    Gunakan tabel untuk menunjukkan informasi di atas.
    """

    # Respon hasil nutrisi
    submit = st.button("Menghitung Nilai Nutrisi!")
    if submit:
        response = get_response_nutrition(image_data, input_prompt_nutrition)
        if response:
            st.subheader("Nutrisi AI:")
            st.write(response)
