import streamlit as st
import pandas as pd
import joblib

# 1. Memuat Model yang Telah Disimpan
# Pastikan file 'model_jantung.pkl' ada di folder yang sama
try:
    model = joblib.load('models/model_heart_disease.pkl')
except FileNotFoundError:
    st.error("Error: File model 'model_jantung.pkl' tidak ditemukan.")
    st.stop()

# Judul Aplikasi Web
st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="wide")
st.title("Aplikasi Web untuk Prediksi Penyakit Jantung")
st.write("Aplikasi ini menggunakan model Machine Learning (Random Forest) untuk memprediksi risiko penyakit jantung berdasarkan data input.")

# 2. Membuat Form Input di Sidebar
st.sidebar.header("Masukkan Data Pasien:")

# Kita akan kumpulkan input dalam sebuah fungsi agar rapi
def user_input_features():
    # Buat slider dan input box
    # (age) Usia
    age = st.sidebar.slider('Usia', 29, 77, 50) # min, max, default
    
    # (sex) Jenis Kelamin (1 = Pria, 0 = Wanita)
    sex = st.sidebar.selectbox('Jenis Kelamin', ('Pria', 'Wanita'))
    
    # (cp) Jenis Nyeri Dada
    cp = st.sidebar.selectbox('Jenis Nyeri Dada (cp)', (0, 1, 2, 3))
    
    # (trestbps) Tekanan Darah Istirahat
    trestbps = st.sidebar.slider('Tekanan Darah (trestbps)', 94, 200, 120)
    
    # (chol) Kolesterol Serum
    chol = st.sidebar.slider('Kolesterol (chol)', 126, 564, 240)
    
    # (fbs) Gula Darah Puasa > 120 mg/dl (1 = ya, 0 = tidak)
    fbs = st.sidebar.selectbox('Gula Darah Puasa > 120 mg/dl (fbs)', (0, 1))
    
    # (restecg) Hasil Elektrokardiografi Istirahat
    restecg = st.sidebar.selectbox('Hasil EKG Istirahat (restecg)', (0, 1, 2))
    
    # (thalach) Detak Jantung Maksimum
    thalach = st.sidebar.slider('Detak Jantung Maks (thalach)', 71, 202, 150)
    
    # (exang) Angina Akibat Latihan (1 = ya, 0 = tidak)
    exang = st.sidebar.selectbox('Angina Akibat Latihan (exang)', (0, 1))
    
    # (oldpeak) Depresi ST
    oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.2, 1.0)
    
    # (slope) Kemiringan Segmen ST
    slope = st.sidebar.selectbox('Slope', (0, 1, 2))
    
    # (ca) Jumlah Pembuluh Darah Utama
    ca = st.sidebar.selectbox('Jumlah Pembuluh Darah (ca)', (0, 1, 2, 3, 4))
    
    # (thal) Status Thalassemia
    thal = st.sidebar.selectbox('Thal', (0, 1, 2, 3))

    # Konversi input 'sex' ke format angka
    sex_numeric = 1 if sex == 'Pria' else 0

    # Kumpulkan data input ke dalam dictionary
    # PENTING: Nama keys harus SAMA PERSIS dengan nama kolom di 'heart.csv'
    data = {
        'age': age,
        'sex': sex_numeric,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    # Ubah dictionary menjadi DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Panggil fungsi untuk mendapatkan input user
input_df = user_input_features()

# 3. Menampilkan Data Input
st.subheader("Data Pasien yang Dimasukkan:")
st.write(input_df)

# 4. Tombol untuk Prediksi
if st.sidebar.button('Prediksi Risiko'):
    # Lakukan prediksi
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Tampilkan hasil
    st.subheader('Hasil Prediksi:')
    
    if prediction[0] == 1:
        st.error(f"**BERISIKO TINGGI** terkena penyakit jantung.")
        st.write(f"Probabilitas: {prediction_proba[0][1] * 100:.2f}%")
    else:
        st.success(f"**BERISIKO RENDAH** terkena penyakit jantung.")
        st.write(f"Probabilitas: {prediction_proba[0][0] * 100:.2f}%")
else:
    st.info("Klik tombol 'Prediksi Risiko' di sidebar untuk melihat hasilnya.")