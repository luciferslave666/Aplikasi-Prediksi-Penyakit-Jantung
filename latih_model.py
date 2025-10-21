import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Memuat data dari file CSV
try: 
    data = pd.read_csv('data/heart-disease-UCI.csv')
    print("Data berhasil dimuat.")
except FileNotFoundError:
    print("File tidak ditemukan. Pastikan path file sudah benar.")
    exit()

# Pisahkan fitur (x) dan target (y)
x = data.drop('target', axis=1)
y = data['target']

# Bagi data menjadi set pelatihan dan pengujian
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Data telah dibagi menjadi set pelatihan dan pengujian.")

# Melatih model
print("Memulai pelatihan model...")

# Memakai RandomForestClassifier sebagai model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Proses pelatihan
model.fit(x_train, y_train)

print("Model telah dilatih.")

# Evaluasi model
prediksi_test = model.predict(x_test)
akurasi = accuracy_score(y_test, prediksi_test)
print(f"Akurasi model pada data pengujian: {akurasi * 100:.2f}%")

# Simpan model yang telah dilatih
joblib.dump(model, 'model_heart_disease.pkl')
print("Model telah disimpan sebagai 'model_heart_disease.pkl'.")