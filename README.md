# 🗑️ Garbage App

**Garbage App** adalah aplikasi website berbasis **Artificial Intelligence (AI)** yang mampu mendeteksi **12 kategori sampah** dari gambar. Aplikasi ini menggunakan model klasifikasi citra yang dilatih dengan TensorFlow dan dijalankan menggunakan Flask.

---

## 📁 Struktur Folder

```

garbage-app/
├── app.py                  # File utama untuk menjalankan server Flask
├── run.py                  # Alternatif file untuk menjalankan server
├── garbage\_model.tflite    # Model TFLite untuk inferensi AI
│
├── models/                 # Model hasil training
│   ├── garbage\_model.keras
│   └── labels.txt
│
├── templates/              # Folder template HTML Flask
│   └── index.html
│
├── uploads/                # Folder sementara untuk menyimpan gambar yang diunggah
│
├── utils/                  # Modul utilitas untuk image processing dan prediksi
│   ├── **init**.py
│   ├── image\_processing.py
│   └── model\_utils.py
│
├── venv310/                # Virtual environment Python (jangan diunggah ke GitHub)
├── **pycache**/            # Cache Python (abaikan)

````

---

## 🧠 Kategori Sampah yang Didukung
1. Sepatu  
2. Sampah Umum  
3. Kaca Putih  
4. Kaca Coklat  
5. Baterai  
6. Organik  
7. Kardus  
8. Pakaian  
9. Kaca Hijau  
10. Logam  
11. Kertas  
12. Plastik

---

## 📦 Dataset
Dataset yang digunakan dilatih dari Kaggle:  
🔗 [Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification/data)

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Install Python 3.10
Unduh dari: [https://www.python.org/downloads/release/python-3109/](https://www.python.org/downloads/release/python-3109/)  
✔️ Pastikan mencentang **"Add Python to PATH"** saat instalasi.

---

### 2. Setup Virtual Environment
```
cd C:\Garbage-Detection\garbage-app

# Buat virtual environment
C:\Python310\python.exe -m venv venv310

# Aktifkan environment
venv310\Scripts\activate
````

---

### 3. Install Dependensi

```
pip install --upgrade pip atau python.exe -m pip install --upgrade pip
pip install tensorflow flask pillow flask-cors
```

---

### 4. Jalankan Aplikasi

```
python app.py
```

Jika berhasil, akan muncul:

```
TFLite model loaded successfully
Running on http://127.0.0.1:5000
```

Buka browser dan akses:
🔗 `http://127.0.0.1:5000` atau IP lokal lain seperti `http://192.168.x.x:5000`

---

## 📷 Cara Menggunakan

1. Pilih gambar sampah (format: JPG, PNG, JPEG, GIF)
2. Klik **Upload**
3. Setelah gambar tampil, klik **Analisis Gambar**
4. Hasil deteksi ditampilkan dalam bentuk:

   * Ringkasan (jumlah total, yang dapat didaur ulang, dan tidak)
   * Detil kategori sampah dan status daur ulang
   * Akurasi dan saran pembuangan

---

## 💡 Tips Penggunaan

* Gunakan gambar dengan **pencahayaan yang baik**
* Fokuskan kamera pada objek sampah
* Hindari gambar **blur** atau terlalu gelap
* Satu gambar bisa mendeteksi **lebih dari satu jenis sampah**

---

## 📝 Lisensi

Proyek ini bersifat open-source dan dikembangkan untuk edukasi dan demonstrasi teknologi AI.

---

### 👤 Disusun oleh:
Atifa Ismi Nawla (3.34.23.1.06)  
Desyana Dewi Hapsari (3.34.23.1.08)  
Kelas IK-2B

**Tugas Kecerdasan Buatan**  
Program Studi D3 Teknik Informatika  
Jurusan Teknik Elektro  
Politeknik Negeri Semarang  
Tahun Ajaran 2024/2025

---
