NAMA : TULUS AMRU
NIM : 23146002

Prediksi Diabetes dan Clustering Lokasi Gerai Kopi
Penjelasan Proyek

Proyek ini merupakan aplikasi berbasis Streamlit yang dibuat sebagai tugas Ujian Akhir Semester (UAS) Data Mining. Aplikasi ini menggabungkan dua studi kasus yang menerapkan teknik data mining, yaitu klasifikasi dan clustering.

Pada Bagian A, aplikasi melakukan prediksi risiko diabetes berdasarkan data pasien menggunakan tiga algoritma klasifikasi, yaitu K-Nearest Neighbor (KNN), Naïve Bayes, dan Decision Tree. Pengguna dapat memilih algoritma yang diinginkan, memasukkan data pasien seperti kadar glukosa, tekanan darah, BMI, usia, dan atribut lainnya, kemudian aplikasi akan memberikan hasil prediksi apakah pasien diperkirakan mengidap diabetes atau tidak mengidap diabetes. Selain itu, aplikasi juga menampilkan hasil evaluasi model berupa Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix.

Pada Bagian B, aplikasi menerapkan algoritma K-Means Clustering untuk mengelompokkan lokasi gerai kopi berdasarkan karakteristik lokasi, seperti koordinat (x dan y), kepadatan penduduk, arus lalu lintas, jumlah pesaing, dan status kawasan komersial. Hasil clustering divisualisasikan dalam bentuk grafik scatter plot berwarna, sehingga pengguna dapat melihat persebaran setiap cluster. Aplikasi juga menyediakan fitur prediksi cluster untuk lokasi baru berdasarkan data yang dimasukkan pengguna.

Proyek ini bertujuan untuk menunjukkan penerapan teknik data mining dalam menyelesaikan permasalahan klasifikasi dan clustering menggunakan Python serta Streamlit sebagai media visualisasi dan implementasi aplikasi berbasis web.

Instruksi Menjalankan Aplikasi

1. Persiapan

Pastikan Python 3.10 atau versi yang lebih baru telah terinstal pada komputer.

2. Install Library

Buka Terminal atau Command Prompt, kemudian install seluruh library yang dibutuhkan.

pip install streamlit pandas numpy matplotlib scikit-learn

Atau jika menggunakan file requirements.txt, jalankan:

pip install -r requirements.txt 3. Struktur Folder

Pastikan struktur folder proyek seperti berikut:

UAS-DataMining/
│
├── app.py
├── diabetes.csv
├── lokasi_gerai_kopi_clean.csv
├── requirements.txt
└── README.md 4. Menjalankan Aplikasi

Buka Terminal atau Command Prompt, kemudian masuk ke folder proyek.

cd "C:\Users\Admin\Documents\TULUS AMRU UAS MAINING"

Jalankan aplikasi menggunakan Streamlit.

streamlit run app.py

Apabila perintah di atas tidak dapat dijalankan, gunakan:

python -m streamlit run app.py 5. Membuka Aplikasi

Setelah aplikasi berhasil dijalankan, browser akan terbuka secara otomatis.

Apabila tidak terbuka, akses melalui alamat berikut:

http://localhost:8501 6. Menggunakan Aplikasi
Bagian A – Prediksi Diabetes
Pilih model klasifikasi (KNN, Naïve Bayes, atau Decision Tree).
Masukkan data pasien.
Klik tombol Prediksi.
Sistem akan menampilkan hasil prediksi serta metrik evaluasi model.
Bagian B – Clustering Gerai Kopi
Masukkan data lokasi gerai.
Klik tombol Prediksi Cluster.
Sistem akan menampilkan hasil cluster beserta rekomendasi lokasi.
