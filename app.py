import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Prediksi Risiko Diabetes Berdasarkan Data Pasien")

st.write("""
Aplikasi ini digunakan untuk memprediksi apakah seorang pasien
berisiko menderita diabetes menggunakan tiga algoritma Machine Learning:

- K-Nearest Neighbor (KNN)
- Naive Bayes
- Decision Tree
""")

df = pd.read_csv("diabetes.csv")

st.subheader("Dataset Diabetes")

st.dataframe(df.head())

st.subheader("Informasi Dataset")

st.write("Jumlah Data :", df.shape[0])
st.write("Jumlah Kolom :", df.shape[1])

st.write(df.describe())

# ==========================
# PREPROCESSING DATA
# ==========================

# Memisahkan fitur dan target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Normalisasi data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.subheader("Preprocessing Data")

st.write("Jumlah Data Training :", X_train.shape[0])
st.write("Jumlah Data Testing :", X_test.shape[0])

# ==========================
# MODEL KNN
# ==========================

st.header("Model K-Nearest Neighbor (KNN)")

# Membuat model KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Melatih model
knn.fit(X_train, y_train)

# Prediksi data testing
y_pred_knn = knn.predict(X_test)

# Menghitung metrik evaluasi
acc = accuracy_score(y_test, y_pred_knn)
prec = precision_score(y_test, y_pred_knn)
rec = recall_score(y_test, y_pred_knn)
f1 = f1_score(y_test, y_pred_knn)

# Menampilkan hasil evaluasi
st.subheader("Hasil Evaluasi KNN")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{acc:.4f}")

with col2:
    st.metric("Precision", f"{prec:.4f}")

col3, col4 = st.columns(2)

with col3:
    st.metric("Recall", f"{rec:.4f}")

with col4:
    st.metric("F1-Score", f"{f1:.4f}")

    # ==========================
# CONFUSION MATRIX
# ==========================

st.subheader("Confusion Matrix KNN")

cm = confusion_matrix(y_test, y_pred_knn)

fig, ax = plt.subplots(figsize=(5,5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Tidak Diabetes", "Diabetes"]
)

disp.plot(ax=ax)

st.pyplot(fig)

st.subheader("Nilai Confusion Matrix")

cm_df = pd.DataFrame(
    cm,
    index=["Actual Tidak Diabetes", "Actual Diabetes"],
    columns=["Prediksi Tidak Diabetes", "Prediksi Diabetes"]
)

st.dataframe(cm_df)

# ==========================
# MODEL NAÏVE BAYES
# ==========================

st.header("Model Naïve Bayes")

# Membuat model Naïve Bayes
nb = GaussianNB()

# Melatih model
nb.fit(X_train, y_train)

# Melakukan prediksi
y_pred_nb = nb.predict(X_test)

# Menghitung metrik evaluasi
acc_nb = accuracy_score(y_test, y_pred_nb)
prec_nb = precision_score(y_test, y_pred_nb)
rec_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)

# Menampilkan hasil evaluasi
st.subheader("Hasil Evaluasi Naïve Bayes")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{acc_nb:.4f}")

with col2:
    st.metric("Precision", f"{prec_nb:.4f}")

col3, col4 = st.columns(2)

with col3:
    st.metric("Recall", f"{rec_nb:.4f}")

with col4:
    st.metric("F1-Score", f"{f1_nb:.4f}")

    # ==========================
# CONFUSION MATRIX NAÏVE BAYES
# ==========================

st.subheader("Confusion Matrix Naïve Bayes")

cm_nb = confusion_matrix(y_test, y_pred_nb)

fig, ax = plt.subplots(figsize=(5,5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_nb,
    display_labels=["Tidak Diabetes", "Diabetes"]
)

disp.plot(ax=ax)

st.pyplot(fig)

st.subheader("Nilai Confusion Matrix")

cm_nb_df = pd.DataFrame(
    cm_nb,
    index=["Actual Tidak Diabetes", "Actual Diabetes"],
    columns=["Prediksi Tidak Diabetes", "Prediksi Diabetes"]
)

st.dataframe(cm_nb_df)

# ==========================
# MODEL DECISION TREE
# ==========================

st.header("Model Decision Tree")

# Membuat model Decision Tree
dt = DecisionTreeClassifier(random_state=42)

# Melatih model
dt.fit(X_train, y_train)

# Melakukan prediksi
y_pred_dt = dt.predict(X_test)

# Menghitung metrik evaluasi
acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
rec_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

# Menampilkan hasil evaluasi
st.subheader("Hasil Evaluasi Decision Tree")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{acc_dt:.4f}")

with col2:
    st.metric("Precision", f"{prec_dt:.4f}")

col3, col4 = st.columns(2)

with col3:
    st.metric("Recall", f"{rec_dt:.4f}")

with col4:
    st.metric("F1-Score", f"{f1_dt:.4f}")

    # ==========================
# CONFUSION MATRIX DECISION TREE
# ==========================

st.subheader("Confusion Matrix Decision Tree")

cm_dt = confusion_matrix(y_test, y_pred_dt)

fig, ax = plt.subplots(figsize=(5,5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_dt,
    display_labels=["Tidak Diabetes", "Diabetes"]
)

disp.plot(ax=ax)

st.pyplot(fig)

st.subheader("Nilai Confusion Matrix")

cm_dt_df = pd.DataFrame(
    cm_dt,
    index=["Actual Tidak Diabetes", "Actual Diabetes"],
    columns=["Prediksi Tidak Diabetes", "Prediksi Diabetes"]
)

st.dataframe(cm_dt_df)

# ==========================
# HALAMAN PREDIKSI DIABETES
# ==========================

st.header("Prediksi Risiko Diabetes")

# Memilih model
pilih_model = st.selectbox(
    "Pilih Model",
    ("KNN", "Naïve Bayes", "Decision Tree")
)

st.subheader("Masukkan Data Pasien")

pregnancies = st.number_input("Pregnancies", min_value=0, value=1)
glucose = st.number_input("Glucose", min_value=0, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)
insulin = st.number_input("Insulin", min_value=0, value=79)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=1, value=30)

# Menyusun data input


data_baru = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
})

data_baru = scaler.transform(data_baru)

if st.button("Prediksi"):

    if pilih_model == "KNN":
        hasil = knn.predict(data_baru)

    elif pilih_model == "Naïve Bayes":
        hasil = nb.predict(data_baru)

    else:
        hasil = dt.predict(data_baru)

    if hasil[0] == 1:
        st.error("Pasien diprediksi MENGIDAP DIABETES")
    else:
        st.success("Pasien diprediksi TIDAK MENGIDAP DIABETES")

        # Membaca dataset gerai kopi

kopi = pd.read_csv("lokasi_gerai_kopi_clean.csv")
st.title("Analisis Klaster Lokasi Gerai Kopi dan Deteksi Zona Sepi")

st.subheader("Dataset Gerai Kopi")

st.dataframe(kopi.head())

st.subheader("Informasi Dataset")

st.write("Jumlah Data :", kopi.shape[0])
st.write("Jumlah Kolom :", kopi.shape[1])

st.write(kopi.describe())

# ====================================
# PREPROCESSING DATA
# ====================================

# PREPROCESSING DATA

X_cluster = kopi[[
    'x',
    'y',
    'population_density',
    'traffic_flow',
    'competitor_count',
    'is_commercial'
]]

st.subheader("Missing Value")

st.write(X_cluster.isnull().sum())

X_cluster = X_cluster.dropna()

scaler_cluster = StandardScaler()

X_scaled = scaler_cluster.fit_transform(X_cluster)

st.subheader("Data Setelah Normalisasi")

kopi = pd.read_csv("lokasi_gerai_kopi_clean.csv")

st.subheader("Nama Kolom Dataset")
st.write(list(kopi.columns))

st.dataframe(kopi.head())
# ====================================
# PREPROCESSING DATA
# ====================================

X_cluster = kopi[[
    'x',
    'y',
    'population_density',
    'traffic_flow',
    'competitor_count',
    'is_commercial'
]]

# Mengecek missing value
st.subheader("Missing Value")
st.write(X_cluster.isnull().sum())

# Menghapus missing value
X_cluster = X_cluster.dropna()

# Normalisasi data
scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(X_cluster)

st.subheader("Data Setelah Normalisasi")

st.dataframe(
    pd.DataFrame(
        X_scaled,
        columns=X_cluster.columns
    ).head()
)
    

    # =====================================
# MEMBUAT MODEL K-MEANS
# =====================================

st.header("Clustering Gerai Kopi Menggunakan K-Means")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster = kmeans.fit_predict(X_scaled)

kopi["Cluster"] = cluster

st.success("Clustering berhasil dilakukan")

st.dataframe(kopi.head())

# =====================================
# VISUALISASI CLUSTER
# =====================================

st.subheader("Visualisasi Cluster")

fig, ax = plt.subplots(figsize=(8,6))

scatter = ax.scatter(
    kopi["x"],
    kopi["y"],
    c=kopi["Cluster"],
    cmap="viridis"
)

plt.colorbar(scatter)

ax.set_xlabel("X")

ax.set_ylabel("Y")

ax.set_title("Hasil Clustering Gerai Kopi")

st.pyplot(fig)

# =====================================
# ANALISIS CLUSTER
# =====================================

st.subheader("Rata-rata Traffic Flow per Cluster")

rata = kopi.groupby("Cluster")["traffic_flow"].mean()

st.dataframe(rata)

cluster_ramai = rata.idxmax()

cluster_sepi = rata.idxmin()

st.success(f"Cluster {cluster_ramai} merupakan zona PALING RAMAI")

st.warning(f"Cluster {cluster_sepi} merupakan zona PALING SEPI")

st.header("Prediksi Cluster Lokasi Baru")

x = st.number_input("Koordinat X")

y = st.number_input("Koordinat Y")

population_density = st.number_input("Population Density")

traffic_flow = st.number_input("Traffic Flow")

competitor_count = st.number_input("Competitor Count")

is_commercial = st.selectbox(
    "Commercial Area",
    [0,1]
)

if st.button("Prediksi Cluster"):

    data = pd.DataFrame({
        "x": [x],
        "y": [y],
        "population_density": [population_density],
        "traffic_flow": [traffic_flow],
        "competitor_count": [competitor_count],
        "is_commercial": [is_commercial]
    })

    data = scaler_cluster.transform(data)

    hasil_cluster = kmeans.predict(data)

    st.success(f"Lokasi termasuk Cluster {hasil_cluster[0]}")

    if hasil_cluster[0] == cluster_ramai:
        st.success("Rekomendasi: Lokasi berada pada zona ramai, berpotensi baik untuk membuka gerai kopi.")

    elif hasil_cluster[0] == cluster_sepi:
        st.warning("Rekomendasi: Lokasi berada pada zona sepi, perlu pertimbangan sebelum membuka gerai kopi.")

    else:
        st.info("Rekomendasi: Lokasi berada pada zona dengan tingkat keramaian sedang.")

    if hasil_cluster[0] == cluster_ramai:
     st.success("Rekomendasi: Lokasi berada pada zona ramai, berpotensi baik untuk membuka gerai kopi.")

    elif hasil_cluster[0] == cluster_sepi:
     st.warning("Rekomendasi: Lokasi berada pada zona sepi, perlu pertimbangan sebelum membuka gerai kopi.")

else:
    st.info("Rekomendasi: Lokasi berada pada zona dengan tingkat keramaian sedang.")

