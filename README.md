# 📊 Superstore Profit Prediction

Aplikasi Machine Learning untuk memprediksi profit penjualan menggunakan data Superstore dengan model CatBoost.

## 🎯 Objective

- Analisis pola penjualan dan faktor yang mempengaruhi profit
- Membangun model prediksi profit dengan akurasi tinggi
- Dashboard interaktif untuk visualisasi dan prediksi real-time

## 📁 Struktur Project

```
.
├── data/
│   ├── raw/
│   │   └── Superstore.csv                # Dataset asli
│   ├── Superstore.csv                    # Dataset dengan features
│   ├── X_train.csv, X_test.csv           # Training/test features
│   └── y_train.csv, y_test.csv           # Training/test target
├── models/
│   ├── profit_prediction_model.pkl       # Model CatBoost Original
│   ├── label_encoders.pkl                # Encoders untuk kategori
│   ├── feature_names.pkl                 # Nama fitur
│   ├── model_metrics.pkl                 # Metrics (optional)
│   └── backup/                           # Semua versi model
├── notebook/
│   ├── 01_eda.ipynb                      # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb            # Data Preprocessing
│   └── 03_modeling.ipynb                 # Model Training
├── app.py                                # Streamlit Dashboard
├── requirements.txt                       # Dependencies
└── README.md                             # Dokumentasi
```

## 🔧 Instalasi

```bash
pip install -r requirements.txt
```

## 📊 Cara Menjalankan

### 1. Training Model (Optional)

Jalankan notebook secara berurutan:
```bash
01_eda.ipynb           # Analisis data
02_preprocessing.ipynb # Preprocessing & feature engineering
03_modeling.ipynb      # Training 4 models
```

### 2. Jalankan Dashboard

```bash
streamlit run app.py
```

Dashboard memiliki 3 halaman:
- **📈 Dashboard Overview**: Business metrics dan visualisasi
- **🔮 Profit Prediction**: Prediksi profit untuk order baru
- **📊 Data Explorer**: Filter dan eksplorasi data

## 🤖 Model

### Model Terpilih: CatBoost Original
- **R² Score**: ~85-90%
- **RMSE**: Low
- **Parameters**: 
  - iterations: 100
  - learning_rate: 0.05
  - depth: 4
  - l2_leaf_reg: 3.0

### 4 Model yang Dilatih:
1. XGBoost Original
2. XGBoost Improved
3. CatBoost Original ⭐
4. CatBoost Improved

### Fitur Penting (High Impact):
- **Sales** - Faktor terpenting
- **Discount** - Impact negatif terhadap profit
- **Quantity** - Jumlah barang
- **Category/Sub-Category** - Jenis produk
- **Segment/Region** - Customer info

### Fitur dengan Impact Rendah (<2%):
- Year, Month, Quarter, Day of Week, Delivery Days

## 💡 Input Prediksi

Form input hanya membutuhkan **7 field penting**:

**Product Info:**
- Category
- Sub-Category

**Customer Info:**
- Segment
- Region

**Transaction:**
- Sales Amount ($)
- Quantity
- Discount (0.0 - 0.8)

## 📊 Contoh Prediksi

**High Profit:**
- Technology + Phones
- Sales: $1000, Qty: 5, Discount: 0.1
- Result: Profit tinggi

**Low Profit/Loss:**
- Furniture + Tables
- Sales: $800, Qty: 2, Discount: 0.4
- Result: Berpotensi rugi

## 🛠️ Tech Stack

- Python 3.11+
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn
- XGBoost, CatBoost
- Streamlit
- Pickle

## 📁 File Requirements

File wajib untuk deployment:
```
models/profit_prediction_model.pkl
models/label_encoders.pkl
models/feature_names.pkl
data/raw/Superstore.csv
```

File optional:
```
models/model_metrics.pkl
```

## 🚀 Deployment

### Local:
```bash
streamlit run app.py
```

### Cloud (Streamlit Cloud):
1. Push ke GitHub
2. Deploy via streamlit.io
3. Pastikan semua file .pkl tersedia

---

**Note**: Model sudah trained dan siap digunakan. Tidak perlu menjalankan notebook kecuali ingin retrain model.
