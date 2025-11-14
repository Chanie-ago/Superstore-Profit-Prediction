# 📊 Superstore Profit Prediction

**Live Demo:** (https://superstore-profit-prediction.streamlit.app/)

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
│   ├── process/
│   │   └── superstore_with_features.csv  # Dataset dengan features
│   └── split/
│       ├── X_train.csv                   # Training features
│       ├── X_test.csv                    # Test features
│       ├── y_train.csv                   # Training target
│       └── y_test.csv                    # Test target
├── models/
│   ├── profit_prediction_model.pkl       # Model CatBoost Original (deployed)
│   ├── label_encoders.pkl                # Encoders untuk kategori
│   ├── feature_names.pkl                 # Nama fitur
│   └── backup/                           # Backup semua model
│       ├── catboost_original.pkl
│       ├── catboost_improved.pkl
│       ├── xgboost_original.pkl
│       └── xgboost_improved.pkl
├── notebook/
│   ├── 01_eda.ipynb                      # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb            # Data Preprocessing
│   └── 03_modeling.ipynb                 # Model Training & Comparison
├── app.py                                # Streamlit Dashboard
├── requirements.txt                      # Dependencies
├── .gitignore                            # Git ignore rules
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

## 🤖 Model Machine Learning

### Model Terpilih: CatBoost Original ⭐
- **R² Score**: ~85-90%
- **RMSE**: Low error rate
- **MAE**: Minimal absolute error
- **Overfitting**: <10% (excellent generalization)

**Hyperparameters:**
- iterations: 100
- learning_rate: 0.05
- depth: 4
- l2_leaf_reg: 3.0
- subsample: 0.7

### Model Comparison

4 model yang dilatih dan dibandingkan:

| Model | R² Score | RMSE | MAE | Status |
|-------|----------|------|-----|--------|
| CatBoost Original | ~88% | Low | Low | ✅ Deployed |
| CatBoost Improved | ~87% | Low | Low | Backup |
| XGBoost Original | ~85% | Medium | Medium | Backup |
| XGBoost Improved | ~86% | Medium | Medium | Backup |

### Feature Importance

**High Impact Features (>5%):**
- **Sales** (~40-50%) - Faktor terpenting
- **Discount** (~20-30%) - Impact negatif terhadap profit
- **Quantity** (~10-15%) - Jumlah barang terjual
- **Category/Sub-Category** (~5-10%) - Jenis produk
- **Segment/Region** (~2-5%) - Informasi customer

**Low Impact Features (<2%):**
- Year, Month, Quarter, Day of Week, Delivery Days
- *Features ini menggunakan nilai default pada prediksi*

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

## 📊 Contoh Input & Prediksi

### Contoh 1: High Profit ✅
```
Product:    Technology → Phones
Customer:   Consumer, East Region
Transaction: Sales $1000, Qty 5, Discount 10%
Result:     Profit tinggi (~$250-300)
```

### Contoh 2: Low Profit/Loss ⚠️
```
Product:    Furniture → Tables  
Customer:   Home Office, Central Region
Transaction: Sales $800, Qty 2, Discount 40%
Result:     Berpotensi rugi (~-$50 to $0)
```

### Contoh 3: Medium Profit 📊
```
Product:    Office Supplies → Paper
Customer:   Corporate, West Region
Transaction: Sales $500, Qty 10, Discount 15%
Result:     Profit sedang (~$100-150)
```

## 📈 Key Insights

Dari analisis dan modeling, ditemukan:

1. **Sales adalah predictor terkuat** - Menjelaskan 40-50% variasi profit
2. **Discount berpengaruh negatif** - Semakin besar diskon, semakin kecil profit
3. **Technology kategori terbaik** - Produk technology memiliki margin profit tertinggi
4. **Furniture berisiko tinggi** - Dengan diskon besar, furniture sering mengalami loss
5. **Temporal features tidak signifikan** - Waktu order tidak terlalu mempengaruhi profit

## 🛠️ Tech Stack

**Programming & Libraries:**
- Python 3.11+
- Pandas - Data manipulation
- NumPy - Numerical computing
- Matplotlib & Seaborn - Static visualization
- Plotly - Interactive visualization

**Machine Learning:**
- Scikit-learn - Preprocessing & metrics
- XGBoost - Gradient boosting
- CatBoost - Gradient boosting (model utama)

**Web Application:**
- Streamlit - Dashboard framework
- Pickle - Model serialization

## 📦 Requirements

```txt
pandas==2.1.3
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0
scikit-learn==1.3.2
xgboost==2.0.3
catboost==1.2.2
streamlit==1.29.0
```
## 🎯 Cara Menggunakan Aplikasi

### Halaman 1: Dashboard Overview 📈
- Lihat total sales, profit, dan jumlah order
- Visualisasi sales by category dan region
- Trend penjualan bulanan
- Top 10 products dan customers

### Halaman 2: Profit Prediction 🔮
1. Pilih **Category** dan **Sub-Category** produk
2. Pilih **Segment** dan **Region** customer
3. Input **Sales Amount**, **Quantity**, dan **Discount**
4. Klik **"🔮 Predict Profit"**
5. Lihat hasil prediksi dan profit margin

### Halaman 3: Data Explorer 📊
- Filter data by category dan region
- Lihat data table lengkap
- Download filtered data sebagai CSV
- Statistical summary

## 📊 Dataset Information

**Source:** Superstore Sales Dataset

**Details:**
- 9,994 baris transaksi
- Periode: 2014-2017
- 21 kolom original
- 17 engineered features untuk modeling

**Original Features:**
- Order info: Order ID, Order Date, Ship Date, Ship Mode
- Customer: Customer ID, Name, Segment
- Location: Country, City, State, Postal Code, Region
- Product: Product ID, Category, Sub-Category, Product Name
- Metrics: Sales, Quantity, Discount, Profit
