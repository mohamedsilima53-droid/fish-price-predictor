# 🐟 Tanzania Fish Price Prediction
### Machine Learning Project — February 2026

A machine learning web application that predicts fish market prices (in Tanzanian Shillings) based on fish characteristics, quality, and market conditions. Built with Scikit-learn and deployed with Streamlit.

---

## 📁 Project Structure

```
├── ml_project.ipynb           # Jupyter Notebook (full ML pipeline)
├── app.py                     # Streamlit web application
├── model.pkl                  # Trained Decision Tree model
├── scaler.pkl                 # StandardScaler (for Linear Regression)
├── encoders.pkl               # Label encoders for categorical features
├── tanzania_fish_prices.csv   # Dataset (500 records)
├── project_report.docx        # Project report (2 pages)
└── README.md                  # This file
```

---

## 🎯 Objective

To design, train, evaluate, and deploy a machine learning model that predicts fish prices in the Tanzanian market based on:
- Fish species, quality grade, and weight
- Region, season, and market type
- Freshness, distance to market, and quantity

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Records | 500 rows |
| Features | 9 input features |
| Target | `price_tzs` (Fish price in TZS) |
| Source | Synthetic — generated to reflect real Tanzanian fish market patterns |

**Features:**

| Feature | Type | Values |
|---------|------|--------|
| `species` | Categorical | Nile Perch, Tilapia, Dagaa, Catfish, Sardine |
| `region` | Categorical | Dar es Salaam, Mwanza, Zanzibar, Mtwara, Tanga, Dodoma, Arusha |
| `season` | Categorical | Rainy, Dry |
| `market_type` | Categorical | Local Market, Export, Wholesale |
| `quality_grade` | Categorical | Grade A, Grade B, Grade C |
| `weight_kg` | Numeric | 0.5 – 15.0 kg |
| `freshness_days` | Numeric | 0 – 7 days |
| `distance_to_market_km` | Numeric | 5 – 300 km |
| `quantity_kg` | Numeric | 10 – 500 kg |

---

## 🤖 Models Trained

| Model | MAE (TZS) | RMSE (TZS) | R² Score | CV R² |
|-------|-----------|------------|----------|-------|
| Linear Regression | ~3,101 | ~4,042 | 0.4678 | 0.4263 |
| **Decision Tree** ✅ | **~2,664** | **~3,523** | **0.5958** | **0.5884** |

> **Decision Tree Regressor** was selected as the best model due to its higher R² score and lower error across all metrics.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/tanzania-fish-price-prediction.git
cd tanzania-fish-price-prediction
```

### 2. Install dependencies
```bash
pip install streamlit scikit-learn pandas numpy joblib
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push all project files to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — your app will be live in minutes!

---

## 📓 Jupyter Notebook Contents (`ml_project.ipynb`)

1. Import Libraries
2. Load & Explore Dataset
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing (encoding, scaling, train/test split)
5. Linear Regression — Training & Evaluation
6. Decision Tree — Training & Evaluation
7. Model Comparison Visualizations
8. Feature Importance Analysis
9. Best Model Selection & Saving

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Programming language |
| Scikit-learn | ML model training & evaluation |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Data visualization |
| Joblib | Model serialization |
| Streamlit | Web application & deployment |

---

## 👥 Group Members

| # | Name | Student ID | Contribution |
|---|------|------------|--------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

---

## 📅 Project Timeline

| Milestone | Date |
|-----------|------|
| Dataset creation & EDA | Week 1 |
| Model training & evaluation | Week 1 |
| Streamlit app development | Week 2 |
| Deployment | Week 2 |
| Report writing | Week 2 |
| **Submission deadline** | **16 February 2026** |
| **Project presentation** | **18 February 2026** |

---

## 📄 License

This project was developed for academic purposes as part of a Machine Learning course, 2026.
