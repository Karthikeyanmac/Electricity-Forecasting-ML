# ⚡ Electricity Demand Forecasting & Energy Intelligence Dashboard

An end-to-end Machine Learning project for analyzing historical electricity consumption and forecasting future demand using a Random Forest Regressor.

The project also features an interactive, dark-themed dashboard built with Streamlit and Plotly to visualize historical trends, model performance, and predicted electricity demand.

## 🚀 Project Overview

Electricity demand forecasting helps understand consumption patterns and estimate future energy requirements.

This project uses historical PJM electricity consumption data to:
- Analyze electricity consumption patterns.
- Perform feature engineering using historical demand.
- Train and evaluate machine learning models.
- Generate recursive 24-observation forecasts.
- Visualize results through an interactive dashboard.

## ✨ Key Features

- Historical electricity consumption analysis
- Time-based feature engineering
- Lag features and rolling mean calculations
- Baseline and Improved Random Forest model comparison
- Recursive 24-observation forecasting
- Interactive Streamlit dashboard
- Plotly-based charts and visualizations
- Forecast insights and CSV export

## 🧠 Machine Learning Model

**Algorithm:** Random Forest Regressor

Two model configurations were evaluated:

1. **Baseline Model:** Uses calendar-based features.
2. **Improved Model:** Uses calendar features along with historical lag features and rolling averages.

### Model Performance

| Evaluation Metric | Baseline | Improved |
|---|---:|---:|
| MAE | 3276.09 | 328.57 |
| RMSE | 4434.92 | 452.96 |
| R² Score | 0.5333 | 0.9951 |

The improved model achieved a substantial reduction in prediction error by incorporating historical electricity consumption patterns.

*Note: R² is a variance-explained metric, not prediction accuracy. The evaluation uses a chronological test split with historical lag values.*

## 🛠️ Technology Stack

- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit
- **Development:** Jupyter Notebook, VS Code

## 📂 Project Structure

```text
Electricity_Forecasting/
│
├── app.py
├── README.md
├── model_comparison.csv
├── future_forecast.csv
│
├── dataset/
│   └── PJME_hourly.csv
│
├── pipeline/
│   └── electricity_forecasting.ipynb
│
└── requirements.txt
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd Electricity_Forecasting
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the dashboard

```bash
streamlit run app.py
```

Open the local address provided by Streamlit, usually:

```text
http://localhost:8501
```

## 📊 Dashboard Modules

1. **Dashboard Overview** – Key performance indicators and consumption insights.
2. **Historical Analytics** – Historical electricity demand patterns.
3. **Model Performance** – Baseline versus improved model evaluation.
4. **Forecast Analytics** – Predicted electricity demand and forecast insights.

## 📈 Forecasting Output

The project generates a recursive forecast for the next 24 hourly observations following the final timestamp in the dataset.

The available PJM dataset ends on August 3, 2018. Therefore, the forecast is a historical-data demonstration and is not a live electricity demand prediction.

## 📚 Dataset

**PJM Hourly Energy Consumption Dataset**

Source: Kaggle – PJM Hourly Energy Consumption

Dataset: `PJME_hourly.csv`

## 🔮 Future Enhancements

- Compare additional forecasting algorithms.
- Add weather and temperature information.
- Implement real-time data integration.
- Improve multi-step forecasting strategies.
- Deploy the dashboard online.

## 👨‍💻 Author

**Karthikeyan S.**

Interested in Machine Learning, Artificial Intelligence, Embedded Systems and Software Development.

## ⭐ Acknowledgement

This project was developed as a hands-on learning experience in Machine Learning, time-series analysis, forecasting and interactive dashboard development.