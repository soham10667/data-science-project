# 🦠 COVID-19 Data Analysis Dashboard

An interactive data analysis and visualization dashboard built with **Python**, **Pandas**, **Plotly**, and **Streamlit** for exploring global COVID-19 stats, trends, and regional metrics.

---

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Streamlit App**
   ```bash
   streamlit run app.py
   ```

---

## ✨ Key Features

- **Cascading & Interactive Filters**: Filter by Date Range, WHO Region, and Country without conflicting inputs.
- **Accurate KPI Summaries**: Cumulative & latest metrics for Confirmed, Active, Recovered cases, and Deaths.
- **Interactive Visualizations**: Time-series charts, 7-day moving averages, top country rankings, and WHO regional treemaps.
- **Data Export**: Inspect filtered data and export custom reports to CSV.

---

## 📁 Project Structure

```
corona/
├── app.py              # Main Streamlit dashboard application
├── requirements.txt    # Python package dependencies
├── data/               # COVID-19 CSV datasets
└── src/                # Data loading, cleaning, and analytics modules
    ├── data_loader.py
    ├── data_cleaner.py
    └── analyzer.py
```

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Data Analysis**: Pandas, NumPy
- **Data Visualization**: Plotly Express & Graph Objects
