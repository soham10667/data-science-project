# 🦠 COVID-19 Data Analysis & Interactive Dashboard

A comprehensive, fully dynamic COVID-19 Data Analysis Dashboard application built with **Python**, **Pandas**, **NumPy**, **Plotly**, and **Streamlit**.

---

## 📌 Features & Improvements

### 🎛️ Cascading & Conflict-Free Filters
- **Dependent Filters**: Selecting a WHO Region (e.g., *Europe*) automatically restricts the Country/Region options to only countries within that region, eliminating conflicting states.
- **"Reset All Filters" Button**: One-click reset in sidebar restores Date Range, WHO Region, Country selections, and Search query to default global state.
- **Active Filter Banner**: Displays real-time summary of currently active filter criteria above KPI cards.
- **Dynamic Date Range & Presets**: Auto-detects minimum and maximum dates directly from dataset (`All Time`, `2020`, `2021`, `Custom Range`).

### 📊 Accurate Cumulative KPI Calculations
- **Non-Duplicate Cumulative Sums**: Extracts the latest record per location within the selected date range to prevent multiplying cumulative numbers across dates.
- **KPI Summary Cards**:
  - Total Confirmed Cases
  - Active Cases ($\text{Active} = \text{Confirmed} - \text{Deaths} - \text{Recovered}$, clipped $\ge 0$)
  - Total Recoveries & Recovery Rate (%)
  - Total Deaths & Mortality Rate (%)

### 📈 Interactive Plotly Visualizations
1. **Cumulative Time-Series Line Chart**: Confirmed, Active, Recovered, and Deaths over time (supports logarithmic axis toggle).
2. **Daily Increments & Moving Averages**: Daily new cases and daily deaths with 7-day rolling moving averages (`New_Cases_7d_MA` and `Daily_Deaths_7d_MA`).
3. **Country Comparison Bar Chart**: Top N countries rankable by Confirmed, Deaths, Recovered, or Active cases.
4. **WHO Regional Treemap**: Hierarchical case breakdown by global WHO health regions.
5. **Case Breakdown Donut Chart**: Proportion of active cases, recoveries, and deaths.
6. **Recovery vs Mortality Rate Trajectory**: Rate trend comparisons over time.

### 📋 Data Inspector & Export
- Filtered dataset inspector with record count indicator.
- Empty filter warning (`⚠️ No records match current filters`) prevents dashboard crashes.
- One-click CSV report exporter (`covid_19_filtered_analysis.csv`).

---

## 📁 Project Structure

```
corona/
├── data/
│   ├── covid_19_clean_complete.csv     # Granular date-wise dataset by country & region
│   ├── country_wise_latest.csv          # Latest country snapshot metrics
│   └── day_wise.csv                     # Daily global aggregated dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Data importing & online fallback handlers
│   ├── data_cleaner.py                  # Schema validation, missing values & column normalization
│   └── analyzer.py                      # Accurate cumulative metrics & time-series calculations
├── app.py                               # Main Streamlit interactive dashboard application
├── requirements.txt                     # Dependencies
└── README.md                            # Project documentation
```

---

## 🚀 How to Run

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Launch Streamlit app
streamlit run app.py
```
