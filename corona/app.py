import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from src.data_loader import load_raw_dataset
from src.data_cleaner import clean_covid_dataset, validate_dataset_schema
from src.analyzer import (
    filter_dataframe,
    compute_kpi_metrics,
    compute_time_series_daily,
    compute_country_comparison,
    compute_who_region_breakdown
)

# Page Configuration
st.set_page_config(
    page_title="COVID-19 Global Data Analysis Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI, KPI Cards, Spacing & Fitting
st.markdown("""
<style>
    /* Global Container Padding */
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Header Gradient Banner */
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        padding: 20px 24px;
        border-radius: 14px;
        margin-bottom: 18px;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .header-title {
        color: #f8fafc;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 4px;
        margin-bottom: 0;
    }
    
    /* Active Filter Banner */
    .filter-banner {
        background: #1e293b;
        border: 1px solid #3b82f6;
        border-radius: 10px;
        padding: 10px 16px;
        color: #93c5fd;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: left;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        margin-bottom: 4px;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* KPI Card Colors */
    .card-confirmed .kpi-value { color: #38bdf8; }
    .card-confirmed { border-left: 4px solid #38bdf8; }
    
    .card-active .kpi-value { color: #fbbf24; }
    .card-active { border-left: 4px solid #fbbf24; }
    
    .card-recovered .kpi-value { color: #4ade80; }
    .card-recovered { border-left: 4px solid #4ade80; }
    
    .card-deaths .kpi-value { color: #f87171; }
    .card-deaths { border-left: 4px solid #f87171; }

    /* Sidebar Spacing */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 18px;
        margin-top: 36px;
        font-size: 0.85rem;
        border-top: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_clean_data():
    """Loads and cleans raw COVID-19 dataset with schema validation."""
    raw_df = load_raw_dataset()
    is_valid, missing_cols = validate_dataset_schema(raw_df)
    if not is_valid:
        raise ValueError(f"Required columns missing in dataset: {', '.join(missing_cols)}")
    cleaned_df = clean_covid_dataset(raw_df)
    return cleaned_df

def main():
    # Load and clean dataset with error handling
    try:
        df = get_clean_data()
    except Exception as e:
        st.error(f"❌ Dataset could not be loaded: {e}")
        st.info("Please verify that dataset CSV files exist in the 'data/' folder.")
        st.stop()

    if df.empty:
        st.error("❌ The loaded dataset is empty.")
        st.stop()

    # Determine dataset min/max dates and available years dynamically
    min_dataset_date = df["Date"].min().date()
    max_dataset_date = df["Date"].max().date()
    available_years = sorted(df["Date"].dt.year.unique().tolist())

    # --- INITIALIZE SESSION STATE FOR FILTERS ---
    if "preset" not in st.session_state:
        st.session_state.preset = "All Time"
    if "start_date" not in st.session_state:
        st.session_state.start_date = min_dataset_date
    if "end_date" not in st.session_state:
        st.session_state.end_date = max_dataset_date
    if "selected_who" not in st.session_state:
        st.session_state.selected_who = []
    if "selected_countries" not in st.session_state:
        st.session_state.selected_countries = []
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    # --- SIDEBAR CONTROLS ---
    st.sidebar.markdown("## 🎛️ Dashboard Filters")

    # Reset All Filters Button
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        st.session_state.preset = "All Time"
        st.session_state.start_date = min_dataset_date
        st.session_state.end_date = max_dataset_date
        st.session_state.selected_who = []
        st.session_state.selected_countries = []
        st.session_state.search_query = ""
        st.rerun()

    st.sidebar.markdown("---")

    # 1. Date Presets & Custom Range
    st.sidebar.markdown("#### 📅 Date Range Filter")
    preset_options = ["All Time"]
    if 2020 in available_years:
        preset_options.append("2020")
    if 2021 in available_years:
        preset_options.append("2021")
    preset_options.append("Custom Range")

    selected_preset = st.sidebar.radio(
        "Quick Date Presets:",
        options=preset_options,
        index=preset_options.index(st.session_state.preset) if st.session_state.preset in preset_options else 0,
        key="preset_radio"
    )

    if selected_preset != st.session_state.preset:
        st.session_state.preset = selected_preset
        if selected_preset == "All Time":
            st.session_state.start_date = min_dataset_date
            st.session_state.end_date = max_dataset_date
        elif selected_preset == "2020":
            st.session_state.start_date = max(min_dataset_date, datetime(2020, 1, 1).date())
            st.session_state.end_date = min(max_dataset_date, datetime(2020, 12, 31).date())
        elif selected_preset == "2021":
            st.session_state.start_date = max(min_dataset_date, datetime(2021, 1, 1).date())
            st.session_state.end_date = min(max_dataset_date, datetime(2021, 12, 31).date())

    if st.session_state.preset == "Custom Range":
        custom_range = st.sidebar.date_input(
            "Select Date Range:",
            value=(st.session_state.start_date, st.session_state.end_date),
            min_value=min_dataset_date,
            max_value=max_dataset_date
        )
        if isinstance(custom_range, tuple) and len(custom_range) == 2:
            s_date, e_date = custom_range
            if s_date <= e_date:
                st.session_state.start_date = s_date
                st.session_state.end_date = e_date
            else:
                st.sidebar.warning("Start date must be before or equal to End date.")

    st.sidebar.markdown("---")

    # 2. WHO Region Filter
    all_who_regions = sorted(df["WHO Region"].unique().tolist())
    st.sidebar.markdown("#### 🏥 WHO Region")
    selected_who = st.sidebar.multiselect(
        "Select WHO Region(s):",
        options=all_who_regions,
        default=st.session_state.selected_who,
        key="who_multiselect"
    )
    st.session_state.selected_who = selected_who

    # 3. Cascading Country / Region Filter
    # Filter dataset by chosen WHO Region to restrict Country options dynamically
    if selected_who:
        valid_countries_df = df[df["WHO Region"].isin(selected_who)]
    else:
        valid_countries_df = df

    available_countries = sorted(valid_countries_df["Country/Region"].unique().tolist())

    # Automatically prune selected countries if they are no longer in available_countries
    pruned_selected_countries = [c for c in st.session_state.selected_countries if c in available_countries]
    if pruned_selected_countries != st.session_state.selected_countries:
        st.session_state.selected_countries = pruned_selected_countries

    st.sidebar.markdown("#### 🌍 Country / Region")
    selected_countries = st.sidebar.multiselect(
        "Select Country/Region(s):",
        options=available_countries,
        default=st.session_state.selected_countries,
        key="country_multiselect"
    )
    st.session_state.selected_countries = selected_countries

    st.sidebar.markdown("---")

    # 4. Search Filter
    st.sidebar.markdown("#### 🔍 Search Filter")
    search_query = st.sidebar.text_input(
        "Search Country or State/Province:",
        value=st.session_state.search_query,
        key="search_text_input"
    )
    st.session_state.search_query = search_query

    # Apply all filters
    filtered_df = filter_dataframe(
        df,
        start_date=st.session_state.start_date,
        end_date=st.session_state.end_date,
        selected_who_regions=st.session_state.selected_who,
        selected_countries=st.session_state.selected_countries,
        search_query=st.session_state.search_query
    )

    # --- HEADER BANNER ---
    st.markdown("""
    <div class="header-banner">
        <h1 class="header-title">🦠 COVID-19 Global Data & Trends Dashboard</h1>
        <p class="header-subtitle">Interactive Exploratory Data Analysis & Time-Series Tracking System</p>
    </div>
    """, unsafe_allow_html=True)

    # --- ACTIVE FILTERS INFO BANNER ---
    filter_country_text = ", ".join(st.session_state.selected_countries) if st.session_state.selected_countries else "Global / All Countries"
    filter_who_text = ", ".join(st.session_state.selected_who) if st.session_state.selected_who else "All WHO Regions"
    search_text = f' | Search: "{st.session_state.search_query}"' if st.session_state.search_query else ""
    date_text = f"{st.session_state.start_date} to {st.session_state.end_date}"

    st.markdown(f"""
    <div class="filter-banner">
        📍 <strong>Active Filters:</strong> &nbsp;
        Countries: <strong>{filter_country_text}</strong> &nbsp;|&nbsp;
        WHO Region: <strong>{filter_who_text}</strong> &nbsp;|&nbsp;
        Date Range: <strong>{date_text}</strong>{search_text}
    </div>
    """, unsafe_allow_html=True)

    # --- KPI METRIC CARDS ---
    metrics = compute_kpi_metrics(filtered_df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card card-confirmed">
            <div class="kpi-title">Total Confirmed</div>
            <div class="kpi-value">{metrics['total_confirmed']:,}</div>
            <div class="kpi-subtext" style="color:#94a3b8">Cumulative Confirmed Cases</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card card-active">
            <div class="kpi-title">Active Cases</div>
            <div class="kpi-value">{metrics['active_cases']:,}</div>
            <div class="kpi-subtext" style="color:#fbbf24">Current Active Infections</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card card-recovered">
            <div class="kpi-title">Total Recovered</div>
            <div class="kpi-value">{metrics['total_recovered']:,}</div>
            <div class="kpi-subtext" style="color:#4ade80">Recovery Rate: <strong>{metrics['recovery_rate']}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card card-deaths">
            <div class="kpi-title">Total Deaths</div>
            <div class="kpi-value">{metrics['total_deaths']:,}</div>
            <div class="kpi-subtext" style="color:#f87171">Mortality Rate: <strong>{metrics['death_rate']}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- CHECK FOR EMPTY FILTERED DATASET ---
    if filtered_df.empty:
        st.warning("⚠️ No records match the current filters. Try resetting or changing your filters.")
        st.stop()

    # Time series daily calculation
    daily_df = compute_time_series_daily(filtered_df)

    # --- DASHBOARD TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Time-Series Trends",
        "📊 Daily Increments & Moving Averages",
        "🌍 Country & Regional Breakdown",
        "🍩 Case Distribution & Rates",
        "📋 Data Inspector & Export"
    ])

    # --- TAB 1: TIME SERIES TRENDS ---
    with tab1:
        st.markdown("### 📈 Cumulative Cases Over Time")
        if not daily_df.empty:
            is_log = st.checkbox("Enable Logarithmic Scale Axis", value=False, key="ts_log_check")
            
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=daily_df["Date"], y=daily_df["Confirmed"],
                mode="lines", name="Confirmed",
                line=dict(color="#38bdf8", width=3)
            ))
            fig_ts.add_trace(go.Scatter(
                x=daily_df["Date"], y=daily_df["Active"],
                mode="lines", name="Active",
                line=dict(color="#fbbf24", width=2.5)
            ))
            fig_ts.add_trace(go.Scatter(
                x=daily_df["Date"], y=daily_df["Recovered"],
                mode="lines", name="Recovered",
                line=dict(color="#4ade80", width=2.5)
            ))
            fig_ts.add_trace(go.Scatter(
                x=daily_df["Date"], y=daily_df["Deaths"],
                mode="lines", name="Deaths",
                line=dict(color="#f87171", width=2.5)
            ))

            fig_ts.update_layout(
                template="plotly_dark",
                height=480,
                margin=dict(l=20, r=20, t=30, b=20),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(type="log" if is_log else "linear", title="Cases Count")
            )
            st.plotly_chart(fig_ts, use_container_width=True)

    # --- TAB 2: DAILY INCREMENTS ---
    with tab2:
        st.markdown("### 📊 Daily New Cases & Deaths Trend")
        if not daily_df.empty:
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("#### Daily New Confirmed Cases")
                fig_daily_cases = go.Figure()
                fig_daily_cases.add_trace(go.Bar(
                    x=daily_df["Date"], y=daily_df["Daily_New_Cases"],
                    name="Daily New Cases",
                    marker_color="#38bdf8",
                    opacity=0.65
                ))
                fig_daily_cases.add_trace(go.Scatter(
                    x=daily_df["Date"], y=daily_df["New_Cases_7d_MA"],
                    name="7-Day Moving Avg",
                    line=dict(color="#0284c7", width=3)
                ))
                fig_daily_cases.update_layout(
                    template="plotly_dark",
                    height=420,
                    margin=dict(l=20, r=20, t=30, b=20),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_daily_cases, use_container_width=True)

            with c2:
                st.markdown("#### Daily New Deaths")
                fig_daily_deaths = go.Figure()
                fig_daily_deaths.add_trace(go.Bar(
                    x=daily_df["Date"], y=daily_df["Daily_Deaths"],
                    name="Daily Deaths",
                    marker_color="#f87171",
                    opacity=0.65
                ))
                fig_daily_deaths.add_trace(go.Scatter(
                    x=daily_df["Date"], y=daily_df["Daily_Deaths_7d_MA"],
                    name="7-Day Moving Avg",
                    line=dict(color="#dc2626", width=3)
                ))
                fig_daily_deaths.update_layout(
                    template="plotly_dark",
                    height=420,
                    margin=dict(l=20, r=20, t=30, b=20),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_daily_deaths, use_container_width=True)

    # --- TAB 3: COUNTRY & REGIONAL BREAKDOWN ---
    with tab3:
        st.markdown("### 🌍 Country & WHO Regional Analysis")
        col_rank, col_who = st.columns([1.1, 0.9])
        
        with col_rank:
            st.markdown("#### Top Countries Comparison")
            metric_choice = st.selectbox(
                "Rank Countries By:",
                ["Confirmed", "Deaths", "Recovered", "Active"],
                index=0,
                key="rank_metric_select"
            )
            top_n = st.slider("Select Top N Countries:", min_value=5, max_value=25, value=10, key="top_n_slider")
            
            top_countries_df = compute_country_comparison(filtered_df, metric=metric_choice, top_n=top_n)
            
            if not top_countries_df.empty:
                color_map = {
                    "Confirmed": "#38bdf8",
                    "Deaths": "#f87171",
                    "Recovered": "#4ade80",
                    "Active": "#fbbf24"
                }
                fig_bar = px.bar(
                    top_countries_df,
                    x=metric_choice,
                    y="Country/Region",
                    orientation="h",
                    text_auto=True,
                    color_discrete_sequence=[color_map.get(metric_choice, "#38bdf8")]
                )
                fig_bar.update_layout(
                    template="plotly_dark",
                    height=450,
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
        with col_who:
            st.markdown("#### WHO Regional Distribution")
            who_df = compute_who_region_breakdown(filtered_df)
            if not who_df.empty:
                fig_treemap = px.treemap(
                    who_df,
                    path=["WHO Region"],
                    values="Confirmed",
                    color="Confirmed",
                    color_continuous_scale="Viridis",
                    title="Confirmed Cases by WHO Region"
                )
                fig_treemap.update_layout(
                    template="plotly_dark",
                    height=480,
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                st.plotly_chart(fig_treemap, use_container_width=True)

    # --- TAB 4: CASE DISTRIBUTION & RATES ---
    with tab4:
        st.markdown("### 🍩 Global Case Proportions & Rate Analysis")
        c_donut, c_rates = st.columns(2)
        
        with c_donut:
            st.markdown("#### Case Breakdown Proportions")
            dist_data = {
                "Status": ["Active", "Recovered", "Deaths"],
                "Count": [metrics["active_cases"], metrics["total_recovered"], metrics["total_deaths"]]
            }
            dist_df = pd.DataFrame(dist_data)
            
            fig_donut = px.pie(
                dist_df,
                values="Count",
                names="Status",
                hole=0.55,
                color="Status",
                color_discrete_map={
                    "Active": "#fbbf24",
                    "Recovered": "#4ade80",
                    "Deaths": "#f87171"
                }
            )
            fig_donut.update_layout(
                template="plotly_dark",
                height=420,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with c_rates:
            st.markdown("#### Recovery Rate (%) vs Mortality Rate (%) Trajectory")
            if not daily_df.empty:
                fig_rates = go.Figure()
                fig_rates.add_trace(go.Scatter(
                    x=daily_df["Date"], y=daily_df["Recovery_Rate"],
                    name="Recovery Rate (%)",
                    line=dict(color="#4ade80", width=2.5)
                ))
                fig_rates.add_trace(go.Scatter(
                    x=daily_df["Date"], y=daily_df["Death_Rate"],
                    name="Mortality Rate (%)",
                    line=dict(color="#f87171", width=2.5)
                ))
                fig_rates.update_layout(
                    template="plotly_dark",
                    height=420,
                    margin=dict(l=20, r=20, t=30, b=20),
                    hovermode="x unified",
                    yaxis=dict(title="Percentage (%)")
                )
                st.plotly_chart(fig_rates, use_container_width=True)

    # --- TAB 5: DATA INSPECTOR & EXPORT ---
    with tab5:
        st.markdown("### 📋 Cleaned Dataset Inspector & Exporter")
        st.info(f"📊 Displaying **{len(filtered_df):,}** records matching current filter criteria.")
        
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Download CSV button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data CSV",
            data=csv_data,
            file_name="covid_19_filtered_analysis.csv",
            mime="text/csv",
            key="csv_download_btn"
        )

    # Footer
    st.markdown("""
    <div class="footer">
        COVID-19 Data Analysis & Visualization Dashboard • Built with Python, Pandas, Plotly & Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
