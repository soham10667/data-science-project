import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def filter_dataframe(
    df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    selected_who_regions: Optional[List[str]] = None,
    selected_countries: Optional[List[str]] = None,
    search_query: Optional[str] = None
) -> pd.DataFrame:
    """
    Applies user sidebar filters dynamically and safely on the dataset.
    Cascading order: Date Range -> WHO Region -> Country/Region -> Search.
    """
    if df.empty:
        return df.copy()

    filtered = df.copy()

    # 1. Filter by Date range
    if "Date" in filtered.columns:
        if start_date is not None:
            filtered = filtered[filtered["Date"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            filtered = filtered[filtered["Date"] <= pd.to_datetime(end_date)]

    # 2. Filter by WHO Region
    if selected_who_regions and len(selected_who_regions) > 0 and "All" not in selected_who_regions:
        filtered = filtered[filtered["WHO Region"].isin(selected_who_regions)]

    # 3. Filter by selected countries
    if selected_countries and len(selected_countries) > 0 and "All" not in selected_countries:
        filtered = filtered[filtered["Country/Region"].isin(selected_countries)]

    # 4. Search filter (case-insensitive substring match on Country or Province/State)
    if search_query:
        query = search_query.strip().lower()
        if query:
            match_country = filtered["Country/Region"].astype(str).str.lower().str.contains(query)
            match_state = filtered["Province/State"].astype(str).str.lower().str.contains(query)
            filtered = filtered[match_country | match_state]

    return filtered


def compute_kpi_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes accurate KPI summary statistics from filtered dataset.
    
    Cumulative COVID Totals Logic:
    To avoid improperly summing cumulative values across multiple dates, 
    for each unique location (Country/Region + Province/State), we extract 
    the latest available record within the selected date range.
    """
    if df.empty:
        return {
            "total_confirmed": 0,
            "total_deaths": 0,
            "total_recovered": 0,
            "active_cases": 0,
            "recovery_rate": 0.0,
            "death_rate": 0.0
        }

    # Group by location and get the latest record in the date range
    if "Date" in df.columns and "Province/State" in df.columns:
        latest_records = (
            df.sort_values("Date")
            .groupby(["Country/Region", "Province/State"], as_index=False)
            .last()
        )
    elif "Date" in df.columns:
        latest_records = (
            df.sort_values("Date")
            .groupby("Country/Region", as_index=False)
            .last()
        )
    else:
        latest_records = df

    confirmed = int(latest_records["Confirmed"].sum())
    deaths = int(latest_records["Deaths"].sum())
    recovered = int(latest_records["Recovered"].sum())
    
    # Active = Confirmed - Deaths - Recovered (non-negative)
    active = max(0, confirmed - deaths - recovered)

    # Rates handling divide-by-zero safely
    recovery_rate = (recovered / confirmed * 100) if confirmed > 0 else 0.0
    death_rate = (deaths / confirmed * 100) if confirmed > 0 else 0.0

    return {
        "total_confirmed": confirmed,
        "total_deaths": deaths,
        "total_recovered": recovered,
        "active_cases": active,
        "recovery_rate": round(recovery_rate, 2),
        "death_rate": round(death_rate, 2)
    }


def compute_time_series_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates daily time-series metrics.
    Calculates Daily New Confirmed, Daily New Deaths, Daily New Recovered,
    and 7-day rolling moving averages.
    
    Note on negative daily increments:
    Dataset corrections occasionally produce negative daily diffs.
    In this implementation, negative daily diffs are safely clipped to 0
    to ensure clean visual trend representation without chart distortion.
    """
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    daily_df = (
        df.groupby("Date", as_index=False)[["Confirmed", "Deaths", "Recovered", "Active"]]
        .sum()
        .sort_values("Date")
    )

    # Calculate daily increments
    daily_df["Daily_New_Cases"] = daily_df["Confirmed"].diff().fillna(0).clip(lower=0)
    daily_df["Daily_Deaths"] = daily_df["Deaths"].diff().fillna(0).clip(lower=0)
    daily_df["Daily_Recovered"] = daily_df["Recovered"].diff().fillna(0).clip(lower=0)

    # Rates over time
    daily_df["Recovery_Rate"] = np.where(
        daily_df["Confirmed"] > 0,
        (daily_df["Recovered"] / daily_df["Confirmed"]) * 100,
        0.0
    )
    daily_df["Death_Rate"] = np.where(
        daily_df["Confirmed"] > 0,
        (daily_df["Deaths"] / daily_df["Confirmed"]) * 100,
        0.0
    )

    # 7-day rolling averages
    daily_df["New_Cases_7d_MA"] = daily_df["Daily_New_Cases"].rolling(window=7, min_periods=1).mean()
    daily_df["Daily_Deaths_7d_MA"] = daily_df["Daily_Deaths"].rolling(window=7, min_periods=1).mean()

    return daily_df


def compute_country_comparison(df: pd.DataFrame, metric: str = "Confirmed", top_n: int = 10) -> pd.DataFrame:
    """
    Computes top N countries ranked by selected metric (Confirmed, Deaths, Recovered, Active).
    """
    if df.empty:
        return pd.DataFrame()

    if "Date" in df.columns:
        latest_df = (
            df.sort_values("Date")
            .groupby(["Country/Region", "Province/State"], as_index=False)
            .last()
        )
    else:
        latest_df = df

    country_agg = (
        latest_df.groupby("Country/Region", as_index=False)[["Confirmed", "Deaths", "Recovered", "Active"]]
        .sum()
    )

    if metric in country_agg.columns:
        country_agg.sort_values(by=metric, ascending=False, inplace=True)

    return country_agg.head(top_n)


def compute_who_region_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates cases by WHO Region.
    """
    if df.empty or "WHO Region" not in df.columns:
        return pd.DataFrame()

    if "Date" in df.columns:
        latest_df = (
            df.sort_values("Date")
            .groupby(["Country/Region", "Province/State"], as_index=False)
            .last()
        )
    else:
        latest_df = df

    who_df = (
        latest_df.groupby("WHO Region", as_index=False)[["Confirmed", "Deaths", "Recovered", "Active"]]
        .sum()
        .sort_values(by="Confirmed", ascending=False)
    )
    return who_df
