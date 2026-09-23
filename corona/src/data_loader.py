import os
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

ONLINE_FALLBACKS = {
    "clean_complete": "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv",
    "day_wise": "https://raw.githubusercontent.com/datasets/covid-19/main/data/worldwide-aggregate.csv"
}

@st.cache_data(show_spinner=False)
def load_raw_dataset(filename: str = "covid_19_clean_complete.csv") -> pd.DataFrame:
    """
    Loads COVID-19 dataset from local data folder. If missing, attempts online retrieval.
    """
    local_path = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        return df
    
    # Check fallback for known filenames
    if filename in ONLINE_FALLBACKS:
        try:
            df = pd.read_csv(ONLINE_FALLBACKS[filename])
            return df
        except Exception as e:
            st.error(f"Error downloading online fallback dataset: {e}")
            
    raise FileNotFoundError(f"Dataset file '{filename}' not found at path {local_path}.")


@st.cache_data(show_spinner=False)
def load_country_latest_dataset(filename: str = "country_wise_latest.csv") -> pd.DataFrame:
    """
    Loads latest country snapshot dataset.
    """
    local_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_day_wise_dataset(filename: str = "day_wise.csv") -> pd.DataFrame:
    """
    Loads day-wise aggregated global dataset.
    """
    local_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    return pd.DataFrame()
