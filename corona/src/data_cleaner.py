import pandas as pd
import numpy as np
from typing import Tuple, List

# Standard country name mapping
COUNTRY_NAME_MAPPINGS = {
    "US": "United States",
    "USA": "United States",
    "Mainland China": "China",
    "UK": "United Kingdom",
    "Great Britain": "United Kingdom",
    "Korea, South": "South Korea",
    "Republic of Korea": "South Korea",
    "St. Martin": "Saint Martin",
    "Taiwan*": "Taiwan",
    "West Bank and Gaza": "Palestine"
}

REQUIRED_COLUMNS = ["Date", "Country/Region", "Confirmed", "Deaths", "Recovered"]

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column names to standard dataset schema.
    Handles variations like Country_Region, Province_State, WHO_Region, etc.
    """
    col_mapping = {}
    for col in df.columns:
        c_clean = col.strip()
        if c_clean in ["Country_Region", "Country/Region", "Country"]:
            col_mapping[col] = "Country/Region"
        elif c_clean in ["Province_State", "Province/State", "State"]:
            col_mapping[col] = "Province/State"
        elif c_clean in ["WHO_Region", "WHO Region"]:
            col_mapping[col] = "WHO Region"
        else:
            col_mapping[col] = c_clean
            
    return df.rename(columns=col_mapping)


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates if required columns exist in the dataframe.
    Returns (is_valid, missing_columns_list).
    """
    normalized_cols = set(df.columns)
    missing = [col for col in REQUIRED_COLUMNS if col not in normalized_cols]
    return len(missing) == 0, missing


def clean_covid_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw COVID-19 dataframe:
    - Normalizes column headers
    - Drops full duplicate rows safely
    - Converts Date column to datetime64
    - Standardizes Country/Region names and strips extra spaces
    - Fills missing numeric values with 0
    - Computes non-negative Active cases column (Active = Confirmed - Deaths - Recovered)
    """
    cleaned_df = df.copy()
    
    # 1. Normalize Column Headers
    cleaned_df = normalize_column_names(cleaned_df)
    
    # 2. Deduplicate
    cleaned_df.drop_duplicates(inplace=True)

    # 3. Handle Date column
    if "Date" in cleaned_df.columns:
        cleaned_df["Date"] = pd.to_datetime(cleaned_df["Date"], errors="coerce")
        cleaned_df.dropna(subset=["Date"], inplace=True)
        cleaned_df.sort_values(by="Date", inplace=True)

    # 4. Handle Country/Region standardization
    if "Country/Region" in cleaned_df.columns:
        cleaned_df["Country/Region"] = (
            cleaned_df["Country/Region"]
            .astype(str)
            .str.strip()
            .replace(COUNTRY_NAME_MAPPINGS)
        )

    # 5. Handle Province/State
    if "Province/State" in cleaned_df.columns:
        cleaned_df["Province/State"] = (
            cleaned_df["Province/State"]
            .fillna("All / Main Region")
            .astype(str)
            .str.strip()
        )
    else:
        cleaned_df["Province/State"] = "All / Main Region"

    # 6. Handle WHO Region
    if "WHO Region" in cleaned_df.columns:
        cleaned_df["WHO Region"] = (
            cleaned_df["WHO Region"]
            .fillna("Global")
            .astype(str)
            .str.strip()
        )
    else:
        cleaned_df["WHO Region"] = "Global"

    # 7. Numeric columns validation & non-negative clip
    numeric_cols = ["Confirmed", "Deaths", "Recovered"]
    for col in numeric_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce").fillna(0)
            cleaned_df[col] = cleaned_df[col].clip(lower=0)
        else:
            cleaned_df[col] = 0

    # 8. Calculate Active cases non-negatively
    cleaned_df["Active"] = (cleaned_df["Confirmed"] - cleaned_df["Deaths"] - cleaned_df["Recovered"]).clip(lower=0)

    # Ensure Lat and Long exist for map/location completeness
    if "Lat" not in cleaned_df.columns:
        cleaned_df["Lat"] = 0.0
    if "Long" not in cleaned_df.columns:
        cleaned_df["Long"] = 0.0

    return cleaned_df
