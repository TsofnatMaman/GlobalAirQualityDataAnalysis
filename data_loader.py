from pathlib import Path
from typing import List
import pandas as pd
import logging

DATA_FILE_NAME = Path("who_ambient_air_quality_database_version_2024_(v6.1).xlsx")
SHEET_NAME = "Update 2024 (V6.1)"
POLLUTION_COLS: List[str] = ["pm10_concentration", "pm25_concentration", "no2_concentration"]

logger = logging.getLogger(__name__)

def load_data(path: Path = DATA_FILE_NAME, sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    if not path.exists():
        logger.error("Data file not found: %s", path)
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name)
    str_cols = df.select_dtypes(include=["object"]).columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df[df["year"].notna()].copy()
        df["year"] = df["year"].astype(int)
    if "country_name" in df.columns:
        df["country_name"] = df["country_name"].astype(str).str.strip()
    if "city" in df.columns:
        df["city"] = df["city"].astype(str).str.strip()
    logger.info("Loaded data from %s (rows=%d, cols=%d)", path, len(df), len(df.columns))
    return df

def get_processed_country_data(df: pd.DataFrame, country: str, city: str = "all") -> pd.DataFrame:
    from exceptions import CountryNotFoundException
    if "country_name" not in df.columns:
        raise KeyError("Expected 'country_name' column in DataFrame")
    country_mask = df["country_name"].astype(str).str.lower() == country.lower()
    country_df = df.loc[country_mask].copy()
    if country_df.empty:
        raise CountryNotFoundException(country)
    if city and city.lower() != "all":
        city_mask = country_df["city"].astype(str).str.lower().str.contains(city.lower(), na=False)
        country_df = country_df.loc[city_mask].copy()
        if country_df.empty:
            return pd.DataFrame(columns=["year"] + POLLUTION_COLS).set_index("year")
    grouped = country_df.groupby("year")[POLLUTION_COLS].mean().dropna(how="all")
    return grouped
