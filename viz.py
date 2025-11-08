from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def data_viewer(df: pd.DataFrame) -> None:
    print(df.head())
    print("describe data:\n", df.describe(include="all"))
    print("columns of data:\n", df.columns.tolist())
    print("sum of null values:\n", df.isnull().sum())
    print("types of columns:\n", df.dtypes)

def column_stats(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        logger.error("Column %s not found in DataFrame", col)
        raise KeyError(f"Column {col} not found")
    vals = df[col].dropna().values.astype(float)
    print(f"mean of {col}: {np.mean(vals):.4f}")
    print(f"median of {col}: {np.median(vals):.4f}")
    print(f"max of {col}: {np.max(vals):.4f}")
    print(f"min of {col}: {np.min(vals):.4f}")
    print(f"std of {col}: {np.std(vals):.4f}")

def f1_global_trends(df: pd.DataFrame, pollution_cols, show: bool = True, save_path: Optional[str] = None):
    global_pollution = df.groupby("year")[pollution_cols].mean().dropna()
    years = global_pollution.index.values
    plt.figure(figsize=(12, 6))
    width = 0.8 / len(pollution_cols)
    for i, col in enumerate(pollution_cols):
        plt.bar(years + i * width, global_pollution[col].values, width=width, label=col)
    plt.xlabel("Year")
    plt.ylabel("Pollution")
    plt.title("Pollution vs Year (Global Averages)")
    plt.legend()
    plt.grid(axis="y")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Saved global trends to %s", save_path)
    if show:
        plt.show()
    plt.close()

def f2_country_trend(df: pd.DataFrame, country: str, get_processed_country_data_fn, city: str = "all", show: bool = True, save_path: Optional[str] = None):
    country_df = get_processed_country_data_fn(df, country, city)
    if country_df.empty:
        logger.warning("No data to plot for %s (%s)", country, city)
        return
    avg_pollution = country_df.mean(axis=1)
    plt.figure(figsize=(10, 5))
    plt.plot(avg_pollution.index, avg_pollution.values, marker="o")
    plt.xlabel("Year")
    plt.ylabel("Average Pollution")
    plt.title(f"Air pollution level per year for country: {country} (city filter: {city})")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Saved country trend to %s", save_path)
    if show:
        plt.show()
    plt.close()

def f3_pm25_vs_pm10(df: pd.DataFrame, show: bool = True, save_path: Optional[str] = None):
    plt.figure(figsize=(8, 6))
    plt.scatter(df["pm25_concentration"], df["pm10_concentration"], marker=".")
    plt.xlabel("PM2.5")
    plt.ylabel("PM10")
    plt.title("PM2.5 vs PM10")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Saved PM2.5 vs PM10 to %s", save_path)
    if show:
        plt.show()
    plt.close()

def f4_population_vs_pm25(df: pd.DataFrame, show: bool = True, save_path: Optional[str] = None):
    sub = df.dropna(subset=["pm25_concentration", "population"])
    plt.figure(figsize=(12, 8))
    plt.scatter(sub["population"], sub["pm25_concentration"], alpha=0.6)
    plt.xscale("log")
    plt.xlabel("Population (log scale)")
    plt.ylabel("PM2.5")
    plt.title("PM2.5 Concentration vs Population (log scale)")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Saved population vs pm25 to %s", save_path)
    if show:
        plt.show()
    plt.close()
