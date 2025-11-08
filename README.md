# Global Air Quality Data Analysis

This project analyzes and visualizes data on global air pollutant concentrations (PM2.5, PM10, NO2) using data from the World Health Organization's (WHO) Ambient Air Quality Database. The analysis includes statistics, trends over the years, creation of a visual map of global PM2.5 concentrations, and a prediction model for future trends.

## Key Technologies

* **Python**
* **Pandas** - For data processing
* **Matplotlib, Seaborn** - For creating graphs and figures
* **OpenCV (cv2)** - For image processing and creating the air pollution map
* **Scikit-learn** - For building the linear regression prediction model

## File Structure

* **`data_loader.py`**: A shared module responsible for loading the main Excel data file (`who_ambient_air_quality_database_version_2024_(v6.1).xlsx`) into a pandas DataFrame (`pf`). It also defines shared constants like the `pollution` list. This prevents code duplication.
* **`viz.py`**: Imports data from `data_loader.py` and contains functions for data analysis, including:
    * Data overview (initial data, statistics, Null values).
    * Calculating column statistics (mean, median, max, min, standard deviation).
    * Displaying global air pollution trends over the years (f1).
    * Displaying air pollution trends for a specific country/city (f2).
    * Showing the relationship between PM2.5 and PM10 (f3).
    * Showing the relationship between PM2.5 concentration and population size (f4).
* **`map_viz.py`**: A file that imports data from `data_loader.py` and uses **OpenCV** to create a visualization of global PM2.5 levels on the physical earth map.
* **`modelling.py`**: Imports data from `data_loader.py` and uses **Scikit-learn** to build a linear regression model. It predicts the global average pollution levels for the next year based on historical trends.
* **`exceptions.py`**: Contains a custom exception class: `CountryNotFoundException`.
* **`who_ambient_air_quality_database_version_2024_(v6.1).xlsx`**: The raw data file (must be downloaded from the WHO repository).
* **`physical-earth-map-geographic-grid-lines.png`** / **`colorsBar.png`**: Graphic reference files used for creating the air pollution map.

## How to Run

1.  Ensure the **Excel** data file is located in the folder (in the project sources, it's named: `who_ambient_air_quality_database_version_2024_(v6.1).xlsx`).
2.  Ensure required libraries are installed: `pip install pandas openpyxl matplotlib seaborn opencv-python scikit-learn`.
3.  Run the **`cli.py`** file to display the statistical and graphical analyses.
