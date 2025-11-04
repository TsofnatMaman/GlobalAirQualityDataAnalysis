# Global Air Quality Data Analysis

This project analyzes and visualizes data on global air pollutant concentrations (PM2.5, PM10, NO2) using data from the World Health Organization's (WHO) Ambient Air Quality Database. The analysis includes statistics, trends over the years, and the creation of a visual map of global PM2.5 concentrations.

## Key Technologies

* **Python**
* **Pandas** - For data processing
* **Matplotlib, Seaborn** - For creating graphs and figures
* **OpenCV (cv2)** - For image processing and creating the air pollution map

## File Structure

* **`tasks_1_2_3.py`**: Contains functions for data analysis, including:
    * Data overview (initial data, statistics, Null values).
    * Calculating column statistics (mean, median, max, min, standard deviation).
    * Displaying global air pollution trends over the years (f1).
    * Displaying air pollution trends for a specific country/city (f2).
    * Showing the relationship between PM2.5 and PM10 (f3).
    * Showing the relationship between PM2.5 concentration and population size (f4).
* **`task_4_analize_coordination_cv.py`**: A file that uses **OpenCV** to create a visualization of global PM2.5 levels on the physical earth map.
* **`exceptions.py`**: Contains a custom exception class: `CountryNotFoundException`.
* **`who_ambient_air_quality_database_version_2024_(v6.1).xlsx`**: The raw data file (must be downloaded from the WHO repository).
* **`physical-earth-map-geographic-grid-lines.png`** / **`colorsBar.png`**: Graphic reference files used for creating the air pollution map.

## How to Run

1.  Ensure the **Excel** data file is located in the folder (in the project sources, it's named: `who_ambient_air_quality_database_version_2024_(v6.1).xlsx`).
2.  Run the **`tasks_1_2_3.py`** file to display the statistical and graphical analyses (including graphs f1 through f4).
3.  Run the **`task_4_analize_coordination_cv.py`** file to generate the air pollution map for 2020. The map will be saved as `AirPollutionMap2020.jpg`.