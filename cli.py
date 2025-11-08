import logging
from pathlib import Path
from data_loader import load_data, POLLUTION_COLS, get_processed_country_data
from modeling import perform_prediction_and_plot, save_prediction_report
import viz
from map_viz import plot_pm25_on_map
from exceptions import CountryNotFoundException

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def main():
    try:
        df = load_data()
    except FileNotFoundError as e:
        logger.error("Please download the Excel file and place it in the project folder.")
        return
    viz.data_viewer(df)
    try:
        viz.column_stats(df, "pm25_concentration")
    except KeyError:
        logger.warning("pm25_concentration missing from dataset.")
    viz.f1_global_trends(df, POLLUTION_COLS, show=False, save_path="global_trends.png")
    logger.info("Global trends plot saved to global_trends.png")
    try:
        viz.f2_country_trend(df, "Israel", get_processed_country_data, city="tel aviv", show=False, save_path="israel_telaviv_trend.png")
        logger.info("Saved Israel (Tel Aviv) trend.")
    except CountryNotFoundException as e:
        logger.error(e)
    global_df = df.groupby("year")[POLLUTION_COLS].mean().dropna()
    preds_global = perform_prediction_and_plot(global_df, target_year=2030, title="Global Pollution Prediction", show=False, save_path="global_prediction_2030.png")
    save_prediction_report(preds_global, "prediction_report_global_2030.csv")
    logger.info("Predictions (global -> 2030): %s", preds_global)
    try:
        country_df = get_processed_country_data(df, "Germany")
        if not country_df.empty:
            preds_country = perform_prediction_and_plot(country_df, target_year=2030, title="Germany Prediction", show=False, save_path="germany_prediction_2030.png")
            save_prediction_report(preds_country, "prediction_report_germany_2030.csv")
            logger.info("Germany preds: %s", preds_country)
    except CountryNotFoundException as e:
        logger.error(e)
    map_image = Path("physical-earth-map-geographic-grid-lines.png")
    output_map = Path("AirPollutionMap2020_with_colorbar.jpg")
    try:
        plot_pm25_on_map(df, 2020, map_image, output_map)
        logger.info("Saved map to %s", output_map)
    except FileNotFoundError:
        logger.warning("Map image not found; skipping map creation.")
    except Exception as e:
        logger.error("Failed to create map: %s", e)

if __name__ == "__main__":
    main()
