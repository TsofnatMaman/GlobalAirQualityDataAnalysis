import pandas as pd
import sys
from exceptions import CountryNotFoundException

DATA_FILE_NAME = "who_ambient_air_quality_database_version_2024_(v6.1).xlsx"
SHEET_NAME = "Update 2024 (V6.1)"

pollution = ["pm10_concentration", "pm25_concentration", "no2_concentration"]

# --- data load ---
try:
    pf = pd.read_excel(DATA_FILE_NAME, sheet_name=SHEET_NAME)
    print(f"Data loaded successfully from {DATA_FILE_NAME}")

except FileNotFoundError:
    print(f"Error: Data file not found.")
    print(f"Please download '{DATA_FILE_NAME}' to the project folder.")
    sys.exit(1)  # exit if file not exist
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    sys.exit(1)


def get_processed_country_data(country, city="all"):
    """
    מסנן את ה-DataFrame הראשי לפי מדינה (ועיר אופציונלית),
    ומחשב את הממוצע השנתי של המזהמים.
    """
    # 1. סינון נתונים לפי מדינה
    country_data = (pf[pf["country_name"].str.lower() == country.lower()])
    if country_data.empty:
        raise CountryNotFoundException(country)

    # 2. סינון לפי עיר (אם סופקה)
    if city != "all":
        country_data = country_data[country_data["city"].str.lower().str.contains(city.lower())]
        if country_data.empty:
            # במקרה הזה לא נזרוק שגיאה, רק נחזיר דאטה-פריים ריק
            print(f"Warning: No data found for city '{city}' in '{country}'.")

    # 3. חישוב ממוצע שנתי והחזרה
    country_pollution = country_data.groupby("year")[pollution].mean().dropna()

    return country_pollution