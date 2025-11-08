import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

from data_loader import pf, pollution, get_processed_country_data
from exceptions import CountryNotFoundException


def _perform_prediction_and_plot(data, target_year, title):
    """
    פונקציית עזר פנימית לביצוע החיזוי והשרטוט.
    משתמשת במודל לוג-ליניארי (y = e^(ax+b))
    """

    predictions = {}
    all_plot_values = []

    plt.figure(figsize=(12, 7))

    for p in pollution:
        # 1. הכנת נתונים לאימון
        X = data.index.values.reshape(-1, 1)
        y = data[p].values

        # 1א. סינון נתונים: אפשר להשתמש רק בערכי y > 0
        positive_mask = y > 0
        if np.sum(positive_mask) < 2:  # אין מספיק נתונים חיוביים לאמן
            print(f"Skipping {p}: Not enough positive data points for Log-Linear model.")
            continue

        X_train = X[positive_mask]
        y_train = y[positive_mask]

        # 1ב. המרת y ל- log(y)
        y_log_train = np.log(y_train)

        # 1ג. יצירת מודל ליניארי פשוט
        model = LinearRegression()

        # 1ד. אימון המודל על log(y)
        model.fit(X_train, y_log_train)

        # 2. הדפסת תוצאות
        # ציון ה-R² מחושב על ההתאמה ל-log(y), לא ל-y
        r2_score = model.score(X_train, y_log_train)
        print(f"  Training results for {p} (Log-Linear Model):")
        print(f"    Model Fit (R² on log(y)): {r2_score:.3f}")

        # 3. חיזוי

        # 3א. חישוב עקומת החיזוי ההיסטורית (במרחב הלוגריתמי)
        y_log_curve_pred = model.predict(X)
        # 3ב. המרה של העקומה בחזרה למרחב הרגיל (אקספוננט)
        y_curve_pred = np.exp(y_log_curve_pred)

        # 3ג. חיזוי הערך העתידי (במרחב הלוגריתמי)
        predicted_log_value = model.predict([[target_year]])[0]
        # 3ד. המרה בחזרה - התוצאה תמיד תהיה חיובית
        predicted_value = np.exp(predicted_log_value)

        predictions[p] = predicted_value

        all_plot_values.extend(y)
        all_plot_values.extend(y_curve_pred)
        all_plot_values.append(predicted_value)

        print(f"Predicted {p} for {target_year}: {predicted_value:.2f}")

        # 4. הצגה ויזואלית
        line = plt.plot(X, y, marker='o', linestyle='-', label=f"Historical {p}")
        line_color = line[0].get_color()
        # שרטוט העקומה האקספוננציאלית
        plt.plot(X, y_curve_pred, color='gray', linestyle='--', label='Log-Linear Curve (Fit)')

        # שרטוט ה-X (תמיד יהיה חיובי)
        plt.plot([target_year], [predicted_value], marker='X', color=line_color, markersize=10, linestyle='None',
                 label=f"Predicted {p} ({predicted_value:.2f})")

    # 5. הגדרות גרף סופיות
    plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Physical Zero')

    current_ymin, current_ymax = plt.ylim()
    # ציר ה-Y תמיד יתחיל ב-0 או נמוך יותר
    plt.ylim(bottom=min(current_ymin, 0), top=(current_ymax * 1.1) + 1)

    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("Pollution")
    plt.title(title)
    plt.grid(True)
    plt.show()


def f5_predict_global_pollution(target_year=None):
    """
    פונקציית Wrapper: מכינה נתונים גלובליים וקוראת לפונקציית החיזוי.
    """
    print("\n--- Global Pollution Prediction (Log-Linear Model) ---")

    # 1. הכנת נתונים
    global_pollution = pf.groupby("year")[pollution].mean().dropna()

    # 2. קביעת שנת יעד
    if target_year is None:
        last_year = global_pollution.index.max()
        target_year = last_year + 1

    print(f"Predicting pollution levels for {target_year} based on historical data...")

    # 3. קריאה לפונקציה המשותפת
    _perform_prediction_and_plot(global_pollution, target_year,
                                 f"Global Pollution Prediction (Log-Linear) for {target_year}")


def f6_predict_country_pollution(country, city="all", target_year=None):
    """
    פונקציית Wrapper: מכינה נתונים למדינה וקוראת לפונקציית החיזוי.
    """
    print(f"\n--- Country Pollution Prediction (Log-Linear Model) for: {country} ({city}) ---")

    # 1. קבלת נתונים מעובדים
    country_pollution = get_processed_country_data(country, city)

    if country_pollution.empty:
        print(f"Not enough historical data to make a prediction for {country} ({city}).")
        return

    # 2. קביעת שנת היעד
    if target_year is None:
        last_year = country_pollution.index.max()
        target_year = last_year + 1

    print(f"Predicting pollution levels for {target_year}...")

    # 3. קריאה לפונקציה המשותפת
    _perform_prediction_and_plot(country_pollution, target_year,
                                 f"Prediction for {country} - {city} (Log-Linear) for {target_year}")


if __name__ == "__main__":

    f5_predict_global_pollution()
    f5_predict_global_pollution(target_year=2030)

    print("-" * 30)

    try:
        f6_predict_country_pollution("Israel", "tel aviv")
        f6_predict_country_pollution("Germany", target_year=2030)

    except CountryNotFoundException as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")