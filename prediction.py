import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# ייבוא הנתונים והפונקציות המשותפות מהמודול שלנו
from data_loader import pf, pollution, get_processed_country_data
from exceptions import CountryNotFoundException


def _perform_prediction_and_plot(data, target_year, title):
    """
    פונקציית עזר פנימית לביצוע החיזוי והשרטוט.
    זוהי הלוגיקה המשותפת שחולצה מ-f5 ו-f6.
    """

    predictions = {}
    all_plot_values = []

    plt.figure(figsize=(12, 7))

    for p in pollution:
        # 1. הכנת נתונים לאימון
        X = data.index.values.reshape(-1, 1)
        y = data[p].values

        if len(X) < 2:
            print(f"Skipping {p}: Not enough data points to train model.")
            continue

        # שימוש במודל היציב (דרגה 2)
        degree = 2
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])

        model.fit(X, y)

        # 2. הדפסת תוצאות
        r2_score = model.score(X, y)
        print(f"  Training results for {p} (Polynomial Degree {degree}):")
        print(f"    Model Fit (R²): {r2_score:.3f}")

        # 3. חיזוי
        y_pred_historical = model.predict(X)

        # --- 🟢 שינוי: הוספת "היגיון פיזיקלי" ---

        # 3א. חישוב התחזית "הגולמית" של המודל
        raw_predicted_value = model.predict([[target_year]])[0]

        # 3ב. "כפייה" של ערך 0 כרצפה פיזיקלית
        # הערך שיוצג בגרף יהיה 0 אם התחזית שלילית
        plotted_value = max(0, raw_predicted_value)

        predictions[p] = plotted_value

        all_plot_values.extend(y)
        all_plot_values.extend(y_pred_historical)
        all_plot_values.append(plotted_value)  # הוספת הערך הכפוי לרשימה

        # 3ג. הדפסת התוצאה עם הערה
        print(f"Predicted {p} for {target_year}: {raw_predicted_value:.2f}")
        if raw_predicted_value < 0:
            print(f"    (Prediction clamped to 0.0 for physical realism)")
        # --- ------------------------------------ ---

        # 4. הצגה ויזואלית
        line = plt.plot(X, y, marker='o', linestyle='-', label=f"Historical {p}")
        line_color = line[0].get_color()
        plt.plot(X, y_pred_historical, color='gray', linestyle='--', label=f'Regression Curve (Fit, D={degree})')

        # שרטוט ה-X בערך ה"כפוי" (0 או הערך החיובי)
        plt.plot([target_year], [plotted_value], marker='X', color=line_color, markersize=10, linestyle='None',
                 label=f"Predicted {p} ({plotted_value:.2f})")  # עדכון הליבל לערך הסופי

    # 5. הגדרות גרף סופיות
    plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Physical Zero')

    current_ymin, current_ymax = plt.ylim()
    min_val = min(all_plot_values) if all_plot_values else 0

    # עדכנו את הלוגיקה כאן כדי לוודא שהציר התחתון הוא 0 או פחות
    new_ymin = min(current_ymin, min_val * 1.1)
    new_ymin = min(new_ymin, 0)  # ודא שציר 0 תמיד כלול

    plt.ylim(bottom=new_ymin - 1, top=(current_ymax * 1.1) + 1)  # -1 קטן כדי למנוע חיתוך

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
    print("\n--- Global Pollution Prediction (Polynomial Degree 2) ---")

    # 1. הכנת נתונים
    global_pollution = pf.groupby("year")[pollution].mean().dropna()

    # 2. קביעת שנת יעד
    if target_year is None:
        last_year = global_pollution.index.max()
        target_year = last_year + 1

    print(f"Predicting pollution levels for {target_year} based on historical data...")

    # 3. קריאה לפונקציה המשותפת
    _perform_prediction_and_plot(global_pollution, target_year,
                                 f"Global Pollution Prediction (Polynomial D=2) for {target_year}")


def f6_predict_country_pollution(country, city="all", target_year=None):
    """
    פונקציית Wrapper: מכינה נתונים למדינה וקוראת לפונקציית החיזוי.
    """
    print(f"\n--- Country Pollution Prediction (Polynomial Degree 2) for: {country} ({city}) ---")

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
                                 f"Prediction for {country} - {city} (Polynomial D=2) for {target_year}")


if __name__ == "__main__":

    f5_predict_global_pollution()
    f5_predict_global_pollution(target_year=2030)

    print("-" * 30)

    try:
        f6_predict_country_pollution("Israel", "tel aviv")
        f6_predict_country_pollution("Germany", target_year=2030)
        f6_predict_country_pollution("India", target_year=2035)

    except CountryNotFoundException as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")