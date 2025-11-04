import pandas as pd
import cv2
from matplotlib.image import imread

df = pd.read_excel("who_ambient_air_quality_database_version_2024_(v6.1).xlsx", sheet_name="Update 2024 (V6.1)")
data = df.loc[(df["year"] == 2020) & (df["pm25_concentration"].notna()) & (df["latitude"].notna()) & (df["longitude"].notna()), ["pm25_concentration", "latitude", "longitude"]]

print(data)

def calc_pixels(lat, lon):
    # פיקסלים אמצע הכדור: 998, 553
    # יחידה למעלה: 6.1
    # יחידה לצד: 5.1
    x = 998 + (lon * 5.1)
    y = 553 - (lat * 6.1)
    return int(x), int(y)


color_bar = cv2.imread("colorsBar.png")

def calc_color(pm25):
    return tuple(int(c) for c in color_bar[100, int(pm25)*20 if int(pm25)*40 < 1700 else 1700])


img = cv2.imread("physical-earth-map-geographic-grid-lines.png")

for x, r in data.iterrows():
    cv2.circle(img, calc_pixels(r.latitude, r.longitude), radius=2, color=calc_color(r.pm25_concentration), thickness=-1)


resized = cv2.resize(img, (int(0.5 * img.shape[1]), int(0.5 * img.shape[0])), interpolation=cv2.INTER_AREA)
cv2.imshow("Air Pollution Map 2020", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("AirPollutionMap2020.jpg", img)
print("img saved to 'AirPollutionMap2020.jpg' file in this folder")