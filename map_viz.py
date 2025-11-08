from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np
import logging
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))

def calc_pixels(lat: float, lon: float, img_shape: Tuple[int, int], center_x: int = 998, center_y: int = 553, unit_x: float = 5.1, unit_y: float = 6.1) -> Tuple[int, int]:
    x = int(center_x + lon * unit_x)
    y = int(center_y - lat * unit_y)
    x = clamp(x, 0, img_shape[1] - 1)
    y = clamp(y, 0, img_shape[0] - 1)
    return x, y

def pm25_to_bgr(pm25: float, vmin: Optional[float], vmax: Optional[float], cmap_name: str, clip_percentiles=(2.5, 97.5), df_for_scale: Optional[pd.Series]=None) -> Tuple[int, int, int]:
    if not np.isfinite(pm25):
        return (200, 200, 200)
    if vmin is None or vmax is None:
        if df_for_scale is None:
            vmin, vmax = 0.0, 100.0
        else:
            lo, hi = np.nanpercentile(df_for_scale, clip_percentiles)
            vmin, vmax = float(lo), float(hi)
            if vmin == vmax:
                vmax = vmin + 1.0
    pm25_clipped = float(max(vmin, min(pm25, vmax)))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(pm25_clipped))
    r, g, b, _ = rgba
    return (int(255 * b), int(255 * g), int(255 * r))

def create_colorbar_image(vmin: float, vmax: float, cmap_name: str, width: int = 300, height: int = 50, output_path: Path = Path("colorbar.png")) -> Path:
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    fig.subplots_adjust(bottom=0.5)
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    cb1.set_label('PM2.5')
    ax.remove()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return output_path

def embed_colorbar_to_image(map_image_path: Path, colorbar_path: Path, output_path: Path, padding: int = 10) -> None:
    map_img = cv2.imread(str(map_image_path))
    cb_img = cv2.imread(str(colorbar_path))
    if map_img is None or cb_img is None:
        raise FileNotFoundError("map image or colorbar image not found")
    h_map, w_map = map_img.shape[:2]
    h_cb, w_cb = cb_img.shape[:2]
    scale = min(0.4 * w_map / w_cb, 0.12 * h_map / h_cb, 1.0)
    new_w = int(w_cb * scale)
    new_h = int(h_cb * scale)
    cb_resized = cv2.resize(cb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = w_map - new_w - padding
    y = h_map - new_h - padding
    overlay = map_img.copy()
    overlay[y:y+new_h, x:x+new_w] = cb_resized
    cv2.imwrite(str(output_path), overlay)

def plot_pm25_on_map(
    df: pd.DataFrame,
    year: int,
    map_image_path: Path,
    output_path: Path,
    pm_col: str = "pm25_concentration",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    point_radius: int = 2,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "viridis",
    colorbar_output: Path = Path("colorbar.png"),
) -> None:
    if not map_image_path.exists():
        raise FileNotFoundError(map_image_path)
    img = cv2.imread(str(map_image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {map_image_path}")
    subset = df.loc[
        (df["year"] == year)
        & df[pm_col].notna()
        & df[lat_col].notna()
        & df[lon_col].notna(),
        [pm_col, lat_col, lon_col],
    ]
    series_for_scale = df[pm_col].dropna().values if pm_col in df.columns else None
    for _, row in subset.iterrows():
        pm = float(row[pm_col])
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        x, y = calc_pixels(lat, lon, img.shape)
        color = pm25_to_bgr(pm, vmin=vmin, vmax=vmax, cmap_name=cmap_name, df_for_scale=series_for_scale)
        cv2.circle(img, (x, y), radius=point_radius, color=color, thickness=-1)
    temp_output = Path("temp_map_with_points.jpg")
    cv2.imwrite(str(temp_output), img)
    if series_for_scale is not None:
        if vmin is None or vmax is None:
            lo, hi = np.nanpercentile(series_for_scale, (2.5, 97.5))
            vmin, vmax = float(lo), float(hi)
            if vmin == vmax:
                vmax = vmin + 1.0
    create_colorbar_image(vmin, vmax, cmap_name, output_path=colorbar_output)
    embed_colorbar_to_image(temp_output, colorbar_output, output_path)
