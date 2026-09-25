import os
import json
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import imageio.v2 as imageio


DEFAULT_FITS_FILES = [
    "frame1.fits",
    "frame2.fits",
    "frame3.fits",
    "frame4.fits"
]

DEFAULT_SEED_PATH = "seed_config.json"
DEFAULT_OUTPUT_DIR = "output"


def file_to_path(file_obj):
    """
    Hem normal dosya yolu hem de Gradio'dan gelen dosya objeleriyle çalışır.
    """
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)


def read_fits_image(path):
    """
    FITS dosyasını okur ve 2 boyutlu görüntü verisi döndürür.
    """
    data = fits.getdata(path)
    data = np.squeeze(data)

    if data.ndim != 2:
        raise ValueError(f"{path} dosyası 2 boyutlu değil. Shape: {data.shape}")

    return data.astype(float)


def robust_normalize(data):
    """
    Görüntüyü ekranda daha anlaşılır göstermek için normalize eder.
    Çok parlak yıldızların görüntüyü bozmasını azaltır.
    """
    vmin = np.percentile(data, 1)
    vmax = np.percentile(data, 99.7)

    if vmax == vmin:
        return np.zeros_like(data)

    norm = (data - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)

    return norm


def local_background_stats(data, x, y, radius=25):
    """
    Hedefin etrafındaki yerel arka planı hesaplar.
    Bu bize SNR / sigma hesabı için yardımcı olur.
    """
    h, w = data.shape

    x = int(round(x))
    y = int(round(y))

    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    patch = data[y1:y2, x1:x2]

    background = np.median(patch)

    mad = np.median(np.abs(patch - background))
    sigma = 1.4826 * mad

    if sigma <= 0:
        sigma = np.std(patch)

    if sigma <= 0:
        sigma = 1.0

    return background, sigma


def refine_centroid(data, expected_x, expected_y, search_radius=18, centroid_radius=5):
    """
    Tahmini koordinatın etrafında en uygun parlak noktayı bulur.
    Sonra centroid hesabıyla konumu daha hassas hale getirir.
    """
    h, w = data.shape

    expected_x = float(expected_x)
    expected_y = float(expected_y)

    x0 = int(round(expected_x))
    y0 = int(round(expected_y))

    sx1 = max(0, x0 - search_radius)
    sx2 = min(w, x0 + search_radius + 1)
    sy1 = max(0, y0 - search_radius)
    sy2 = min(h, y0 + search_radius + 1)

    search_patch = data[sy1:sy2, sx1:sx2]

    if search_patch.size == 0:
        return {
            "x": expected_x,
            "y": expected_y,
            "snr": 0.0,
            "flux": 0.0,
            "sigma": 0.0,
            "status": "empty_search_patch"
        }

    background, sigma = local_background_stats(data, expected_x, expected_y)

    significance = (search_patch - background) / sigma

    peak_y, peak_x = np.unravel_index(np.argmax(significance), significance.shape)

    peak_x_global = sx1 + peak_x
    peak_y_global = sy1 + peak_y

    cx1 = max(0, peak_x_global - centroid_radius)
    cx2 = min(w, peak_x_global + centroid_radius + 1)
    cy1 = max(0, peak_y_global - centroid_radius)
    cy2 = min(h, peak_y_global + centroid_radius + 1)

    centroid_patch = data[cy1:cy2, cx1:cx2]

    weights = centroid_patch - background
    weights = np.clip(weights, 0, None)

    if np.sum(weights) <= 0:
        refined_x = float(peak_x_global)
        refined_y = float(peak_y_global)
        flux = 0.0
    else:
        yy, xx = np.mgrid[cy1:cy2, cx1:cx2]
        refined_x = float(np.sum(xx * weights) / np.sum(weights))
        refined_y = float(np.sum(yy * weights) / np.sum(weights))
        flux = float(np.sum(weights))

    peak_value = data[int(round(peak_y_global)), int(round(peak_x_global))]
    snr = float((peak_value - background) / sigma)

    return {
        "x": refined_x,
        "y": refined_y,
        "snr": snr,
        "flux": flux,
        "sigma": float(sigma),
        "status": "tracked"
    }


def compute_motion_metrics(points):
    """
    Takip edilen noktaların hareket kalitesini ölçer.
    """
    coords = np.array([[p["x"], p["y"]] for p in points], dtype=float)

    diffs = np.diff(coords, axis=0)
    speeds = np.sqrt(np.sum(diffs ** 2, axis=1))

    avg_speed = float(np.mean(speeds)) if len(speeds) else 0.0
    speed_std = float(np.std(speeds)) if len(speeds) else 0.0

    angles = np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0])) if len(diffs) else np.array([])

    direction_changes = []
    for i in range(1, len(angles)):
        diff = abs(angles[i] - angles[i - 1])
        diff = min(diff, 360 - diff)
        direction_changes.append(diff)

    direction_change_mean_deg = float(np.mean(direction_changes)) if direction_changes else 0.0

    center = coords.mean(axis=0)
    centered = coords - center

    if len(coords) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        projections = centered @ direction
        closest = np.outer(projections, direction)
        distances = np.sqrt(np.sum((centered - closest) ** 2, axis=1))
        line_fit_rmse = float(np.sqrt(np.mean(distances ** 2)))
    else:
        line_fit_rmse = 0.0

    snrs = np.array([p.get("snr", 0.0) for p in points], dtype=float)
    sigmas = np.array([p.get("sigma", 0.0) for p in points], dtype=float)

    avg_snr = float(np.mean(snrs))
    min_snr = float(np.min(snrs))
    avg_sigma = float(np.mean(sigmas))

    snr_score = min(max(avg_snr / 8.0, 0.0), 1.0)
    speed_score = 1.0 / (1.0 + speed_std / (avg_speed + 1e-6))
    line_score = 1.0 / (1.0 + line_fit_rmse / 5.0)

    confidence_score = 100.0 * (
        0.40 * snr_score +
        0.30 * speed_score +
        0.30 * line_score
    )

    return {
        "avg_speed": avg_speed,
        "speed_std": speed_std,
        "direction_change_mean_deg": direction_change_mean_deg,
        "line_fit_rmse": line_fit_rmse,
        "avg_snr": avg_snr,
        "min_snr": min_snr,
        "avg_sigma": avg_sigma,
        "confidence_score": float(confidence_score)
    }







def save_annotated_frames(images, points, target_name, output_dir):
    """
    Hedefin etraf?n? k?rm?z? daireyle i?aretleyen PNG kareleri ?retir.
    Ayr?ca hareket y?n?n? k?rm?z? oklarla g?sterir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    png_paths = []

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]

    for i, image in enumerate(images):
        point = points[i]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(robust_normalize(image), cmap="gray", origin="lower")

        # Hareket izi ?izgisi
        ax.plot(xs, ys, color="red", linewidth=1.7, alpha=0.85)

        # Hareket y?n? oklar?
        for j in range(1, len(points)):
            dx = xs[j] - xs[j - 1]
            dy = ys[j] - ys[j - 1]

            ax.arrow(
                xs[j - 1],
                ys[j - 1],
                dx,
                dy,
                head_width=14,
                head_length=18,
                length_includes_head=True,
                color="red",
                linewidth=1.6,
                alpha=0.9
            )

        # Bu frame'deki hedef
        circle = plt.Circle(
            (point["x"], point["y"]),
            radius=18,
            edgecolor="red",
            facecolor="none",
            linewidth=2.5
        )

        ax.add_patch(circle)
        ax.scatter([point["x"]], [point["y"]], color="red", s=24)

        ax.text(
            point["x"] + 22,
            point["y"] + 22,
            f"{target_name} / Frame {i}",
            color="red",
            fontsize=10,
            weight="bold"
        )

        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])

        ax.set_title(f"Annotated Frame {i} - Motion Direction")
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")

        png_path = output_dir / f"annotated_frame_{i}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        png_paths.append(str(png_path))

    return png_paths


def save_blink_gif(png_paths, output_dir):
    """
    ??aretli karelerden blink GIF ?retir.
    T?m kareleri ayn? boyuta getirir.
    """
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    gif_path = output_dir / "blink_candidates.gif"

    frames = []
    target_size = None

    for path in png_paths:
        img = Image.open(path).convert("RGB")

        if target_size is None:
            target_size = img.size
        else:
            img = img.resize(target_size)

        frames.append(img)

    if not frames:
        raise ValueError("GIF ?retmek i?in PNG karesi bulunamad?.")

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=650,
        loop=0
    )

    return str(gif_path)


def save_csv(points, metrics, output_dir):
    """
    Koordinatlar? ve analiz ?zetini CSV dosyas?na kaydeder.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "selected_candidate.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["frame", "x", "y", "snr", "flux", "sigma", "status"])

        for i, point in enumerate(points):
            writer.writerow([
                i,
                point.get("x", 0.0),
                point.get("y", 0.0),
                point.get("snr", 0.0),
                point.get("flux", 0.0),
                point.get("sigma", 0.0),
                point.get("status", "")
            ])

        writer.writerow([])
        writer.writerow(["metric", "value"])

        for key, value in metrics.items():
            writer.writerow([key, value])

    return str(csv_path)


def load_seed(seed_path):
    with open(seed_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_tracking_pipeline(
    fits_files=None,
    seed_path=DEFAULT_SEED_PATH,
    output_dir=DEFAULT_OUTPUT_DIR
):
    """
    Ana takip boru hattı.
    Web sitesi de bu fonksiyonu çağıracak.
    """
    if fits_files is None:
        fits_files = DEFAULT_FITS_FILES

    fits_files = [file_to_path(f) for f in fits_files]

    if len(fits_files) != 4:
        raise ValueError("Bu demo için tam olarak 4 FITS dosyası gerekir.")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    logs = []
    logs.append("FITS dosyaları alındı.")

    seed = load_seed(seed_path)
    target_name = seed.get("target_name", "Unknown Target")

    logs.append(f"Seed config okundu. Hedef: {target_name}")

    images = [read_fits_image(path) for path in fits_files]
    logs.append("FITS görüntüleri okundu.")

    frame0_seed = seed["frame0"]
    frame1_seed = seed["frame1"]

    p0 = refine_centroid(
        images[0],
        frame0_seed["x"],
        frame0_seed["y"],
        search_radius=12,
        centroid_radius=5
    )

    p1 = refine_centroid(
        images[1],
        frame1_seed["x"],
        frame1_seed["y"],
        search_radius=12,
        centroid_radius=5
    )

    points = [p0, p1]

    logs.append("İlk iki karede seed koordinatları hassaslaştırıldı.")

    for frame_index in range(2, 4):
        previous = points[-1]
        before_previous = points[-2]

        vx = previous["x"] - before_previous["x"]
        vy = previous["y"] - before_previous["y"]

        predicted_x = previous["x"] + vx
        predicted_y = previous["y"] + vy

        tracked_point = refine_centroid(
            images[frame_index],
            predicted_x,
            predicted_y,
            search_radius=22,
            centroid_radius=5
        )

        tracked_point["predicted_x"] = predicted_x
        tracked_point["predicted_y"] = predicted_y

        points.append(tracked_point)

        logs.append(
            f"Frame {frame_index}: tahmini konumdan hedef takip edildi."
        )

    metrics = compute_motion_metrics(points)
    logs.append("Hareket metrikleri hesaplandı.")

    

    png_paths = save_annotated_frames(
        images=images,
        points=points,
        target_name=target_name,
        output_dir=output_dir
    )

    logs.append("İşaretli PNG kareleri üretildi.")

    gif_path = save_blink_gif(png_paths, output_dir)
    logs.append("Blink GIF üretildi.")

    csv_path = save_csv(points, metrics, output_dir)
    logs.append("CSV analiz dosyası oluşturuldu.")

    coordinate_table = []

    for i, p in enumerate(points):
        coordinate_table.append({
            "frame": i,
            "x": round(p["x"], 3),
            "y": round(p["y"], 3),
            "snr": round(p.get("snr", 0.0), 3),
            "flux": round(p.get("flux", 0.0), 3),
            "sigma": round(p.get("sigma", 0.0), 3),
            "status": p.get("status", "")
        })

    metric_table = [
        {"metric": key, "value": round(value, 4)}
        for key, value in metrics.items()
    ]

    return {
        "target_name": target_name,
        "points": points,
        "metrics": metrics,
        "coordinate_table": coordinate_table,
        "metric_table": metric_table,
        "gif_path": gif_path,
        "png_paths": png_paths,
        "original_png_paths": png_paths,
        "comparison_png_paths": png_paths,
        "csv_path": csv_path,
        "log": "\n".join(logs)
    }


if __name__ == "__main__":
    result = run_tracking_pipeline()

    print("\nASTROVIA tracking tamamlandı.")
    print("Hedef:", result["target_name"])

    print("\nKoordinatlar:")
    for row in result["coordinate_table"]:
        print(row)

    print("\nMetrikler:")
    for row in result["metric_table"]:
        print(row)

    print("\nGIF:", result["gif_path"])
    print("CSV:", result["csv_path"])

    print("\nLog:")
    print(result["log"])