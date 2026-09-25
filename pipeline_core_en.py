"""
pipeline_core.py  -  Gerçek Tespit Motoru v2
=============================================
v1'e göre değişiklikler:
  * Hizalama artık YILDIZ EŞLEŞTIRMESIYLE ölçülüyor ve gerçek kayma raporlanıyor
    (phase-correlation v1'de 0 döndürdü; kareler zaten ortak grid'de olabilir).
  * YILDIZ KALINTISI REDDI: medyan referansta parlak yıldız olan konumdaki
    fark-tespitleri atılıyor -> yıldız kalıntısı yanlış pozitifleri temizlenir.
  * WZ53 PROBU: bilinen seed'den 4 karedeki tahmini konum hesaplanıp fark
    görüntüsündeki SNR ölçülüyor -> asteroit eşiğin üstünde mi görüyoruz.
  * Sönük kaynakları koru (max_per_frame büyütüldü, yıldız temizliği yükü azalttı).

Kurulum:  pip install numpy scipy scikit-image astropy photutils
Çalıştırma:  python pipeline_core.py
"""

import itertools
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

try:
    from astropy.time import Time
    HAS_TIME = True
except Exception:
    HAS_TIME = False

from photutils.detection import DAOStarFinder
from scipy.ndimage import shift as nd_shift
from scipy.spatial import cKDTree


DEFAULT_FILES = ["frame1.fits", "frame2.fits", "frame3.fits", "frame4.fits"]
DEFAULT_PIXSCALE = 0.256

# WZ53 bilinen seed konumları (HAM piksel), doğrulama için:
#   frame1.fits (en geç): (2011.9, 1738.0)
#   frame2.fits         : (2005.6, 1709.6)
WZ53_SEED = {
    "frame1.fits": (2011.9282, 1738.0714),
    "frame2.fits": (2005.6062, 1709.6221),
}


# ----------------------------------------------------------------------
# OKUMA + KRONOLOJIK SIRALAMA
# ----------------------------------------------------------------------
def _read_one(path):
    with fits.open(path) as h:
        idx = 0
        for i, hdu in enumerate(h):
            d = getattr(hdu, "data", None)
            if d is not None and np.ndim(np.squeeze(d)) == 2:
                idx = i
                break
        header = h[idx].header
        data = np.squeeze(h[idx].data).astype(float)
    mjd = header.get("MJD-OBS")
    if mjd is None and HAS_TIME and header.get("DATE-OBS"):
        try:
            mjd = Time(header["DATE-OBS"]).mjd
        except Exception:
            mjd = None
    cdelt = header.get("CDELT1")
    pixscale = abs(float(cdelt)) * 3600.0 if cdelt else DEFAULT_PIXSCALE
    return {"path": str(path), "name": Path(path).name, "data": data,
            "mjd": float(mjd) if mjd is not None else None,
            "pixscale": pixscale, "header": header}


def load_frames(paths, log):
    frames = [_read_one(p) for p in paths]
    have_time = all(f["mjd"] is not None for f in frames)
    if have_time:
        frames.sort(key=lambda f: f["mjd"])
        log.append("Frames sorted chronologically by MJD (old -> new):")
    else:
        log.append("WARNING: no MJD, using filename order.")
    t0 = frames[0]["mjd"] if have_time else None
    for i, f in enumerate(frames):
        f["t_index"] = i
        f["dt_min"] = (f["mjd"] - t0) * 24 * 60 if have_time else 0.0
        log.append(f"   [{i}] {f['name']}  (+{f['dt_min']:.2f} dk)")
    return frames


# ----------------------------------------------------------------------
# YILDIZ TESPITI (hizalama ve kalıntı reddi için ortak yardımcı)
# ----------------------------------------------------------------------
def _cen(tbl):
    """photutils sürümleri arasında merkez sütun adları değişebilir; uygun olanı bul."""
    for xx, yy in (("xcentroid", "ycentroid"), ("x_centroid", "y_centroid"),
                   ("x_peak", "y_peak"), ("x_0", "y_0"),
                   ("x_fit", "y_fit"), ("xcenter", "ycenter")):
        if xx in tbl.colnames and yy in tbl.colnames:
            return xx, yy
    raise KeyError("Centroid columns not found: " + ",".join(tbl.colnames))


def detect_stars(data, fwhm=4.0, threshold_sigma=8.0, max_n=300):
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    srcs = finder(data - median)
    if srcs is None or not len(srcs):
        return np.empty((0, 2)), std
    srcs.sort("flux", reverse=True)
    xc, yc = _cen(srcs)
    xy = np.array([[float(r[xc]), float(r[yc])] for r in srcs[:max_n]])
    return xy, std


# ----------------------------------------------------------------------
# HIZALAMA (yıldız eşleştirmesiyle ölç + gerekiyorsa uygula)
# ----------------------------------------------------------------------
def align_frames(frames, ref_index, log, match_radius=8.0):
    ref_xy, _ = detect_stars(frames[ref_index]["data"])
    ref_tree = cKDTree(ref_xy) if len(ref_xy) else None
    log.append(f"Alignment reference: [{ref_index}] {frames[ref_index]['name']} "
               f"({len(ref_xy)} stars)")

    for i, f in enumerate(frames):
        if i == ref_index or ref_tree is None:
            f["aligned"] = f["data"].copy()
            f["shift_yx"] = (0.0, 0.0)
            continue
        xy, _ = detect_stars(f["data"])
        offsets = []
        for (x, y) in xy:
            dist, idx = ref_tree.query([x, y])
            if dist <= match_radius:
                offsets.append((ref_xy[idx][0] - x, ref_xy[idx][1] - y))
        if offsets:
            offsets = np.array(offsets)
            dx = float(np.median(offsets[:, 0]))
            dy = float(np.median(offsets[:, 1]))
            scatter = float(np.median(np.abs(offsets - np.median(offsets, axis=0))))
        else:
            dx = dy = 0.0
            scatter = -1.0
        if abs(dx) > 0.05 or abs(dy) > 0.05:
            f["aligned"] = nd_shift(f["data"], shift=(dy, dx), order=1, mode="nearest")
        else:
            f["aligned"] = f["data"].copy()
        f["shift_yx"] = (dy, dx)
        log.append(f"   [{i}] {f['name']} measured shift (dx,dy)="
                   f"({dx:+.2f},{dy:+.2f}) px  matched_stars={len(offsets)}  "
                   f"scatter={scatter:.2f}px")
    return frames


# ----------------------------------------------------------------------
# FARK GÖRÜNTÜSÜ + TESPIT + YILDIZ KALINTISI REDDI
# ----------------------------------------------------------------------
def detect_on_differences(frames, fwhm, threshold_sigma, max_per_frame,
                          star_reject_px, log):
    stack = np.median([f["aligned"] for f in frames], axis=0)
    log.append("Median reference built.")

    # parlaklık-ölçekli statik yıldız kataloğu: parlak yıldız = geniş reddetme yarıçapı
    _, smed, sstd = sigma_clipped_stats(stack, sigma=3.0)
    sfind = DAOStarFinder(fwhm=fwhm, threshold=6.0 * sstd)
    stbl = sfind(stack - smed)
    if stbl is not None and len(stbl):
        sxc, syc = _cen(stbl)
        star_xy = np.array([[float(r[sxc]), float(r[syc])] for r in stbl])
        star_flux = np.array([float(r["flux"]) for r in stbl])
        pos = star_flux[star_flux > 0]
        fmed = float(np.median(pos)) if pos.size else 1.0
        star_rad = np.clip(
            star_reject_px + 4.0 * np.log10(np.maximum(star_flux / fmed, 1.0)),
            star_reject_px, 25.0)
        star_tree = cKDTree(star_xy)
        max_rad = float(star_rad.max())
    else:
        star_xy = np.empty((0, 2)); star_rad = np.empty(0)
        star_tree = None; max_rad = star_reject_px
    log.append(f"Static star catalogue: {len(star_xy)} sources "
               f"(brightness-scaled rejection {star_reject_px:.0f}-25px).")

    detections_per_frame = []
    det_id = 0
    for f in frames:
        diff = f["aligned"] - stack
        mean, median, std = sigma_clipped_stats(diff, sigma=3.0)
        finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
        srcs = finder(diff - median)

        kept, rejected = [], 0
        if srcs is not None and len(srcs):
            xc, yc = _cen(srcs)
            srcs.sort("flux", reverse=True)
            for row in srcs:
                x = float(row[xc]); y = float(row[yc])
                if star_tree is not None:
                    hit = False
                    for si in star_tree.query_ball_point([x, y], max_rad):
                        dd = ((x - star_xy[si][0]) ** 2
                              + (y - star_xy[si][1]) ** 2) ** 0.5
                        if dd <= star_rad[si]:
                            hit = True
                            break
                    if hit:
                        rejected += 1
                        continue
                # gerçek mover'ın konumu medyan yığında BOŞtur (arka plan);
                # yıldız/parıltı kalıntısı yığında da parlaktır -> reddet
                iy, ix = int(round(y)), int(round(x))
                if 0 <= iy < stack.shape[0] and 0 <= ix < stack.shape[1]:
                    win = stack[max(0, iy-2):iy+3, max(0, ix-2):ix+3]
                    if win.size and float(np.max(win)) > smed + 6.0 * sstd:
                        rejected += 1
                        continue
                # chip-gap / kenar reddi: gerçek gökyüzü gürültülüdür; chip boşluğu
                # medyan yığında DÜZ bir platodur (yerel std ~0). Böyle bir bölgenin
                # kenarındaki tespit chip-gap sızıntısıdır -> reddet.
                iy, ix = int(round(y)), int(round(x))
                H, W = stack.shape
                if ix < 8 or iy < 8 or ix > W - 8 or iy > H - 8:
                    rejected += 1
                    continue
                nb = stack[max(0, iy-6):iy+7, max(0, ix-6):ix+7]
                if nb.size:
                    flat = np.mean(np.abs(nb - np.median(nb)) < 1e-6)
                    if flat > 0.25:          # komşuluğun %25'i tıpatıp aynı = plato
                        rejected += 1
                        continue
                peak = float(row["peak"])
                kept.append({
                    "id": det_id, "t_index": f["t_index"], "path": f["path"],
                    "name": f["name"], "mjd": f["mjd"], "dt_min": f["dt_min"],
                    "x": x, "y": y, "flux": float(row["flux"]), "peak": peak,
                    "snr": float(peak / std) if std > 0 else 0.0,
                    "sharpness": float(row["sharpness"]),
                    "roundness1": float(row["roundness1"]),
                    "roundness2": float(row["roundness2"]),
                    "sigma": float(std),
                })
                det_id += 1
                if len(kept) >= max_per_frame:
                    break
        detections_per_frame.append(kept)
        log.append(f"   [{f['t_index']}] {f['name']}: {len(kept)} aday "
                   f"({rejected} star residuals rejected, std={std:.2f})")
    return detections_per_frame, stack


# ----------------------------------------------------------------------
# WZ53 PROBU: bilinen konumda fark-SNR ölç
# ----------------------------------------------------------------------
def probe_known_target(frames, stack, seed, log, box=6):
    name_to_frame = {f["name"]: f for f in frames}
    have = [n for n in seed if n in name_to_frame]
    if len(have) < 2:
        log.append("Seed probe: no seed frames configured.")
        return
    (n1, n2) = have[:2]
    f1, f2 = name_to_frame[n1], name_to_frame[n2]
    (x1, y1), (x2, y2) = seed[n1], seed[n2]
    dti = f1["t_index"] - f2["t_index"]
    if dti == 0:
        return
    vx = (x1 - x2) / dti
    vy = (y1 - y2) / dti

    log.append("\nWZ53 PROBU (bilinen konumda fark-SNR):")
    for f in frames:
        k = f["t_index"]
        px = x2 + vx * (k - f2["t_index"])
        py = y2 + vy * (k - f2["t_index"])
        diff = f["aligned"] - stack
        _, _, std = sigma_clipped_stats(diff, sigma=3.0)
        xi, yi = int(round(px)), int(round(py))
        y0, y1b = max(0, yi - box), min(diff.shape[0], yi + box + 1)
        x0, x1b = max(0, xi - box), min(diff.shape[1], xi + box + 1)
        patch = diff[y0:y1b, x0:x1b]
        if patch.size:
            peak = float(np.max(patch))
            snr = peak / std if std > 0 else 0.0
            log.append(f"   kare[{k}] {f['name']}: tahmini ({px:.1f},{py:.1f}) "
                       f"fark-SNR={snr:.1f}")
        else:
            log.append(f"   frame[{k}] {f['name']}: position out of frame.")


# ----------------------------------------------------------------------
# TRACKLET LINKING
# ----------------------------------------------------------------------
def _predict(p_i, p_j, t_i, t_j, t):
    frac = 0.0 if t_j == t_i else (t - t_i) / (t_j - t_i)
    return (p_i[0] + (p_j[0] - p_i[0]) * frac,
            p_i[1] + (p_j[1] - p_i[1]) * frac)


def _line_rmse(points_xy):
    coords = np.array(points_xy, dtype=float)
    if len(coords) < 3:
        return 0.0
    c = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(c, full_matrices=False)
    proj = c @ vh[0]
    dist = np.sqrt(((c - np.outer(proj, vh[0])) ** 2).sum(axis=1))
    return float(np.sqrt(np.mean(dist ** 2)))


def link_tracklets(detections_per_frame, frames, pixscale,
                   match_tol_px=5.0, min_step_px=2.0, max_step_px=140.0,
                   min_frames=3, log=None):
    n = len(frames)
    times = [f["mjd"] if f["mjd"] is not None else f["t_index"] for f in frames]
    trees = [cKDTree(np.array([[d["x"], d["y"]] for d in dets])) if dets else None
             for dets in detections_per_frame]
    cands = {}
    for i, j in itertools.combinations(range(min(3, n)), 2):
        for di in detections_per_frame[i]:
            for dj in detections_per_frame[j]:
                step = np.hypot(dj["x"] - di["x"], dj["y"] - di["y"]) / (j - i)
                if step < min_step_px or step > max_step_px:
                    continue
                p_i, p_j = (di["x"], di["y"]), (dj["x"], dj["y"])
                matched = {i: di, j: dj}
                for k in range(n):
                    if k in (i, j) or trees[k] is None:
                        continue
                    px, py = _predict(p_i, p_j, times[i], times[j], times[k])
                    dist, idx = trees[k].query([px, py])
                    if dist <= match_tol_px:
                        matched[k] = detections_per_frame[k][idx]
                if len(matched) < min_frames:
                    continue
                key = tuple(sorted(d["id"] for d in matched.values()))
                if key in cands:
                    continue
                order = sorted(matched.keys())
                pts = [(matched[k]["x"], matched[k]["y"]) for k in order]
                rmse = _line_rmse(pts)
                steps = [np.hypot(matched[b]["x"] - matched[a]["x"],
                                  matched[b]["y"] - matched[a]["y"]) / (b - a)
                         for a, b in zip(order[:-1], order[1:])]
                step_std = float(np.std(steps)) if steps else 0.0
                step_mean = float(np.mean(steps)) if steps else 0.0
                avg_snr = float(np.mean([matched[k]["snr"] for k in order]))
                rate = None
                if frames[0]["mjd"] is not None:
                    a, b = order[0], order[-1]
                    disp = np.hypot(matched[b]["x"] - matched[a]["x"],
                                    matched[b]["y"] - matched[a]["y"])
                    dtm = (times[b] - times[a]) * 24 * 60
                    rate = disp * pixscale / dtm if dtm > 0 else None
                line_score = 1.0 / (1.0 + rmse / 3.0)
                vel_score = 1.0 / (1.0 + step_std / (step_mean + 1e-6))
                snr_score = min(max(avg_snr / 8.0, 0.0), 1.0)
                score = 100.0 * (0.30 * line_score + 0.30 * vel_score
                                 + 0.25 * snr_score + 0.15 * len(matched) / n)
                cands[key] = {"frames_matched": len(matched),
                              "detections": {k: matched[k] for k in order},
                              "order": order, "line_rmse": rmse,
                              "step_mean_px": step_mean, "step_std_px": step_std,
                              "avg_snr": avg_snr, "rate_arcsec_min": rate,
                              "score": score}
    ranked = sorted(cands.values(), key=lambda c: c["score"], reverse=True)
    if log is not None:
        log.append(f"\nTracklet linking: {len(ranked)} candidates generated.")
    return ranked


def run_detection_pipeline(fits_files=None, fwhm=4.0, threshold_sigma=5.0,
                           max_per_frame=150, match_tol_px=5.0,
                           star_reject_px=6.0, target_name="Aday",
                           apply_filters=True):
    log = []
    if fits_files is None:
        fits_files = DEFAULT_FILES
    frames = load_frames(fits_files, log)
    pixscale = frames[0]["pixscale"]
    ref_index = len(frames) // 2
    align_frames(frames, ref_index, log)
    detections_per_frame, stack = detect_on_differences(
        frames, fwhm, threshold_sigma, max_per_frame, star_reject_px, log)
    probe_known_target(frames, stack, WZ53_SEED, log)
    candidates = link_tracklets(detections_per_frame, frames, pixscale,
                                match_tol_px=match_tol_px, log=log)
    if apply_filters:
        candidates = dedup_candidates(candidates)
        candidates = reject_detector_fixed(candidates, frames)
        candidates = reject_weak_tracklets(candidates)
        candidates = reject_common_motion(candidates)
        if log is not None:
            log.append(f"After dedup + artefact/noise rejection: "
                       f"{len(candidates)} unique candidates.")
    return {"target_name": target_name, "frames": frames,
            "detections_per_frame": detections_per_frame,
            "candidates": candidates, "pixscale": pixscale,
            "ref_index": ref_index,
            "log": "\n".join(log)}


# ----------------------------------------------------------------------
# AI ÖZELLIK ÇIKARICI (eğitim ve canlı arayüz ORTAK kullanır)
# ----------------------------------------------------------------------
FEATURE_ORDER = [
    "frames_matched", "line_rmse", "vel_consistency",
    "snr_mean", "snr_cv", "sharp_mean", "round_abs_mean",
    "flux_cv", "rate_arcsec_min", "min_edge_dist",
]


def extract_candidate_features(candidate, image_shape):
    """Bir tracklet adayından real/bogus sınıflandırması için özellik vektörü.
    Yıldız kalıntısı tespitinin kalbi: snr_cv (parlaklık tutarsızlığı) ve
    round_abs_mean (eliptiklik) yüksekse büyük ihtimalle bogus."""
    import numpy as _np
    dets = [candidate["detections"][k] for k in candidate["order"]]
    snr = _np.array([d["snr"] for d in dets], dtype=float)
    flux = _np.array([d["flux"] for d in dets], dtype=float)
    sharp = _np.array([d["sharpness"] for d in dets], dtype=float)
    r1 = _np.array([abs(d["roundness1"]) for d in dets], dtype=float)
    r2 = _np.array([abs(d["roundness2"]) for d in dets], dtype=float)
    xs = _np.array([d["x"] for d in dets], dtype=float)
    ys = _np.array([d["y"] for d in dets], dtype=float)
    H, W = image_shape
    eps = 1e-6
    edge = float(min(xs.min(), ys.min(), W - 1 - xs.max(), H - 1 - ys.max()))
    feats = {
        "frames_matched": float(candidate["frames_matched"]),
        "line_rmse": float(candidate["line_rmse"]),
        "vel_consistency": float(candidate["step_std_px"] / (candidate["step_mean_px"] + eps)),
        "snr_mean": float(snr.mean()),
        "snr_cv": float(snr.std() / (snr.mean() + eps)),
        "sharp_mean": float(sharp.mean()),
        "round_abs_mean": float((r1.mean() + r2.mean()) / 2.0),
        "flux_cv": float(flux.std() / (abs(flux.mean()) + eps)),
        "rate_arcsec_min": float(candidate.get("rate_arcsec_min") or 0.0),
        "min_edge_dist": edge,
    }
    return feats


def features_to_vector(feats):
    return [feats[k] for k in FEATURE_ORDER]


# ----------------------------------------------------------------------
# IASC RESMI KABUL KRITERLERI (Doğru/Yanlış Sinyal Kılavuzu)
# ----------------------------------------------------------------------
# Kılavuzdaki eşikler:
LINE_RMSE_MAX = 1.5     # px  -> "must move along a straight line"
VEL_CV_MAX = 0.25       #      -> "must move at constant velocity"
MAG_CHANGE_MAX = 1.0    # kadir-> "brightness must not change by more than 1 mag"
SNR_MIN = 5.0           #      -> "rejected if SNR is below 5.0"
ROUND_MAX = 0.7         #      -> "yuvarlak/eliptik" (aşırı eliptik değil)


def evaluate_criteria(candidate, image_shape, pixscale=DEFAULT_PIXSCALE):
    """Bir adayı IASC Doğru/Yanlış Sinyal Kılavuzu'nun 5 kriterine göre denetler.
    Her kriter için {geçti, değer, eşik} döndürür + genel karar.
    Kararı AÇIKLANABILIR yapar: neden kabul/ret edildiği tek tek görülür."""
    import numpy as _np
    order = candidate["order"]
    dets = [candidate["detections"][k] for k in order]
    flux = _np.array([max(d["flux"], 1e-6) for d in dets], dtype=float)
    snr = _np.array([d["snr"] for d in dets], dtype=float)
    r1 = _np.array([abs(d["roundness1"]) for d in dets], dtype=float)
    r2 = _np.array([abs(d["roundness2"]) for d in dets], dtype=float)
    xs = _np.array([d["x"] for d in dets]); ys = _np.array([d["y"] for d in dets])

    # 1) doğrusallık
    line_rmse = float(candidate["line_rmse"])
    c_line = line_rmse <= LINE_RMSE_MAX

    # 2) sabit hız
    vel_cv = float(candidate["step_std_px"] / (candidate["step_mean_px"] + 1e-6))
    c_vel = vel_cv <= VEL_CV_MAX

    # 3) sabit parlaklık (<1 kadir)
    mag_change = float(2.5 * _np.log10(flux.max() / flux.min()))
    c_bright = mag_change <= MAG_CHANGE_MAX

    # 4) SNR >= 5 (en zayıf kare bile)
    min_snr = float(snr.min())
    c_snr = min_snr >= SNR_MIN

    # 5) yuvarlak/eliptik şekil
    round_abs = float((r1.mean() + r2.mean()) / 2.0)
    c_shape = round_abs <= ROUND_MAX

    # hareket yönü (piksel çerçevesinde; gerçek PA için WCS gerekir)
    dx = xs[-1] - xs[0]; dy = ys[-1] - ys[0]
    pa_pixel = float(_np.degrees(_np.arctan2(dx, dy)) % 360)
    rate = candidate.get("rate_arcsec_min")

    checks = {
        "duz_cizgi": {"gecti": c_line, "deger": round(line_rmse, 3),
                      "esik": LINE_RMSE_MAX, "ad": "Linear track (straight line)"},
        "sabit_hiz": {"gecti": c_vel, "deger": round(vel_cv, 3),
                      "esik": VEL_CV_MAX, "ad": "Constant velocity"},
        "sabit_parlaklik": {"gecti": c_bright, "deger": round(mag_change, 3),
                            "esik": MAG_CHANGE_MAX, "ad": "Constant brightness (<1 mag)"},
        "snr": {"gecti": c_snr, "deger": round(min_snr, 1),
                "esik": SNR_MIN, "ad": "SNR >= 5"},
        "sekil": {"gecti": c_shape, "deger": round(round_abs, 3),
                  "esik": ROUND_MAX, "ad": "Round / point-like shape"},
    }
    overall = all(v["gecti"] for v in checks.values())
    return {
        "checks": checks,
        "overall": overall,
        "rate_arcsec_min": rate,
        "pa_pixel_deg": round(pa_pixel, 1),
        "n_failed": sum(1 for v in checks.values() if not v["gecti"]),
    }


# ----------------------------------------------------------------------
# ADAY TEKILLEŞTIRME (aynı iz -> tek aday)
# ----------------------------------------------------------------------
def dedup_candidates(cands, min_shared=2, pos_px=7.0):
    """Aynı nesnenin farklı kare-altkümeleriyle tekrar sayılmasını önler.
    İki aday >=2 ORTAK TESPIT paylaşıyorsa (ya da orta konumları çok yakınsa)
    aynı nesnedir; en çok kareli / en yüksek skorlu sürüm tutulur."""
    import numpy as _np
    ranked = sorted(cands, key=lambda d: (d["frames_matched"], d["score"]),
                    reverse=True)
    kept, kept_ids, kept_pos = [], [], []
    for c in ranked:
        order = c["order"]
        ids = set(c["detections"][k]["id"] for k in order)
        mx = float(_np.mean([c["detections"][k]["x"] for k in order]))
        my = float(_np.mean([c["detections"][k]["y"] for k in order]))
        dup = False
        for s, (px, py) in zip(kept_ids, kept_pos):
            if (len(ids & s) >= min_shared
                    or (abs(mx - px) < pos_px and abs(my - py) < pos_px)):
                dup = True
                break
        if not dup:
            kept.append(c); kept_ids.append(ids); kept_pos.append((mx, my))
    return sorted(kept, key=lambda d: d["score"], reverse=True)


# ----------------------------------------------------------------------
# DEDEKTÖR-SABIT ARTEFAKT REDDI (sıcak piksel/bozuk kolon -> sahte "yavaş mover")
# ----------------------------------------------------------------------
def reject_detector_fixed(cands, frames, max_spread_px=3.0):
    """Hizalama sonrası sahte hareket eden dedektör hatalarını eler.
    Adayın konumu HAM dedektör koordinatına çevrilir (hizalama kayması geri alınır);
    ham koordinatta neredeyse hiç yer değiştirmiyorsa (sabit piksel) -> reddet.
    Gerçek asteroit gökyüzünde hareket ettiği için ham koordinatta da yayılır."""
    shifts = {f["t_index"]: f.get("shift_yx", (0.0, 0.0)) for f in frames}
    kept = []
    for c in cands:
        rx, ry = [], []
        for k in c["order"]:
            dy, dx = shifts.get(k, (0.0, 0.0))
            rx.append(c["detections"][k]["x"] - dx)   # ham = hizalı - kayma
            ry.append(c["detections"][k]["y"] - dy)
        spread = max(max(rx) - min(rx), max(ry) - min(ry))
        if spread > max_spread_px:   # ham koordinatta hareket var -> gerçek
            kept.append(c)
    return kept


# ----------------------------------------------------------------------
# ZAYIF İZ REDDI (az noktalı + düşük SNR gürültü izleri)
# ----------------------------------------------------------------------
def reject_weak_tracklets(cands, snr3_min=8.0):
    """Az tespitli izler daha az güvenilirdir -> makul bir SNR tabanı iste.
    4/4 kare: IASC SNR>=5 yeterli. 3/4 kare: en zayıf kare SNR>=snr3_min.
    Eşik, gerçek keşifleri (SNR~11) korurken saf gürültü izlerini (SNR 5-6)
    eleyecek şekilde seçildi. Keşif aracı olduğundan fazla agresif değil;
    son karar IASC kriterleri + insan onayı."""
    kept = []
    for c in cands:
        min_snr = min(c["detections"][k]["snr"] for k in c["order"])
        if c["frames_matched"] >= 4:
            kept.append(c)
        elif min_snr >= snr3_min:
            kept.append(c)
    return kept


if __name__ == "__main__":
    result = run_detection_pipeline()
    print(result["log"])
    print("\n" + "=" * 70)
    print("CANDIDATE LIST (by score, cleaned)")
    print("=" * 70)
    if not result["candidates"]:
        print("Aday yok. threshold_sigma=4.0, match_tol_px=7.0 dene.")
    else:
        for rank, c in enumerate(result["candidates"][:8], start=1):
            rate = c["rate_arcsec_min"]
            rate_s = f"{rate:.3f} arcsec/dk" if rate is not None else "—"
            print(f"\n#{rank} skor={c['score']:.1f} "
                  f"kare={c['frames_matched']}/{len(result['frames'])} "
                  f"rmse={c['line_rmse']:.2f}px "
                  f"hiz={c['step_mean_px']:.1f}px/kare ({rate_s}) "
                  f"avgSNR={c['avg_snr']:.1f}")
            for k in c["order"]:
                d = c["detections"][k]
                print(f"     kare[{k}] {d['name']}: x={d['x']:.1f} y={d['y']:.1f} "
                      f"snr={d['snr']:.1f}")


# ----------------------------------------------------------------------
# ASTEROIT OLMA İHTİMALİ (keşif skoru; eleme değil, sıralama için)
# ----------------------------------------------------------------------
def asteroid_likelihood(candidate, image_shape, pixscale=DEFAULT_PIXSCALE,
                        ai_prob=None):
    """Adayı elemeden 0-100 arası 'asteroit olma ihtimali' verir.
    Bileşenler: IASC kriter geçme oranı + hareket kalitesi + SNR (+ varsa AI).
    Amaç keşif: hiçbir aday atılmaz, sadece en olasıdan sıralanır."""
    import numpy as _np
    crit = evaluate_criteria(candidate, image_shape, pixscale)
    n_pass = sum(1 for v in crit["checks"].values() if v["gecti"])
    crit_frac = n_pass / len(crit["checks"])          # 0..1

    order = candidate["order"]
    snr = _np.array([candidate["detections"][k]["snr"] for k in order], float)
    snr_score = float(_np.clip(_np.mean(snr) / 20.0, 0, 1))
    line_score = 1.0 / (1.0 + candidate["line_rmse"] / 2.0)
    vel_cv = candidate["step_std_px"] / (candidate["step_mean_px"] + 1e-6)
    vel_score = 1.0 / (1.0 + vel_cv)
    frames_score = candidate["frames_matched"] / 4.0

    # ağırlıklar: kriterler baskın, sonra hareket kalitesi, sonra AI/SNR
    base = (0.45 * crit_frac + 0.20 * line_score + 0.15 * vel_score
            + 0.10 * frames_score + 0.10 * snr_score)
    if ai_prob is not None:
        base = 0.75 * base + 0.25 * float(ai_prob)     # AI varsa harmanla
    # bir IASC kriteri bile geçemezse yüksek ihtimal veremeyiz (artefaktları alta iter)
    if n_pass < len(crit["checks"]):
        base = min(base, 0.45 - 0.08 * (len(crit["checks"]) - n_pass))
    return round(100.0 * max(base, 0.0), 1), crit


# ----------------------------------------------------------------------
# ORTAK-HAREKET REDDİ (mis-registration: tüm alan aynı vektörle kayar)
# ----------------------------------------------------------------------
def reject_common_motion(cands, min_cluster=5, vtol=1.0):
    """Çok sayıda aday AYNI hız vektörünü paylaşıyorsa, bu gerçek asteroit değil
    hizalama kaymasıdır (alan topluca kayar). O kümeyi eler. Gerçek asteroit,
    kalabalıktan farklı bir vektörle hareket ettiği için korunur."""
    import numpy as _np
    if len(cands) < min_cluster:
        return cands
    vecs = []
    for c in cands:
        order = c["order"]
        a, b = order[0], order[-1]
        span = (b - a) or 1
        vx = (c["detections"][b]["x"] - c["detections"][a]["x"]) / span
        vy = (c["detections"][b]["y"] - c["detections"][a]["y"]) / span
        vecs.append((vx, vy))
    vecs = _np.array(vecs)
    keep = _np.ones(len(cands), bool)
    for i in range(len(cands)):
        d = _np.hypot(vecs[:, 0] - vecs[i, 0], vecs[:, 1] - vecs[i, 1])
        cluster = _np.where(d < vtol)[0]
        if len(cluster) >= min_cluster:
            keep[cluster] = False
    return [c for c, k in zip(cands, keep) if k]
