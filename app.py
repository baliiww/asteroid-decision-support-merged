"""
app_v2.py  -  Entegre Arayüz (gerçek tespit + IASC kriterleri + canlı AI)
=========================================================================
Akış: 4 FITS yükle -> Analizi Başlat -> motor seed'siz aday bulur ->
her aday IASC kriterlerine göre denetlenir -> (varsa) AI real/bogus modeli
olasılık verir -> aday listesi + kriter paneli + blink GIF + insan onayı.

Orijinal app.py'ye dokunmaz (yedek olarak durur).

Kurulum:  pip install gradio  (+ pipeline_core bağımlılıkları)
Çalıştırma:  python app_v2.py
AI modeli varsa (realbogus_model.joblib) otomatik yüklenir; yoksa kriter-temelli
çalışır.
"""

import warnings
warnings.filterwarnings("ignore")
try:
    import spaces  # HuggingFace ZeroGPU
except Exception:  # yerelde spaces yoksa no-op dekoratör
    class _S:
        def GPU(self, *a, **k):
            def deco(f): return f
            return deco
    spaces = _S()

import tempfile
import shutil
from pathlib import Path

import numpy as np
import gradio as gr
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import pipeline_core_en as pc
import astrometry
from tracker_core import run_tracking_pipeline

SAMPLE_ASSET_SPACE = "balimyabaci/asteroid-decision-support-system"
MODEL_ASSET_SPACE = "omertugrulbayram/asteroid-decision-support"


def ensure_space_asset(repo_id, filename):
    """Use a local asset when present; otherwise fetch the public file from its source Space."""
    local = Path(filename)
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="space")
    shutil.copy2(cached, local)
    return str(local)


# --- AI modeli (opsiyonel) ---
# Constants + asset helper must be defined before model loading.
AI = None
AI_LOAD_ERROR = None
try:
    import joblib
    model_path = ensure_space_asset(MODEL_ASSET_SPACE, "realbogus_model.joblib")
    AI = joblib.load(model_path)
except Exception as exc:
    AI_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
    AI = None


TITLE = "AI-Assisted Decision Support System for Asteroid Search Processes"


def ai_prob(cand, shape):
    if AI is None:
        return None
    feats = pc.extract_candidate_features(cand, shape)
    vec = [feats[k] for k in AI["features"]]
    try:
        return float(AI["model"].predict_proba([vec])[0, 1])
    except Exception:
        return None


def _norm(img):
    img = np.nan_to_num(np.asarray(img, float))
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((img - lo) / (hi - lo), 0, 1)


def render_blink(frames, records, out_dir):
    """Kronolojik karelerde adayları işaretle -> blink GIF + PNG'ler.
    Yeşil=KABUL, kırmızı=RET."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pngs = []
    for f in frames:
        k = f["t_index"]
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(_norm(f["aligned"]), cmap="gray", origin="lower")
        for rec in records:
            if k in rec["points"]:
                x, y = rec["points"][k]
                lk = rec["likelihood"]
                col = "#22c55e" if lk >= 60 else ("#facc15" if lk >= 35 else "#ef4444")
                ax.add_patch(plt.Circle((x, y), 20, edgecolor=col,
                                        facecolor="none", linewidth=2.2))
                ax.text(x + 24, y + 24, f"#{rec['rank']}", color=col,
                        fontsize=9, weight="bold")
        ax.set_title(f"Frame {k}  ({f['name']})")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        p = out_dir / f"blink_{k}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        pngs.append(str(p))
    gif = out_dir / "blink.gif"
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    if imgs:
        size = imgs[0].size
        imgs = [im.resize(size) for im in imgs]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=650, loop=0)
    return str(gif)


def render_candidate_stamp(frames, rec, out_dir, half=50):
    """Seçili adayın 4 karedeki YAKIN PLAN blink'i (IASC 'Verify Object' gibi).
    Aday her karede halka ile işaretlenir; hareketi yakından görünür."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pts = rec["points"]
    cx = float(np.mean([p[0] for p in pts.values()]))
    cy = float(np.mean([p[1] for p in pts.values()]))
    pngs = []
    for f in frames:
        k = f["t_index"]
        data = f["aligned"]
        h, w = data.shape
        x0, x1 = int(round(cx - half)), int(round(cx + half))
        y0, y1 = int(round(cy - half)), int(round(cy + half))
        xs0, xs1 = max(0, x0), min(w, x1)
        ys0, ys1 = max(0, y0), min(h, y1)
        cut = data[ys0:ys1, xs0:xs1]
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        if cut.size:
            ax.imshow(_norm(cut), cmap="gray", origin="lower",
                      extent=[xs0, xs1, ys0, ys1])
        if k in pts:
            x, y = pts[k]
            ax.add_patch(plt.Circle((x, y), 8, edgecolor="#4de1ff",
                                    facecolor="none", linewidth=2))
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_title(f"Frame {k}", fontsize=9, color="#333")
        ax.axis("off")
        p = out_dir / f"stamp_{rec['rank']}_{k}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
        pngs.append(str(p))
    gif = out_dir / f"stamp_{rec['rank']}.gif"
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    if imgs:
        size = imgs[0].size
        imgs = [im.resize(size) for im in imgs]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=600, loop=0)
    return str(gif)


def render_motion_trace(rec, out_dir):
    """Candidate path across frames (motion trace map)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(rec["points"].keys())
    xs = [rec["points"][k][0] for k in ks]; ys = [rec["points"][k][1] for k in ks]
    fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a")
    ax.plot(xs, ys, "-", color="#ff5a5a", lw=2, zorder=1)
    ax.scatter(xs, ys, s=150, facecolor="#ff5a5a", edgecolor="white", lw=2, zorder=2)
    for k, x, y in zip(ks, xs, ys):
        ax.annotate(f"F{k}  X{x:.1f} Y{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(10, 8), color="#e9ecff", fontsize=8)
    ax.set_title("Motion Trace Map", color="#e9ecff", fontweight="bold")
    ax.set_xlabel("X", color="#9aa4c8"); ax.set_ylabel("Y", color="#9aa4c8")
    ax.tick_params(colors="#9aa4c8")
    for sp in ax.spines.values(): sp.set_color("#26304a")
    ax.grid(color="#1a2340", lw=0.6)
    p = out_dir / f"trace_{rec['rank']}.png"
    fig.savefig(p, dpi=120, facecolor="#0a0e1a", bbox_inches="tight"); plt.close(fig)
    return str(p)


def render_coord_chart(rec, out_dir):
    """X and Y pixel coordinate vs frame index."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(rec["points"].keys())
    xs = [rec["points"][k][0] for k in ks]; ys = [rec["points"][k][1] for k in ks]
    fig, ax = plt.subplots(figsize=(5.6, 4.6), facecolor="white")
    ax.plot(ks, xs, "-o", color="#1f77b4", label="X coordinate")
    ax.plot(ks, ys, "-o", color="#ff7f0e", label="Y coordinate")
    ax.set_title("Coordinate Change Across Frames", fontweight="bold")
    ax.set_xlabel("Frame"); ax.set_ylabel("Pixel Coordinate")
    ax.legend(); ax.grid(color="#dddddd", lw=0.6)
    p = out_dir / f"chart_{rec['rank']}.png"
    fig.savefig(p, dpi=120, facecolor="white", bbox_inches="tight"); plt.close(fig)
    return str(p)


def criteria_html(rec):
    crit = rec["crit"]
    rows = ""
    for c in crit["checks"].values():
        mark = "✓" if c["gecti"] else "✗"
        cls = "ok" if c["gecti"] else "no"
        rows += (f"<div class='crit-row {cls}'><span class='mk'>{mark}</span>"
                 f"<span class='nm'>{c['ad']}</span>"
                 f"<span class='vl'>{c['deger']} <em>(threshold {c['esik']})</em></span></div>")
    prob = rec["prob"]
    prob_s = f"{prob*100:.1f}%" if prob is not None else "no model"
    lk = rec["likelihood"]
    if lk >= 60:
        vcls, label = "high", "High asteroid likelihood"
    elif lk >= 35:
        vcls, label = "mid", "Medium asteroid likelihood"
    else:
        vcls, label = "low", "Low asteroid likelihood"
    n_pass = sum(1 for c in crit["checks"].values() if c["gecti"])
    if crit["overall"]:
        tag_html = ("<div class='tag pass'>✓ ASTEROID CANDIDATE — 5/5 criteria passed</div>")
    else:
        tag_html = ("<div class='tag review'>⚠ Review — "
                    f"{n_pass}/5 criteria passed (a candidate must pass all "
                    "five)</div>")

    astro = rec.get("astro")
    if astro and astro.get("rate_arcsec_min") is not None:
        rate_s = f"{astro['rate_arcsec_min']:.3f}\"/min"
        pa_s = f"{astro['pa_deg']:.1f}° (N→E)"
    else:
        rate = crit["rate_arcsec_min"]
        rate_s = f"{rate:.3f}\"/min" if rate is not None else "—"
        pa_s = f"{crit['pa_pixel_deg']}° (pixel)"

    astro_html = ""
    if astro and astro.get("epochs"):
        e0, e1 = astro["epochs"][0], astro["epochs"][-1]
        astro_html = (f"<div class='astro'>Sky coordinates (RA/Dec):<br>"
                      f"first: <b>{e0['ra_hms']} &nbsp; {e0['dec_dms']}</b><br>"
                      f"last: <b>{e1['ra_hms']} &nbsp; {e1['dec_dms']}</b></div>")

    catalog_html = ""
    if rec.get("known_status") == "on":
        if rec.get("known"):
            catalog_html = (f"<div class='catalog known'>Catalogue check: "
                            f"<b>{rec['known']}</b> — known asteroid</div>")
        elif crit["overall"]:
            catalog_html = ("<div class='catalog new'>Catalogue check: not catalogued "
                            "→ <b>potential new candidate</b></div>")
        else:
            catalog_html = ("<div class='catalog known'>Catalogue check: not catalogued, "
                            "but fails criteria → likely artefact</div>")

    return f"""
<div class="crit-card">
  {tag_html}
  <div class="verdict {vcls}">Asteroid likelihood: {lk:.0f}% — {label}</div>
  <div class="meta">Criteria: <b>{n_pass}/5 passed</b> &nbsp;|&nbsp; Rate: <b>{rate_s}</b>
      &nbsp;|&nbsp; PA: <b>{pa_s}</b>
      &nbsp;|&nbsp; Frames: <b>{rec['cand']['frames_matched']}/4</b>
      &nbsp;|&nbsp; ML real/bogus: <b>{prob_s}</b></div>
  <div class="crit-list">{rows}</div>
  {astro_html}
  {catalog_html}
  <p class="note">No candidate is discarded; candidates are ranked by asteroid likelihood.
     Criteria follow the IASC True/False Signal Guide; the final decision rests with a human.</p>
</div>
"""


def coord_rows(rec):
    out = []
    for k in rec["order"]:
        d = rec["cand"]["detections"][k]
        out.append([k, d["name"], round(d["x"], 2), round(d["y"], 2),
                    round(d["snr"], 1), round(d["flux"], 1)])
    return out



def render_reference_motion_trace(points, out_dir):
    """Render the reference-guided 2024 WZ53 motion trail used only in sample mode."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]
    ks = list(range(len(points)))
    fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a")
    ax.plot(xs, ys, "-", color="#ff5a5a", lw=2, zorder=1)
    ax.scatter(xs, ys, s=150, facecolor="#ff5a5a", edgecolor="white", lw=2, zorder=2)
    for k, x, y in zip(ks, xs, ys):
        ax.annotate(f"F{k}  X{x:.1f} Y{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(10, 8), color="#e9ecff", fontsize=8)
    ax.set_title("2024 WZ53 — Motion Trace Map", color="#e9ecff", fontweight="bold")
    ax.set_xlabel("X", color="#9aa4c8"); ax.set_ylabel("Y", color="#9aa4c8")
    ax.tick_params(colors="#9aa4c8")
    for sp in ax.spines.values(): sp.set_color("#26304a")
    ax.grid(color="#1a2340", lw=0.6)
    p = out_dir / "wz53_motion_trace.png"
    fig.savefig(p, dpi=120, facecolor="#0a0e1a", bbox_inches="tight"); plt.close(fig)
    return str(p)


def render_reference_coord_chart(points, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ks = list(range(len(points)))
    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]
    fig, ax = plt.subplots(figsize=(5.6, 4.6), facecolor="white")
    ax.plot(ks, xs, "-o", label="X coordinate")
    ax.plot(ks, ys, "-o", label="Y coordinate")
    ax.set_title("2024 WZ53 — Coordinate Change Across Frames", fontweight="bold")
    ax.set_xlabel("Frame"); ax.set_ylabel("Pixel Coordinate")
    ax.legend(); ax.grid(color="#dddddd", lw=0.6)
    p = out_dir / "wz53_coordinate_chart.png"
    fig.savefig(p, dpi=120, facecolor="white", bbox_inches="tight"); plt.close(fig)
    return str(p)


def build_reference_demo(paths, out_dir):
    """Old-project WZ53 reference workflow. It is visual/verification context, not discovery."""
    ref_dir = Path(out_dir) / "wz53_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref = run_tracking_pipeline(
        fits_files=paths,
        seed_path="seed_config.json",
        output_dir=str(ref_dir),
    )
    points = ref["points"]
    motion = render_reference_motion_trace(points, ref_dir)
    chart = render_reference_coord_chart(points, ref_dir)
    coords = []
    reference_map = {}
    for i, (path, p) in enumerate(zip(paths, points)):
        name = Path(path).name
        reference_map[name] = (float(p["x"]), float(p["y"]))
        coords.append([i, name, round(float(p["x"]), 2), round(float(p["y"]), 2),
                       round(float(p.get("snr", 0.0)), 1), round(float(p.get("flux", 0.0)), 1)])
    m = ref.get("metrics", {})
    info = f"""
    <div class='crit-card reference-card'>
      <div class='tag pass'>REFERENCE DEMO — 2024 WZ53</div>
      <div class='verdict high'>Known sample target tracked across all four FITS frames</div>
      <div class='meta'>Reference-guided visualization: the known seed is used only for the sample demonstration.
      The seedless discovery + IASC + RandomForest pipeline below remains independent.</div>
      <div class='crit-list'>
        <div class='crit-row ok'><span class='mk'>✓</span><span class='nm'>Mean SNR</span><span class='vl'>{m.get('avg_snr', 0):.2f}</span></div>
        <div class='crit-row ok'><span class='mk'>✓</span><span class='nm'>Line RMSE</span><span class='vl'>{m.get('line_fit_rmse', 0):.3f} px</span></div>
        <div class='crit-row ok'><span class='mk'>✓</span><span class='nm'>Mean step</span><span class='vl'>{m.get('avg_speed', 0):.2f} px/frame</span></div>
      </div>
      <p class='note'>This panel reproduces the original 2024 WZ53 demo story: marked frames, blink, motion trace and coordinate change.</p>
    </div>
    """
    return {
        "gif": ref["gif_path"], "coords": coords, "motion": motion, "chart": chart,
        "info": info, "reference_map": reference_map, "log": ref.get("log", "")
    }


def cache_aligned_frames(frames, out_dir):
    cache_dir = Path(out_dir) / "aligned_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for f in frames:
        p = cache_dir / f"aligned_{f['t_index']}.npy"
        np.save(p, np.asarray(f["aligned"], dtype=np.float32))
        meta.append({"t_index": int(f["t_index"]), "name": f["name"], "cache_path": str(p)})
    return meta


def load_cached_frames(meta):
    frames = []
    for m in meta:
        frames.append({"t_index": m["t_index"], "name": m["name"],
                       "aligned": np.load(m["cache_path"], mmap_mode="r")})
    return frames


def match_reference_candidate(records, frames, reference_map, max_median_px=30.0):
    """Match the reference WZ53 path to a seedless candidate only after discovery."""
    if not reference_map or not records:
        return None, None
    frame_by_name = {f["name"]: f for f in frames}
    best_rank, best_med = None, None
    for r in records:
        ds = []
        for k in r["order"]:
            d = r["cand"]["detections"][k]
            name = d["name"]
            if name not in reference_map or name not in frame_by_name:
                continue
            rx, ry = reference_map[name]
            dy, dx = frame_by_name[name].get("shift_yx", (0.0, 0.0))
            rx_aligned, ry_aligned = rx + dx, ry + dy
            ds.append(float(np.hypot(d["x"] - rx_aligned, d["y"] - ry_aligned)))
        if len(ds) >= 3:
            med = float(np.median(ds))
            if best_med is None or med < best_med:
                best_med, best_rank = med, r["rank"]
    if best_med is None or best_med > max_median_px:
        return None, best_med
    return best_rank, best_med


def render_selected_assets(rec, state):
    rank = rec["rank"]
    out_dir = Path(state["out_dir"])
    stamp = out_dir / "stamps" / f"stamp_{rank}.gif"
    motion = out_dir / "traces" / f"trace_{rank}.png"
    chart = out_dir / "charts" / f"chart_{rank}.png"
    if not stamp.exists():
        frames = load_cached_frames(state.get("frame_cache", []))
        render_candidate_stamp(frames, rec, out_dir / "stamps")
    if not motion.exists():
        render_motion_trace(rec, out_dir / "traces")
    if not chart.exists():
        render_coord_chart(rec, out_dir / "charts")
    return str(stamp) if stamp.exists() else None, str(motion) if motion.exists() else None, str(chart) if chart.exists() else None

def known_object_labels(records, wcs, ref):
    """Keşiften SONRA: her adayı SkyBoT kataloğuyla kıyasla.
    Dönen: {rank: bilinen_isim veya None}, durum ('on'/'off').
    Keşfi etkilemez; sadece 'yeni mi bilinen mi' bilgisi ekler."""
    labels = {r["rank"]: None for r in records}
    if wcs is None or ref.get("mjd") is None:
        return labels, "off"
    try:
        from astroquery.imcce import Skybot
        from astropy.time import Time
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        center = wcs.pixel_to_world(ref["data"].shape[1] / 2,
                                    ref["data"].shape[0] / 2)
        sb = Skybot.cone_search(center, 0.2 * u.deg,
                                Time(ref["mjd"], format="mjd"), location="F51")
        known = SkyCoord(sb["RA"], sb["DEC"])
        for r in records:
            astro = r.get("astro")
            if not astro or not astro["epochs"]:
                continue
            ep = min(astro["epochs"],
                     key=lambda e: abs((e["mjd"] or ref["mjd"]) - ref["mjd"]))
            cc = SkyCoord(ep["ra_deg"] * u.deg, ep["dec_deg"] * u.deg)
            seps = cc.separation(known).arcsec
            m = int(np.argmin(seps))
            if seps[m] <= 15.0:
                labels[r["rank"]] = str(sb["Name"][m])
        return labels, "on"
    except Exception:
        return labels, "off"



def analyze_core(files, reference_map=None, out_dir=None):
    if not files or len(files) != 4:
        return (None, gr.update(choices=[], value=None),
                "<div class='crit-card'>4 FITS files required.</div>",
                [], None, None, None, "Upload exactly 4 frames.", {}, None)

    paths = [f if isinstance(f, str) else f.name for f in files]
    result = pc.run_detection_pipeline(fits_files=paths)
    frames = result["frames"]
    shape = frames[0]["data"].shape
    pixscale = result["pixscale"]

    records = []
    ref_hdr = result["frames"][result["ref_index"]]["header"]
    wcs = astrometry.repair_wcs(ref_hdr)
    for cand in result["candidates"]:
        prob = ai_prob(cand, shape)
        lk, crit = pc.asteroid_likelihood(cand, shape, pixscale, ai_prob=prob)
        points = {k: (cand["detections"][k]["x"], cand["detections"][k]["y"])
                  for k in cand["order"]}
        astro = astrometry.candidate_astrometry(cand, wcs)
        records.append({"cand": cand, "crit": crit, "prob": prob,
                        "likelihood": lk, "points": points,
                        "order": cand["order"], "astro": astro})

    records.sort(key=lambda r: r["likelihood"], reverse=True)
    records = records[:30]
    for i, r in enumerate(records, start=1):
        r["rank"] = i

    ref_frame = result["frames"][result["ref_index"]]
    known_map, known_status = known_object_labels(records, wcs, ref_frame)
    for r in records:
        r["known"] = known_map.get(r["rank"])
        r["known_status"] = known_status

    matched_rank, matched_distance = match_reference_candidate(records, frames, reference_map)

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="ast_merged_")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    gif = render_blink(frames, records, Path(out_dir) / "vis")
    frame_cache = cache_aligned_frames(frames, out_dir)

    choices, choice_by_rank = [], {}
    for r in records:
        rate = r["crit"]["rate_arcsec_min"]
        rate_s = f"{rate:.2f}\"/min" if rate else "—"
        mark = "✓ " if r["crit"]["overall"] else ""
        wz = "★ 2024 WZ53 · " if r["rank"] == matched_rank else ""
        label = f"{wz}{mark}Candidate #{r['rank']} — {r['likelihood']:.0f}% ({rate_s})"
        choices.append(label); choice_by_rank[r["rank"]] = label

    high = [r for r in records if r["likelihood"] >= 60]
    new_high = [r for r in high if r.get("known_status") == "on" and not r.get("known") and r["crit"]["overall"]]
    cat_note = ""
    if records and records[0].get("known_status") == "on":
        cat_note = f" Catalogue check done; potential NEW (high-likelihood, not catalogued): {len(new_high)}."
    match_note = ""
    if reference_map:
        if matched_rank is not None:
            match_note = f" Reference 2024 WZ53 path matched seedless Candidate #{matched_rank} (median offset {matched_distance:.1f}px)."
        else:
            match_note = " Reference 2024 WZ53 demo is shown, but the seedless detector did not produce a confident path match; no other candidate is relabelled as WZ53."
    summary = (f"{len(records)} moving candidates found, ranked by asteroid likelihood "
               f"(high-likelihood: {len(high)}).{cat_note} "
               f"{'RandomForest real/bogus model active.' if AI else 'No ML model loaded (criteria-only fallback' + (f': {AI_LOAD_ERROR}' if AI_LOAD_ERROR else '') + ').'}"
               f"{match_note} No candidate is discarded; the final decision rests with a human.")

    csv_path = Path(out_dir) / "candidates.csv"
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write("rank,likelihood,catalogue,frame,file,mjd,x,y,ra_deg,dec_deg,ra_hms,dec_dms,snr,rate_arcsec_min,pa_deg,ml_real_prob\n")
        for r in records:
            astro = r.get("astro")
            emap = {e["frame"]: e for e in astro["epochs"]} if astro else {}
            rate = astro["rate_arcsec_min"] if astro else r["crit"]["rate_arcsec_min"]
            pa = astro["pa_deg"] if astro else r["crit"]["pa_pixel_deg"]
            cat = r.get("known") or ("not_catalogued" if r.get("known_status") == "on" else "")
            for k in r["order"]:
                d = r["cand"]["detections"][k]; e = emap.get(k, {})
                fp.write(f"{r['rank']},{r['likelihood']},{cat},{k},{d['name']},{d.get('mjd')},{d['x']:.2f},{d['y']:.2f},"
                         f"{e.get('ra_deg','')},{e.get('dec_deg','')},{e.get('ra_hms','')},{e.get('dec_dms','')},"
                         f"{d['snr']:.1f},{rate},{pa},{r['prob']}\n")

    state = {"out_dir": str(out_dir), "records": records, "frame_cache": frame_cache,
             "panels": {r["rank"]: (criteria_html(r), coord_rows(r)) for r in records}}

    initial_rank = matched_rank if matched_rank in choice_by_rank else (records[0]["rank"] if records else None)
    initial_choice = choice_by_rank.get(initial_rank)
    initial_rec = next((r for r in records if r["rank"] == initial_rank), None)
    if initial_rec:
        first_html = criteria_html(initial_rec); first_coords = coord_rows(initial_rec)
        first_stamp, first_motion, first_chart = render_selected_assets(initial_rec, state)
    else:
        first_html = "<div class='crit-card'>No moving candidates found.</div>"
        first_coords, first_stamp, first_motion, first_chart = [], None, None, None

    return (gif, gr.update(choices=choices, value=initial_choice), first_html, first_coords,
            first_stamp, first_motion, first_chart, summary + "\n\n" + result["log"], state, str(csv_path))


def analyze_uploaded(files):
    base = analyze_core(files)
    return base + (gr.update(visible=False), None, None, [], None, None)


def select_candidate(choice, state):
    if not state or "panels" not in state or not choice:
        return "<div class='crit-card'>Run analysis first.</div>", [], None, None, None
    try:
        rank = int(choice.split("#")[1].split(" ")[0])
    except Exception:
        return "<div class='crit-card'>Selection could not be parsed.</div>", [], None, None, None
    rec = next((r for r in state.get("records", []) if r["rank"] == rank), None)
    if rec is None:
        return "<div class='crit-card'>Candidate not found.</div>", [], None, None, None
    html, coords = state["panels"].get(rank, (criteria_html(rec), coord_rows(rec)))
    stamp, motion, chart = render_selected_assets(rec, state)
    return html, coords, stamp, motion, chart


def analyze_sample():
    try:
        paths = [ensure_space_asset(SAMPLE_ASSET_SPACE, f) for f in pc.DEFAULT_FILES]
    except Exception as exc:
        base = (None, gr.update(choices=[], value=None),
                "<div class='crit-card'>Sample FITS could not be loaded.</div>",
                [], None, None, None, f"Sample asset error: {exc}", {}, None)
        return base + (gr.update(visible=True),
                       f"<div class='crit-card'>Sample FITS download failed: {exc}</div>", None, [], None, None)

    out_dir = tempfile.mkdtemp(prefix="wz53_merged_")
    try:
        ref = build_reference_demo(paths, out_dir)
    except Exception as exc:
        ref = {"gif": None, "coords": [], "motion": None, "chart": None,
               "info": f"<div class='crit-card'>Reference demo could not be generated: {exc}</div>",
               "reference_map": {}, "log": f"Reference demo error: {exc}"}
    base = analyze_core(paths, reference_map=ref.get("reference_map"), out_dir=out_dir)
    # Put reference-demo log after the seedless log without changing its decision process.
    base = list(base)
    base[7] = base[7] + "\n\n--- 2024 WZ53 REFERENCE DEMO ---\n" + ref.get("log", "")
    return tuple(base) + (gr.update(visible=True), ref.get("info"), ref.get("gif"),
                          ref.get("coords", []), ref.get("motion"), ref.get("chart"))


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --ink:#e9ecff; --dim:#9aa4c8; --violet:#8b7bff; --cyan:#4de1ff;
  --line:rgba(139,123,255,.22); --card:rgba(18,22,44,.55); --ok:#3ddc97; --no:#ff6b8a;
}
*{ font-family:'Inter',system-ui,sans-serif; }
html,body{ height:auto !important; min-height:100% !important; overflow-y:auto !important; overflow-x:hidden !important; }
gradio-app,.gradio-container{ height:auto !important; min-height:100vh !important; max-height:none !important; overflow:visible !important; background:transparent !important; color:var(--ink) !important; }
@property --n1{ syntax:'<color>'; inherits:false; initial-value:rgba(139,123,255,.18); }
@property --n2{ syntax:'<color>'; inherits:false; initial-value:rgba(77,225,255,.12); }
@property --n3{ syntax:'<color>'; inherits:false; initial-value:rgba(124,92,255,.14); }
body{
  background:
    radial-gradient(1200px 700px at 15% -5%, var(--n1), transparent 60%),
    radial-gradient(1000px 600px at 100% 10%, var(--n2), transparent 55%),
    radial-gradient(900px 900px at 60% 120%, var(--n3), transparent 60%),
    linear-gradient(180deg,#070a16,#05060f 60%,#03040a) !important;
  animation:nebula 48s ease-in-out infinite;
}
/* otomatik dönen nebula: mor -> mavi -> camgöbeği -> aurora -> turuncu */
@keyframes nebula{
  0%,100%{ --n1:rgba(139,123,255,.18); --n2:rgba(77,225,255,.12); --n3:rgba(124,92,255,.14); }
  20%{ --n1:rgba(80,130,255,.18); --n2:rgba(120,90,255,.12); --n3:rgba(60,150,255,.14); }
  40%{ --n1:rgba(60,200,230,.17); --n2:rgba(80,160,255,.12); --n3:rgba(77,225,255,.14); }
  60%{ --n1:rgba(70,220,160,.16); --n2:rgba(60,200,200,.12); --n3:rgba(90,230,150,.13); }
  80%{ --n1:rgba(255,150,90,.15); --n2:rgba(220,110,180,.12); --n3:rgba(255,120,120,.13); }
}
@media (prefers-reduced-motion:reduce){ body{animation:none;} }
body::before{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none; opacity:.5;
  background-image:
    radial-gradient(1.2px 1.2px at 20% 30%, rgba(255,255,255,.9), transparent),
    radial-gradient(1px 1px at 70% 20%, rgba(200,220,255,.8), transparent),
    radial-gradient(1.4px 1.4px at 40% 70%, rgba(255,255,255,.7), transparent),
    radial-gradient(1px 1px at 85% 60%, rgba(180,200,255,.7), transparent),
    radial-gradient(1px 1px at 55% 45%, rgba(255,255,255,.6), transparent),
    radial-gradient(1.3px 1.3px at 10% 80%, rgba(255,255,255,.7), transparent),
    radial-gradient(1px 1px at 90% 85%, rgba(210,225,255,.7), transparent);
  background-repeat:repeat; background-size:600px 600px; animation:drift 120s linear infinite;
}
@keyframes drift{ from{background-position:0 0;} to{background-position:600px 400px;} }
@media (prefers-reduced-motion:reduce){ body::before{animation:none;} .trace .rock{animation:none;} }
.gradio-container{ max-width:1280px !important; margin:0 auto !important;
  position:relative; z-index:1; }

.hero{ text-align:center; padding:40px 16px 26px; }
.eyebrow{ font-family:'JetBrains Mono',monospace; letter-spacing:.28em; text-transform:uppercase;
  font-size:11px; color:var(--cyan); opacity:.85; margin-bottom:16px; }
.hero h1{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:33px; line-height:1.15;
  margin:0 auto; max-width:900px;
  background:linear-gradient(120deg,#ffffff,#c9c2ff 45%,#7fe9ff);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  text-shadow:0 0 40px rgba(124,92,255,.25); }
.hero p{ color:var(--dim); max-width:760px; margin:16px auto 0; font-size:15px; line-height:1.6; }
.trace{ position:relative; width:260px; height:20px; margin:24px auto 4px; }
.trace .ln{ position:absolute; top:50%; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,var(--violet),var(--cyan),transparent); }
.trace .d{ position:absolute; top:50%; width:5px; height:5px; margin:-2.5px 0 0 -2.5px;
  border-radius:50%; background:rgba(255,255,255,.5); }
.trace .d1{left:12%;} .trace .d2{left:38%;} .trace .d3{left:64%;} .trace .d4{left:90%;}
.trace .rock{ position:absolute; top:50%; width:9px; height:9px; margin:-4.5px 0 0 -4.5px;
  border-radius:50%; background:var(--cyan);
  box-shadow:0 0 12px var(--cyan),0 0 24px rgba(77,225,255,.6); animation:track 6s ease-in-out infinite; }
@keyframes track{ 0%{left:12%;} 45%{left:90%;} 55%{left:90%;} 100%{left:12%;} }
.badges{ display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-top:20px; }
.badges span{ font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--ink);
  padding:5px 11px; border-radius:999px; border:1px solid var(--line); background:rgba(139,123,255,.08); }

.crit-card{ border:1px solid var(--line); border-radius:18px; padding:20px; color:var(--ink);
  background:var(--card); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  box-shadow:0 20px 60px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.05); }
.verdict{ font-family:'Space Grotesk',sans-serif; font-size:19px; font-weight:700;
  padding:11px 15px; border-radius:12px; margin-bottom:12px; }
.verdict.accept{ background:linear-gradient(135deg,rgba(61,220,151,.20),rgba(77,225,255,.10));
  border:1px solid rgba(61,220,151,.5); color:#8ff0c4; }
.verdict.reject{ background:linear-gradient(135deg,rgba(255,107,138,.16),rgba(124,92,255,.06));
  border:1px solid rgba(255,107,138,.45); color:#ffb3c2; }
.verdict.high{ background:linear-gradient(135deg,rgba(61,220,151,.20),rgba(77,225,255,.10));
  border:1px solid rgba(61,220,151,.5); color:#8ff0c4; }
.verdict.mid{ background:linear-gradient(135deg,rgba(250,204,21,.18),rgba(124,92,255,.06));
  border:1px solid rgba(250,204,21,.5); color:#ffe08a; }
.verdict.low{ background:linear-gradient(135deg,rgba(148,163,184,.14),rgba(124,92,255,.05));
  border:1px solid rgba(148,163,184,.4); color:#cbd5e1; }
.meta{ color:var(--dim); font-size:13.5px; margin-bottom:14px; } .meta b{ color:var(--ink); }
.crit-list{ display:grid; gap:8px; }
.crit-row{ display:grid; grid-template-columns:26px 1fr auto; align-items:center;
  border:1px solid var(--line); border-radius:11px; padding:9px 13px; background:rgba(10,14,30,.5); }
.crit-row.ok .mk{ color:var(--ok); } .crit-row.no .mk{ color:var(--no); }
.crit-row .mk{ font-weight:800; font-size:17px; } .crit-row .nm{ color:var(--ink); font-size:14px; }
.crit-row .vl{ font-family:'JetBrains Mono',monospace; font-size:13px; color:var(--ink); }
.crit-row .vl em{ color:var(--dim); font-style:normal; font-size:11px; }
.astro{ margin-top:14px; padding:12px 14px; border-radius:12px; color:#dfe4ff;
  background:linear-gradient(135deg,rgba(124,92,255,.14),rgba(77,225,255,.07));
  border:1px solid var(--line); font-size:14px; font-family:'JetBrains Mono',monospace; line-height:1.7; }
.catalog{ margin-top:10px; padding:10px 14px; border-radius:12px; font-size:14px; }
.catalog.known{ background:rgba(148,163,184,.14); border:1px solid rgba(148,163,184,.4); color:#cbd5e1; }
.catalog.new{ background:linear-gradient(135deg,rgba(255,196,77,.16),rgba(61,220,151,.10));
  border:1px solid rgba(255,196,77,.5); color:#ffe0a3; }
.tag{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:15px;
  padding:9px 14px; border-radius:12px; margin-bottom:10px; }
.tag.pass{ background:rgba(61,220,151,.18); border:1px solid rgba(61,220,151,.55); color:#8ff0c4; }
.tag.review{ background:rgba(255,196,77,.14); border:1px solid rgba(255,196,77,.45); color:#ffe0a3; }
.note{ color:var(--dim); font-size:12.5px; margin-top:14px; }
.reference-card{ border-color:rgba(77,225,255,.38) !important; }

.gradio-container .block, .gradio-container .form{
  background:rgba(14,18,38,.5) !important; border:1px solid var(--line) !important;
  border-radius:14px !important; }
.gradio-container label span{ color:var(--dim) !important; }
.gradio-container h1, .gradio-container h2{ font-family:'Space Grotesk',sans-serif !important; color:var(--ink) !important; }
button.primary, .gradio-container button.primary{
  background:linear-gradient(120deg,var(--violet),var(--cyan)) !important; border:none !important;
  color:#0a0e1a !important; font-weight:700 !important; border-radius:12px !important;
  box-shadow:0 8px 24px rgba(124,92,255,.35) !important; }
button.primary:hover{ filter:brightness(1.08); transform:translateY(-1px); }
"""

with gr.Blocks(title=TITLE, theme=gr.themes.Soft(primary_hue="blue"), css=CSS) as demo:
    st = gr.State({})

    gr.HTML(f"""
    <div class="hero">
      <div class="eyebrow">IASC · IAC 2026 · Decision Support System</div>
      <h1>{TITLE}</h1>
      <div class="trace">
        <div class="ln"></div>
        <div class="d d1"></div><div class="d d2"></div><div class="d d3"></div><div class="d d4"></div>
        <div class="rock"></div>
      </div>
      <p>Finds moving objects across 4 Pan-STARRS frames without seed positions, checks them against the official IASC
      criteria{' and validates them with AI' if AI else ''}; the final decision is left to a human.</p>
      <div class="badges">
        <span>Difference Imaging</span><span>Tracklet Linking</span>
        <span>{'AI · RF' if AI else 'Rule-based'}</span><span>Astrometry</span>
      </div>
    </div>""")

    with gr.Row():
        with gr.Column(scale=1):
            files_in = gr.File(label="4 FITS frames", file_count="multiple",
                               file_types=[".fits", ".fit"], type="filepath")
            with gr.Row():
                analyze_btn = gr.Button("Run Analysis", variant="primary")
                sample_btn = gr.Button("Run 2024 WZ53 sample demo")
            csv_out = gr.DownloadButton("Download candidates (CSV)", value=None)
            summary_out = gr.Textbox(label="Summary / Log", lines=16, interactive=False)
        with gr.Column(scale=1):
            gif_out = gr.Image(label="Blink — candidates (green=high, yellow=medium, red=low likelihood)",
                               type="filepath", height=520)

    with gr.Column(visible=False) as sample_panel:
        gr.Markdown("## 2024 WZ53 Reference Demo")
        with gr.Row():
            sample_gif = gr.Image(label="2024 WZ53 — marked blink", type="filepath", height=430)
            sample_info = gr.HTML(value="")
        sample_coords = gr.Dataframe(
            headers=["frame", "file", "x", "y", "snr", "flux"],
            label="2024 WZ53 reference coordinates", interactive=False)
        with gr.Row():
            sample_motion = gr.Image(label="2024 WZ53 — Motion Trace", type="filepath", height=360)
            sample_chart = gr.Image(label="2024 WZ53 — Coordinate Change", type="filepath", height=360)

    gr.Markdown("## Candidate Evaluation (IASC criteria + AI)")
    with gr.Row():
        with gr.Column(scale=1):
            cand_sel = gr.Radio(label="Select candidate", choices=[])
            stamp_out = gr.Image(label="Selected candidate — zoomed blink (watch the motion)",
                                 type="filepath", height=300)
            coords_out = gr.Dataframe(
                headers=["frame", "file", "x", "y", "snr", "flux"],
                label="Selected candidate coordinates", interactive=False)
        with gr.Column(scale=1):
            crit_out = gr.HTML(value="<div class='crit-card'>Waiting for analysis.</div>")

    gr.Markdown("## Motion & Coordinate Analysis")
    with gr.Row():
        with gr.Column(scale=1):
            motion_out = gr.Image(label="Motion Trace Map", type="filepath", height=380)
        with gr.Column(scale=1):
            chart_out = gr.Image(label="Coordinate Chart", type="filepath", height=380)

    with gr.Accordion("Project / Team", open=False):
        gr.Markdown("""
**AI-Assisted Decision Support System for Asteroid Search Processes** — IASC/IAC 2026.
The system finds moving objects across PS1 frames using difference imaging + tracklet
linking, automatically applies the IASC True/False Signal Guide criteria, and makes
accuracy measurable with a real/bogus AI model. It does not declare confirmed
discoveries; it prioritises candidates and presents them for human verification.
""")

    common_outputs = [gif_out, cand_sel, crit_out, coords_out, stamp_out,
                      motion_out, chart_out, summary_out, st, csv_out,
                      sample_panel, sample_info, sample_gif, sample_coords, sample_motion, sample_chart]
    analyze_btn.click(analyze_uploaded, inputs=files_in, outputs=common_outputs)
    sample_btn.click(analyze_sample, inputs=[], outputs=common_outputs)
    cand_sel.change(select_candidate, inputs=[cand_sel, st],
                    outputs=[crit_out, coords_out, stamp_out, motion_out, chart_out])


if __name__ == "__main__":
    demo.launch()
