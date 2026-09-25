"""
ai_train.py  -  Sentetik Enjeksiyon + Real/Bogus AI Eğitimi
============================================================
NE YAPAR:
  1) Gerçek 4 kareye, konumu ve parlaklığı BILINEN sahte hareketli nesneler
     (Gaussian PSF) enjekte eder.
  2) Aynı tespit boru hattını çalıştırır.
  3) Kurtarılan adayları "gerçek" (enjekte edilene uyan) / "bogus" (yıldız
     kalıntısı, rastgele) diye ETIKETLER -> yer gerçeği elde edilir.
  4) RandomForest real/bogus sınıflandırıcısı eğitir.
  5) Precision / Recall / F1 / ROC-AUC + SNR'a göre tamamlık eğrisi raporlar.
  6) Modeli kaydeder (realbogus_model.joblib) -> arayüzde CANLI kullanılır.

Bu, hem "AI destekli" iddiasının motoru hem de "doğruluk oranı" kanıtıdır.

Kurulum:  pip install scikit-learn joblib   (pipeline_core bağımlılıkları da lazım)
Çalıştırma:  python ai_train.py
   (birkaç dakika sürer; N_TRIALS / MOVERS_PER_TRIAL ile ayarlanır)
"""

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             precision_recall_fscore_support, confusion_matrix)

import pipeline_core_en as pc
from astropy.stats import sigma_clipped_stats


# ---- ayarlar ----
N_TRIALS = 20            # deneme sayısı (artır -> daha çok veri, daha yavaş)
MOVERS_PER_TRIAL = 6     # her denemede enjekte edilen sahte hareketli sayısı
PSF_SIGMA = 1.6          # sahte kaynak PSF genişliği (px); FWHM ~ 3.8px
SNR_RANGE = (3.0, 40.0)  # enjekte edilen parlaklık aralığı (SNR)
SPEED_RANGE = (5.0, 45.0)  # px/kare (asteroit-benzeri hareket aralığı)
MATCH_TRUTH_PX = 3.0     # aday enjekte edilene bu kadar yakınsa "gerçek"
RANDOM_SEED = 42


def inject_gaussian(img, x, y, amp, sigma=PSF_SIGMA):
    """Görüntüye tek bir Gaussian nokta kaynak ekler (yerel yama)."""
    h, w = img.shape
    r = int(np.ceil(4 * sigma))
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1] += amp * np.exp(
        -(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    )


def make_movers(n, shape, noise, rng):
    """n adet geçerli (kare içinde kalan) sahte hareketli üretir."""
    h, w = shape
    movers = []
    tries = 0
    while len(movers) < n and tries < n * 50:
        tries += 1
        x0 = rng.uniform(150, w - 150)
        y0 = rng.uniform(150, h - 150)
        ang = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(*SPEED_RANGE)
        vx, vy = speed * np.cos(ang), speed * np.sin(ang)
        target_snr = rng.uniform(*SNR_RANGE)
        # 4 karede de içeride mi?
        pos = [(x0 + vx * k, y0 + vy * k) for k in range(4)]
        if all(60 < px < w - 60 and 60 < py < h - 60 for px, py in pos):
            movers.append({
                "pos": pos,
                "amp": target_snr * noise,
                "snr": target_snr,
            })
    return movers


def label_candidate(cand, movers):
    """Aday bir enjekte edilene uyuyorsa (gerçek) True, yoksa False.
    Ayrıca hangi mover'a uyduğunu döndürür (tamamlık için)."""
    order = cand["order"]
    cx = np.array([cand["detections"][k]["x"] for k in order])
    cy = np.array([cand["detections"][k]["y"] for k in order])
    for mi, m in enumerate(movers):
        mx = np.array([m["pos"][k][0] for k in order])
        my = np.array([m["pos"][k][1] for k in order])
        d = np.hypot(cx - mx, cy - my)
        if np.median(d) < MATCH_TRUTH_PX:
            return True, mi
    return False, -1


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    log = []

    print("Kareler yükleniyor ve hizalanıyor...")
    frames = pc.load_frames(pc.DEFAULT_FILES, log)
    ref = len(frames) // 2
    pc.align_frames(frames, ref, log)
    base_aligned = [f["aligned"].copy() for f in frames]
    shape = base_aligned[0].shape
    pixscale = frames[0]["pixscale"]

    # gürültü tahmini (fark std ~ enjeksiyon genliği ölçeği)
    base_stack = np.median(base_aligned, axis=0)
    _, _, noise = sigma_clipped_stats(base_aligned[0] - base_stack, sigma=3.0)
    noise = float(noise) if noise > 0 else 8.0
    print(f"Referans hizalandı. Görüntü {shape}, gürültü~{noise:.2f}, "
          f"piksel ölçeği {pixscale:.3f}\"/px")
    print(f"{N_TRIALS} deneme x {MOVERS_PER_TRIAL} sahte hareketli enjekte "
          f"edilecek. Bu birkaç dakika sürebilir...\n")

    X, y = [], []
    injected_records = []  # (snr, recovered_bool)

    for trial in range(N_TRIALS):
        movers = make_movers(MOVERS_PER_TRIAL, shape, noise, rng)

        # enjekte edilmiş kareler
        injected = [base_aligned[k].copy() for k in range(len(frames))]
        for m in movers:
            for k, (px, py) in enumerate(m["pos"]):
                inject_gaussian(injected[k], px, py, m["amp"])

        # frames kopyası (aligned = injected)
        frames_inj = []
        for k, f in enumerate(frames):
            g = dict(f)
            g["aligned"] = injected[k]
            frames_inj.append(g)

        dets, _ = pc.detect_on_differences(
            frames_inj, fwhm=4.0, threshold_sigma=5.0,
            max_per_frame=150, star_reject_px=6.0, log=[])
        cands = pc.link_tracklets(dets, frames_inj, pixscale,
                                  match_tol_px=5.0, log=None)

        recovered = set()
        for c in cands:
            is_real, mi = label_candidate(c, movers)
            feats = pc.extract_candidate_features(c, shape)
            X.append(pc.features_to_vector(feats))
            y.append(1 if is_real else 0)
            if is_real:
                recovered.add(mi)

        for mi, m in enumerate(movers):
            injected_records.append((m["snr"], mi in recovered))

        n_pos = sum(1 for c in cands if label_candidate(c, movers)[0])
        print(f"  Deneme {trial+1:2d}/{N_TRIALS}: {len(cands):3d} aday "
              f"({n_pos} gerçek / {len(cands)-n_pos} bogus), "
              f"kurtarılan mover {len(recovered)}/{len(movers)}")

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    print(f"\nToplam veri: {len(y)} aday  "
          f"(gerçek={int(y.sum())}, bogus={int((1-y).sum())})")

    if y.sum() < 10 or (1 - y).sum() < 10:
        print("UYARI: Sınıf dengesizliği yüksek. N_TRIALS'i artır.")

    # eğitim / test
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\n" + "=" * 60)
    print("REAL/BOGUS SINIFLANDIRICI - TEST SONUÇLARI")
    print("=" * 60)
    print(classification_report(yte, pred, target_names=["bogus", "gerçek"]))
    try:
        print(f"ROC-AUC: {roc_auc_score(yte, proba):.4f}")
    except Exception:
        pass
    p, r, f1, _ = precision_recall_fscore_support(
        yte, pred, average="binary", zero_division=0)
    print(f"Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")
    print("Confusion matrix [satır=gerçek etiket, sütun=tahmin]:")
    print(confusion_matrix(yte, pred))

    print("\nÖzellik önemleri:")
    for name, imp in sorted(zip(pc.FEATURE_ORDER, clf.feature_importances_),
                            key=lambda t: t[1], reverse=True):
        print(f"   {name:18s} {imp:.3f}")

    # tamamlık (completeness) vs enjekte SNR
    print("\n" + "=" * 60)
    print("TAMAMLIK (completeness) - enjekte SNR'a göre")
    print("=" * 60)
    rec = np.array(injected_records, dtype=object)
    snrs = np.array([r[0] for r in injected_records])
    recs = np.array([1 if r[1] else 0 for r in injected_records])
    bins = [3, 5, 8, 12, 20, 30, 40]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (snrs >= lo) & (snrs < hi)
        if m.sum():
            print(f"   SNR {lo:2d}-{hi:2d}: {recs[m].mean()*100:5.1f}% "
                  f"({int(recs[m].sum())}/{int(m.sum())})")
    print(f"   GENEL tamamlık: {recs.mean()*100:.1f}%")

    # modeli kaydet
    joblib.dump({"model": clf, "features": pc.FEATURE_ORDER,
                 "image_shape": shape}, "realbogus_model.joblib")
    print("\nModel kaydedildi: realbogus_model.joblib "
          "(arayüzde canlı kullanılacak).")


if __name__ == "__main__":
    main()
