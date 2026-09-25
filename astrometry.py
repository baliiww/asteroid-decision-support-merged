"""
astrometry.py  -  Aday -> RA/Dec, hız, PA, MPC-benzeri ölçüm satırları
======================================================================
Referans karenin (hizalama referansı) onarılmış WCS'ini kullanır.
Hizalanmış koordinatlar referans grid'inde olduğundan, referans WCS ile
RA/Dec'e çevrilir (yıldız hizalaması sayesinde tüm epoklar tutarlıdır).
"""

import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


def repair_wcs(header):
    """Tekil CD/PC yerine CRVAL+CRPIX+CDELT+CROTA2 ile temiz TAN WCS kurar."""
    try:
        w = WCS(naxis=2)
        w.wcs.crpix = [header["CRPIX1"], header["CRPIX2"]]
        w.wcs.crval = [header["CRVAL1"], header["CRVAL2"]]
        cd1 = header["CDELT1"]; cd2 = header["CDELT2"]
        w.wcs.ctype = [header.get("CTYPE1", "RA---TAN"),
                       header.get("CTYPE2", "DEC--TAN")]
        crota2 = float(header.get("CROTA2", header.get("CROTA1", 0.0)))
        th = np.radians(crota2)
        w.wcs.cd = [[cd1 * np.cos(th), -cd2 * np.sin(th)],
                    [cd1 * np.sin(th),  cd2 * np.cos(th)]]
        if not w.has_celestial:
            return None
        return w
    except Exception:
        return None


def _fmt(ra_deg, dec_deg):
    sc = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    ra = sc.ra.to_string(unit=u.hour, sep=" ", precision=2, pad=True)
    dec = sc.dec.to_string(unit=u.deg, sep=" ", precision=1, pad=True,
                           alwayssign=True)
    return ra, dec


def candidate_astrometry(candidate, wcs):
    """Adayın hizalanmış piksel izini RA/Dec'e çevirir; hız + PA + MPC satırları."""
    if wcs is None:
        return None
    order = candidate["order"]
    epochs = []
    for k in order:
        d = candidate["detections"][k]
        sky = wcs.pixel_to_world(d["x"], d["y"])
        ra, dec = float(sky.ra.deg), float(sky.dec.deg)
        rah, decd = _fmt(ra, dec)
        epochs.append({"frame": k, "name": d["name"], "mjd": d["mjd"],
                       "ra_deg": ra, "dec_deg": dec, "ra_hms": rah,
                       "dec_dms": decd, "snr": d["snr"]})

    # kronolojik sırala
    epochs.sort(key=lambda e: (e["mjd"] if e["mjd"] is not None else e["frame"]))
    rate = pa = sep = None
    if len(epochs) >= 2 and epochs[0]["mjd"] is not None:
        c1 = SkyCoord(epochs[0]["ra_deg"] * u.deg, epochs[0]["dec_deg"] * u.deg)
        c2 = SkyCoord(epochs[-1]["ra_deg"] * u.deg, epochs[-1]["dec_deg"] * u.deg)
        sep = float(c1.separation(c2).arcsec)
        pa = float(c1.position_angle(c2).to(u.deg).value)
        dt_min = (epochs[-1]["mjd"] - epochs[0]["mjd"]) * 24 * 60
        rate = sep / dt_min if dt_min else None

    mpc_lines = [f"MJD {e['mjd']:.5f}  RA {e['ra_hms']}  Dec {e['dec_dms']}"
                 for e in epochs]
    return {"epochs": epochs, "rate_arcsec_min": rate, "pa_deg": pa,
            "sep_arcsec": sep, "mpc_lines": mpc_lines}
