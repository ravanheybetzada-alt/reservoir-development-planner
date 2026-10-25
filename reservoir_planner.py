#!/usr/bin/env python3
"""
reservoir_planner.py   (ONE program: parameters + contours + map + wells)     v8
======================================================================
Screening tool for a gas(-condensate) field development project.

HOW YOU USE IT
  Just run:   python reservoir_planner.py
  It asks you, step by step:
    1. which settings .json file to use (optional: costs, prices, overrides)
    2. which MAP PICTURE to use (always asked: pick from the folder or paste a path)
    3. which production-history file to use
    3b. which core-analysis file and which field-description (.docx) file to use
        (the reservoir PARAMETERS are calculated from these - see below)
    4. the MAP SCALE  -> you click the two ends of the scale bar on the picture
                         (or type pixels / metres-per-pixel if you prefer)
    5. EXISTING WELLS -> click them on the picture (or type pixel coordinates)
    6. DEPTH LABELS   -> NEW: detect the thin contour lines and READ their depth numbers
                         automatically (OCR), or click labels by hand (optional; gives the
                         gas column above the gas-water contact)
    7. platform reach and plant capacity (optional)
  then it measures the field area, computes gas in place, forecasts production,
  finds the best number of wells, places them and shows the maps.
  Use --no-ask to skip all questions (needs the options on the command line).

INPUT
  * a picture of the field map (the field must be filled with one colour,
    e.g. the green "developed gasfield" polygon; faults = thick black lines)
  * reservoir data in a JSON file (see params_example.json)
  * optional: monthly production history (.xlsx / .csv)
  * optional: depth labels read off the map + the gas-water contact depth

OUTPUT (in --outdir)
  * contours.png      (automatic contours) red = bold faults/outline, coloured = contour lines
                      with the depth that was read, grey = contour lines without a number
  * wells_map.png     map with proposed well locations, drainage circles and the
                      field area that no well reaches (purple)
  * forecast.png      production forecast for the recommended number of wells
                      + comparison of 1..N wells + NPV vs number of wells
  * history_match.png model vs. real production history (if history given)
  * forecast.csv      yearly production table
  * summary.json      all numbers of the run

WHAT IT DOES
  0. PARAMETERS (new in v7): the reservoir parameters are no longer typed into a JSON by hand.
     They are calculated from the data files and every value is printed with its source:
       porosity, permeability, net-to-gross  <- core analysis (net samples: porosity >= 8%,
                                               permeability >= 0.1 mD where measured)
       gas specific gravity                  <- gas composition table in the field description
       gas saturation                        <- handout, else 1 - BVW/porosity (Buckles)
       pressures, depth, gas price           <- handout table (hydrostatic / 25% fallbacks)
       net pay                               <- calibrated to the operator's GIIP (giip_bcm),
                                               else handout, else cores (upper limit)
       reservoir temperature                 <- surface temperature + gradient x depth
       condensate-gas ratio                  <- production history
       recovery factor                       <- reserves_bcm / giip_bcm, else 0.75
     A value in your JSON always wins; --prefer-handout puts the handout before the cores;
     --save-params FILE writes all calculated values to a JSON you can edit.
  1. Measures the field area from the coloured polygon on the map.
  2. Gas in place: volumetric  G = A*h*phi*Sg / Bg  (Bg from Hall-Yarborough Z),
     or the operator's GIIP if you give "giip_bcm" in the JSON.
  3. Production forecast: "tank" (p/Z material-balance) model. Each well
     delivers q = C*(p^2 - pwf^2); the reservoir pressure falls as gas is
     produced, so the rate declines.  C is calibrated from the initial rate
     of a well: measured from the production history if you give one,
     otherwise from the Darcy gas-deliverability equation (an UPPER bound).
  4. DRAINAGE RADIUS: from the production history. The ultimate recovery (EUR) of the
     history well is found by three decline fits (rate-vs-cumulative, exponential,
     hyperbolic), divided by the recovery factor to get the gas it drains, and divided
     by the gas in place per km2 to get its drainage area and radius. Without a
     history, give "drainage_radius_m" in the JSON (default 800 m).
  4b. CONTOURS (new in v8): contour lines are the THIN dark lines; faults, the field outline and
     bold text are the BOLD ones and are taken out first. The numbers printed along the contour
     lines are read with OCR (the picture is enlarged and turned in 30-degree steps, a number
     counts only if it is read at 2+ angles and is a multiple of the contour interval). Each
     contour line gets the number next to it, and the labelled lines give the depth surface.
     Lines that carry two different numbers are dropped. Readings are cached next to the map.
  5. WELL LOCATIONS (new in v6): wells are added one at a time, each at the spot
     whose drainage circle reaches the MOST field area that no other well already
     drains. A spot is allowed only if it is
        - far enough from the field edge (gas-water contact) and from faults,
        - has enough gas column above the contact (if depth labels were given),
        - within the horizontal reach of the platform (if a reach is given),
        - at least min_well_spacing_m from every other well.
     Existing wells are kept fixed and their drainage is subtracted first.
     Faults are treated as LEAKY by default (pressure communicates on production time scales);
     use --sealing to restrict each well to its own fault block.
     The search stops by itself when a new well would add less than min_gain_km2.
  6. Tries 1, 2, 3 ... wells in that order, computes NPV for each, and recommends the
     fewest wells whose NPV is within npv_tolerance of the maximum.

LIMITS (read before using the numbers)
  * Depth comes from the labels YOU read off the map (about +/-20-30 m); one flat
    gas-water contact is assumed.
  * Fault lines are found as thick black strokes: this suits maps like the NAM L15 map
    (200 dpi). Text in bold black can be mistaken for a fault - check wells_map.png.
  * Drainage is assumed circular; faults are not given a transmissibility.
  * Only the wells you give as existing are subtracted. Unknown existing wells will make
    the new-area numbers too optimistic.
  * One average set of reservoir properties is used for the whole area.
  * Costs, RF, temperature etc. are assumptions: change them in the JSON.

Requires: numpy pandas scipy matplotlib pillow openpyxl
Optional (only needed if the map is a .pdf): pymupdf
Optional (automatic contour numbers): python -m pip install rapidocr-onnxruntime

EXAMPLES
  python reservoir_planner.py
  python reservoir_planner.py --map L12-FC.png --params params.json --history L15-02.xlsx
  python reservoir_planner.py --map map.png --scalebar-px 64 --scalebar-m 1000 \
      --params params_example.json --history L15-02.xlsx \
      --seed-wells "256,167" --reach 3750 --outdir out --no-ask
"""
import argparse
import json
import math
import re
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

if "--no-show" in sys.argv:          # no pop-up windows, only save the files
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
if "--no-show" not in sys.argv and matplotlib.get_backend().lower() == "agg":
    for _be in ("TkAgg", "QtAgg", "Qt5Agg"):      # try to get a window system if the default has none
        try:
            plt.switch_backend(_be)
            break
        except Exception:
            continue
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator
from scipy.optimize import brentq, curve_fit, minimize_scalar
from scipy.signal import fftconvolve
from scipy.spatial.distance import cdist

P_SC_BAR = 1.01325
T_SC_K = 288.15
DAYS_PER_MONTH = 30.4375

DEFAULTS = {
    "name": "reservoir",
    "surface_temp_c": 12.0,
    "geothermal_gradient_c_per_100m": 3.0,
    "reservoir_temp_c": None,          # overrides the gradient estimate
    "gas_specific_gravity": 0.61,      # air = 1; ~0.61 for the L12/L15 gas
    "permeability_md": 100.0,          # only used for the Darcy rate
    "recovery_factor": 0.75,           # cap on recovered fraction of GIIP
    "giip_bcm": None,                  # operator's gas in place; replaces the volumetric value
    "match_history": True,             # scale the gas in place to reproduce the history
    "gas_price_usd_per_1000m3": 200.0,
    "condensate_price_usd_per_m3": 450.0,
    "cgr_m3_per_million_m3": 10.0,     # used only when there is no history
    "well_cost_musd": 30.0,            # PLACEHOLDER - replace with your data
    "fixed_capex_musd": 0.0,
    "opex_musd_per_year": 4.0,         # PLACEHOLDER
    "discount_rate": 0.10,
    "field_life_years": 25,
    "min_rate_mm3_per_day": None,      # None -> rate where revenue = opex
    "min_well_spacing_m": 500.0,
    "edge_buffer_m": 150.0,
    "npv_tolerance": 0.02,             # recommend the FEWEST wells within this fraction of the max NPV
    "rate_cap_mm3d": None,             # plant / export capacity, million m3/day (None = no limit)
    "wellbore_radius_m": 0.1,
    "skin": 0.0,
    # --- well-location method (v6) ---
    "drainage_radius_m": None,         # None -> calibrated from the history (else 800 m)
    "eur_abandon_mm3_month": 0.5,      # abandonment rate for the EUR fit, million m3 per MONTH per well
    "leaky_faults": True,              # faults leak on production time scales
    "use_faults": True,                # detect faults on the map
    "fault_min_pixels": 60,            # smallest line piece counted as a fault (200 dpi map)
    "fault_color": None,               # [R,G,B] of the fault lines; None = thick black lines
    "fault_color_tol": 80,             # colour tolerance (sum of the 3 channel differences)
    "fault_standoff_m": 120.0,         # keep wells this far from faults
    "min_gas_column_m": 80.0,          # needs depth labels + gas-water contact
    "platform_reach_m": None,          # None = no limit (first existing well = platform)
    "min_gain_km2": 0.5,               # stop when a new well adds less undrained area than this
    "wellbore_note": "",
    # --- parameter calculation (v7) ---
    "net_phi_cutoff": 0.08,            # a core sample is net pay above this porosity ...
    "net_k_cutoff_md": 0.1,            # ... and above this permeability (where measured)
    "buckles_bvw": 0.025,              # bulk volume water, only used to estimate Sg
    "pressure_gradient_bar_per_m": 0.1,  # hydrostatic, only used if no pressure is given
    "operating_pressure_fraction": 0.25,  # flowing pressure as a fraction of reservoir pressure
    "gross_thickness_m": None,         # gross reservoir thickness (only for the core-based net pay)
    "reserves_bcm": None,              # operator reserves; with giip_bcm gives the recovery factor
    # --- automatic contours (v8) ---
    "contour_interval_m": 20.0,        # spacing of the contour lines (printed on the map)
    "contour_dark_threshold": 170,     # a pixel is line-dark if its brightest channel is below this
    "contour_min_length_px": 40,       # shorter pieces are digits / specks, not contour lines
    "contour_label_max_dist_px": 30.0, # a number belongs to a line if it is this close
}
REQUIRED = ["net_pay_m", "porosity", "gas_saturation", "depth_m",
            "reservoir_pressure_bar", "operating_pressure_bar"]


# --------------------------------------------------------------------------
# gas PVT
# --------------------------------------------------------------------------
def pseudo_critical(gamma):
    """Sutton pseudo-critical properties -> (T in K, P in bar)."""
    tpc_r = 169.2 + 349.5 * gamma - 74.0 * gamma ** 2
    ppc_psia = 756.8 - 131.0 * gamma - 3.6 * gamma ** 2
    return tpc_r / 1.8, ppc_psia / 14.5038


def z_factor(p_bar, t_k, gamma):
    """Hall-Yarborough gas compressibility factor."""
    tpc, ppc = pseudo_critical(gamma)
    t = tpc / t_k                      # reciprocal reduced temperature
    pr = p_bar / ppc
    a = 0.06125 * t * math.exp(-1.2 * (1 - t) ** 2)
    b = t * (14.76 - 9.76 * t + 4.58 * t * t)
    c = t * (90.7 - 242.2 * t + 42.4 * t * t)
    d = 2.18 + 2.82 * t

    def f(y):
        return (-a * pr + (y + y ** 2 + y ** 3 - y ** 4) / (1 - y) ** 3
                - b * y ** 2 + c * y ** d)

    y = brentq(f, 1e-12, 0.95)
    return a * pr / y


def gas_viscosity_cp(p_bar, t_k, gamma, z):
    """Lee-Gonzalez-Eakin viscosity (cP)."""
    m = 28.96 * gamma
    t_r = t_k * 1.8
    p_psia = p_bar * 14.5038
    rho = 1.4935e-3 * p_psia * m / (z * t_r)          # g/cm3
    k = (9.4 + 0.02 * m) * t_r ** 1.5 / (209 + 19 * m + t_r)
    x = 3.5 + 986.0 / t_r + 0.01 * m
    y = 2.4 - 0.2 * x
    return 1e-4 * k * math.exp(x * rho ** y)


class GasPVT:
    """Table of Z and p/Z so that p can be found from p/Z quickly."""

    def __init__(self, t_k, gamma, p_max_bar):
        self.t_k, self.gamma = t_k, gamma
        self.p = np.linspace(1.0, p_max_bar * 1.05, 400)
        self.z_tab = np.array([z_factor(p, t_k, gamma) for p in self.p])
        self.pz = self.p / self.z_tab
        if not np.all(np.diff(self.pz) > 0):
            raise ValueError("p/Z is not monotonic - check temperature / gas gravity")

    def z(self, p):
        return float(np.interp(p, self.p, self.z_tab))

    def p_from_pz(self, pz):
        return float(np.interp(pz, self.pz, self.p))

    def bg(self, p):
        """Formation volume factor, res m3 / std m3."""
        return self.z(p) * P_SC_BAR * self.t_k / (p * T_SC_K)


# --------------------------------------------------------------------------
# map -> area
# --------------------------------------------------------------------------
def segment_field(rgb, field_color=None, tol=60, min_rel_size=0.02, merge_pieces=False):
    """Boolean mask of the coloured field polygon.

    field_color=None -> any 'green' pixel: catches both the bright
    "developed" green NAM uses and the pale "undeveloped" green, while
    excluding white background, black contour lines, red block-boundary
    lines and the cyan GWC line.
    Faults/contours drawn on top of the polygon are filled by closing.

    By default, only the SINGLE LARGEST matching blob is kept, so a
    neighbouring field bleeding into the picture is not merged in.
    Pass merge_pieces=True only if the field's own polygon is genuinely
    split into separate pieces that should be combined.
    """
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    if field_color is None:
        near_white = (r > 230) & (g > 230) & (b > 230)
        mask = (g > r) & (g > b) & (g - b >= 15) & (g >= 140) & ~near_white
    else:
        mask = np.abs(rgb.astype(int) - np.array(field_color)).sum(axis=2) <= tol
    raw_px = int(mask.sum())
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)), iterations=1)
    mask = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(mask)
    if n == 0:
        raise SystemExit("No field-coloured pixels found. Use --field-color R,G,B.")
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    if n > 1:
        others = sorted(sizes, reverse=True)[1:4]
        print(f"   (found {n} separate blobs of this colour; "
              f"keeping the largest ({int(sizes.max())} px) and dropping the rest "
              f"({', '.join(str(int(s)) for s in others)}{', ...' if n > 4 else ''} px) "
              f"- pass --merge-disconnected-pieces if that's wrong.)")
    if merge_pieces:
        keep = [i + 1 for i, s in enumerate(sizes) if s >= min_rel_size * sizes.max()]
    else:
        keep = [int(np.argmax(sizes)) + 1]
    mask = np.isin(lab, keep)
    return mask, raw_px


# --------------------------------------------------------------------------
# production history
# --------------------------------------------------------------------------
def _col(df, *keys):
    for c in df.columns:
        if any(k in str(c).lower() for k in keys):
            return c
    return None


def load_history(path, n_wells=1):
    path = Path(path)
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    c_date, c_type = df.columns[0], _col(df, "type")
    c_gas, c_cond = _col(df, "gas"), _col(df, "condens")
    if c_gas is None:
        raise SystemExit("History file needs a 'Gas' column.")
    if np.issubdtype(df[c_date].dtype, np.number):
        df[c_date] = pd.to_datetime(df[c_date], unit="D", origin="1899-12-30")
    df[c_date] = pd.to_datetime(df[c_date])
    if c_type is not None:
        df = df[df[c_type].astype(str).str.lower().str.startswith("prod")]
    out = pd.DataFrame({
        "date": df[c_date].dt.to_period("M").dt.to_timestamp(),
        "gas": pd.to_numeric(df[c_gas], errors="coerce").fillna(0.0),
        "cond": pd.to_numeric(df[c_cond], errors="coerce").fillna(0.0) if c_cond else 0.0,
    }).groupby("date").sum().reset_index()
    out["days"] = out["date"].dt.days_in_month
    out["q_m3d"] = out["gas"] / out["days"]
    first = out[out["gas"] > 0].head(36)
    q_w0 = first["q_m3d"].mean() / n_wells
    cgr = out["cond"].sum() / max(out["gas"].sum(), 1.0)
    return out, q_w0, cgr


def history_eur(hist, q_lim_mm3_month, n_wells=1):
    """EUR per well (million m3) from three decline fits to the production history.

    The fit window ends at the first long shut-in (3+ consecutive near-zero months),
    because the rates after a shut-in do not follow the earlier decline.
    Returns None if there are too few usable months."""
    q = hist["gas"].values / 1e6                       # million m3 per month
    t = np.arange(len(q), dtype=float)
    cum = np.cumsum(q)
    head = q[:24][q[:24] > 0]
    ref = float(np.median(head)) if len(head) else float(q.max())
    low = q < 0.1 * ref
    end, run = len(q), 0
    for i, flag in enumerate(low):
        run = run + 1 if flag else 0
        if run >= 3:
            end = i - 2
            break
    use = (~low) & (t < end)
    if use.sum() < 12:
        return None
    ql = q_lim_mm3_month
    out = {}
    try:
        slope, icpt = np.polyfit(cum[use], q[use], 1)
        if slope < 0:
            out["rate_cum"] = (ql - icpt) / slope
    except Exception:
        pass
    try:
        pe, _ = curve_fit(lambda x, qi, d: qi * np.exp(-d * x), t[use], q[use],
                          p0=[q[use][0], 0.01], maxfev=20000)
        out["exponential"] = (pe[0] - ql) / pe[1]
    except Exception:
        pass
    try:
        hyp = lambda x, qi, di, b: qi / np.power(1 + b * di * x, 1 / b)
        ph, _ = curve_fit(hyp, t[use], q[use], p0=[q[use][0], 0.02, 0.5],
                          bounds=([0.1, 1e-4, 0.01], [100, 1, 1.5]), maxfev=20000)
        qi, di, b = ph
        out["hyperbolic"] = qi ** b / (di * (1 - b)) * (qi ** (1 - b) - ql ** (1 - b))
    except Exception:
        pass
    vals = [v for v in out.values() if np.isfinite(v) and v > cum[end - 1]]
    if not vals:
        return None
    sel = float(np.mean(vals)) / n_wells
    return {"methods": {k: round(float(v) / n_wells, 1) for k, v in out.items()},
            "selected_mm3": sel, "low_mm3": float(min(vals)) / n_wells,
            "high_mm3": float(max(vals)) / n_wells,
            "months_used": int(use.sum()), "fit_ends_month": int(end)}


# --------------------------------------------------------------------------
# reservoir parameters: CALCULATED from the data files (v7)
# --------------------------------------------------------------------------
MW_AIR = 28.9647
AIR_DENSITY_KG_M3 = 1.225                     # dry air at 15 C and 1 atm
COMP_MW = {                                   # molar masses, g/mol
    "hydrogen": 2.016, "hydrogen sulphide": 34.08, "carbon dioxide": 44.01, "nitrogen": 28.014,
    "methane": 16.043, "ethane": 30.07, "propane": 44.097, "i-butane": 58.123, "n-butane": 58.123,
    "i-pentane": 72.15, "n-pentane": 72.15, "hexanes": 86.18, "mc pentane": 84.16, "benzene": 78.11,
    "cyclohexane": 84.16, "heptanes": 100.2, "mc hexane": 98.19, "toluene": 92.14, "octanes": 114.23,
    "e-benzene": 106.17, "m/p xylene": 106.17, "o-xylene": 106.17, "nonanes": 128.26,
    "1,2,4 tmb": 120.19, "decanes": 142.28, "undecanes +": 160.0,
}
HANDOUT_KEYS = (("depth_m", "depth"), ("net_pay_m", "net pay"), ("permeability_md", "permeab"),
                ("porosity", "porosity"), ("gas_saturation", "gas saturation"),
                ("condensate_saturation", "condensate saturation"), ("gas_density", "gas density"),
                ("condensate_density", "condensate density"), ("gas_price_usd_per_1000m3", "gas price"),
                ("reservoir_pressure_bar", "reservoir pressure"),
                ("operating_pressure_bar", "operating pressure"))


def _num(s):
    try:
        return float(str(s).replace(",", ".").strip())
    except ValueError:
        return None


def gas_gravity_from_composition(rows):
    """rows = [(name, component_label, mole_percent)] -> (specific gravity, molar mass g/mol)."""
    tot = mw = 0.0
    for name, comp, y in rows:
        if y is None:
            continue
        m = COMP_MW.get(str(name).strip().lower())
        if m is None:
            c = str(comp).strip().upper().replace("+", "")
            if c.startswith("C") and c[1:].isdigit():
                m = 14.0266 * int(c[1:]) + 2.0159          # n-alkane estimate
        if m is None:
            continue
        tot += y
        mw += y * m
    if tot <= 0:
        return None
    return (mw / tot) / MW_AIR, mw / tot


def read_field_description(path):
    """Read the reservoir table and the gas-composition table from the project handout (.docx)."""
    try:
        from docx import Document
    except ImportError:
        print("   (reading a .docx needs python-docx:  python -m pip install python-docx  - skipped)")
        return None
    doc = Document(str(path))
    out = {"reservoirs": {}, "gas_rows": None}
    for tb in doc.tables:
        rows = [[c.text.strip() for c in r.cells] for r in tb.rows]
        if not rows or not rows[0]:
            continue
        hdr = [h.lower() for h in rows[0]]
        if hdr[0] == "reservoir" and any("net pay" in h for h in hdr):
            for r in rows[1:]:
                if not r[0]:
                    continue
                d = {}
                for h, v in zip(hdr[1:], r[1:]):
                    key = next((k for k, kw in HANDOUT_KEYS if kw in h), None)
                    val = _num(v)
                    if key and val is not None:
                        d[key] = val / 100.0 if key in ("porosity", "gas_saturation",
                                                       "condensate_saturation") else val
                out["reservoirs"][r[0]] = d
        elif hdr[0] == "component" and len(hdr) > 2 and "name" in hdr[1] and out["gas_rows"] is None:
            out["gas_rows"] = [(r[1], r[0], _num(r[2])) for r in rows[1:]
                               if r[0].lower() != "total" and len(r) > 2]
    return out


def is_core_file(f):
    """True if a spreadsheet looks like a core analysis (has depth and porosity columns)."""
    try:
        f = Path(f)
        df = pd.read_csv(f, nrows=2) if f.suffix.lower() == ".csv" else pd.read_excel(f, nrows=2)
    except Exception:
        return False
    cols = " ".join(str(c).lower() for c in df.columns)
    return "depth" in cols and "poro" in cols


def read_core(path, phi_cut, k_cut):
    """Statistics of the core analysis. A sample is 'net' if porosity >= phi_cut and,
    where permeability was measured, permeability >= k_cut (mD)."""
    path = Path(path)
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    cd, cp, ck = _col(df, "depth"), _col(df, "poro"), _col(df, "perm")
    if cd is None or cp is None:
        raise SystemExit(f"{path.name}: needs 'Depth' and 'Porosity' columns.")
    d = pd.to_numeric(df[cd], errors="coerce")
    phi = pd.to_numeric(df[cp], errors="coerce")
    k = pd.to_numeric(df[ck], errors="coerce") if ck else pd.Series(np.nan, index=df.index)
    ok = d.notna() & phi.notna()
    d, phi, k = d[ok], phi[ok], k[ok]
    if phi.max() > 1.5:                                    # given in per cent
        phi = phi / 100.0
    net = (phi >= phi_cut) & (k.isna() | (k >= k_cut))
    kn = k[net & k.notna() & (k > 0)]
    if net.sum() == 0:
        raise SystemExit("No core sample passes the net-pay cut-offs - lower net_phi_cutoff.")
    return {"n": int(len(d)), "n_net": int(net.sum()), "ntg": float(net.mean()),
            "porosity": float(phi[net].mean()), "porosity_all": float(phi.mean()),
            "k_geo": float(np.exp(np.log(kn).mean())) if len(kn) else None,
            "k_arith": float(kn.mean()) if len(kn) else None, "n_k": int(len(kn)),
            "depth_m": float(d[net].mean()), "top_m": float(d.min()), "bottom_m": float(d.max()),
            "cored_m": float(d.max() - d.min()), "phi_cut": phi_cut, "k_cut": k_cut}


class ParamTable:
    """Collects every parameter with the way it was obtained, so nothing is a black box."""

    def __init__(self):
        self.rows = []

    def add(self, name, value, source, note="", alt=None):
        self.rows.append({"parameter": name, "value": value, "source": source,
                          "how": note, "other_sources": alt or {}})

    def show(self):
        print("\nPARAMETERS - value | source | how it was obtained   "
              "((!) = another source differs by more than 25%)")
        for r in self.rows:
            v = r["value"]
            vs = f"{v:.4g}" if isinstance(v, float) else str(v)
            flag = ""
            alts = []
            for s, av in r["other_sources"].items():
                alts.append(f"{s}: {av:.4g}")
                if isinstance(v, (int, float)) and v and abs(av - v) / abs(v) > 0.25:
                    flag = " (!)"
            print(f"  {r['parameter']:<26} {vs:>10} | {r['source']:<14} | {r['how']}"
                  + (f"   [others: {', '.join(alts)}]{flag}" if alts else ""))

    def as_list(self):
        return [{**r, "value": (round(r["value"], 5) if isinstance(r["value"], float) else r["value"])}
                for r in self.rows]


def _choose(tab, name, order, vals, notes=None):
    for src in order:
        v = vals.get(src)
        if v is not None:
            alt = {s: vals[s] for s in order if s != src and vals.get(s) is not None}
            tab.add(name, v, src, (notes or {}).get(src, ""), alt)
            return v
    return None


def resolve_params(prm, hints, core, hand, prefer_hand, tab):
    """Fill prm with the reservoir parameters. Order of preference for each one:
    your JSON value > value calculated from the data files > value in the handout > assumption.
    (--prefer-handout puts the handout before the core calculation.)"""
    J = {k: v for k, v in hints.items() if not k.startswith("_")}
    H, C = hand or {}, core or {}

    def go(name, order, vals, notes=None):
        v = _choose(tab, name, order, vals, notes)
        if v is not None:
            prm[name] = v
        return prm.get(name)

    core_note = (f"mean of {C.get('n_net')} of {C.get('n')} core samples with porosity >= "
                 f"{C.get('phi_cut', 0):.0%}" if C else "")
    ordc = ["json", "handout", "core"] if prefer_hand else ["json", "core", "handout"]
    phi = go("porosity", ordc, {"json": J.get("porosity"), "core": C.get("porosity"),
                                "handout": H.get("porosity")}, {"core": core_note})
    go("permeability_md", ordc, {"json": J.get("permeability_md"), "core": C.get("k_geo"),
                                 "handout": H.get("permeability_md")},
       {"core": f"geometric mean of {C.get('n_k')} net samples with measured permeability"})
    depth = go("depth_m", ["json", "handout", "core"],
               {"json": J.get("depth_m"), "handout": H.get("depth_m"), "core": C.get("depth_m")},
               {"core": "mean depth of the net core samples"})
    sg_b = (min(0.9, max(0.5, 1 - prm["buckles_bvw"] / phi)) if phi else None)
    go("gas_saturation", ["json", "handout", "buckles"],
       {"json": J.get("gas_saturation"), "handout": H.get("gas_saturation"), "buckles": sg_b},
       {"buckles": f"1 - BVW/porosity with bulk-volume-water {prm['buckles_bvw']} (typical sandstone)"})
    hyd = depth * prm["pressure_gradient_bar_per_m"] if depth else None
    pi = go("reservoir_pressure_bar", ["json", "handout", "hydrostatic"],
            {"json": J.get("reservoir_pressure_bar"), "handout": H.get("reservoir_pressure_bar"),
             "hydrostatic": hyd},
            {"hydrostatic": f"depth x {prm['pressure_gradient_bar_per_m']} bar/m"})
    frac = prm["operating_pressure_fraction"] * pi if pi else None
    ordo = ["json", "handout", "fraction"] if prefer_hand else ["json", "fraction", "handout"]
    go("operating_pressure_bar", ordo,
       {"json": J.get("operating_pressure_bar"), "handout": H.get("operating_pressure_bar"),
        "fraction": frac},
       {"fraction": f"{prm['operating_pressure_fraction']:.0%} of reservoir pressure - the flowing "
                    "pressure the tank model needs; the handout value is the INITIAL operating pressure"})
    gg = gas_gravity_from_composition(H["gas_rows"]) if H.get("gas_rows") else None
    go("gas_specific_gravity", ["json", "composition", "handout_density", "default"],
       {"json": J.get("gas_specific_gravity"), "composition": gg[0] if gg else None,
        "handout_density": (H["gas_density"] / AIR_DENSITY_KG_M3 if H.get("gas_density") else None),
        "default": DEFAULTS["gas_specific_gravity"]},
       {"composition": (f"molar mass {gg[1]:.2f} g/mol from the gas analysis / {MW_AIR}" if gg else ""),
        "handout_density": "handout gas density / 1.225 kg/m3 (air)"})
    go("gas_price_usd_per_1000m3", ["json", "handout", "default"],
       {"json": J.get("gas_price_usd_per_1000m3"), "handout": H.get("gas_price_usd_per_1000m3"),
        "default": DEFAULTS["gas_price_usd_per_1000m3"]})
    rf_res = (J["reserves_bcm"] / J["giip_bcm"] if J.get("reserves_bcm") and J.get("giip_bcm") else None)
    go("recovery_factor", ["json", "reserves", "default"],
       {"json": J.get("recovery_factor"), "reserves": rf_res, "default": DEFAULTS["recovery_factor"]},
       {"reserves": "reserves_bcm / giip_bcm"})
    if depth:                                                         # reservoir temperature
        t_c = (prm["reservoir_temp_c"] if prm["reservoir_temp_c"] is not None else
               prm["surface_temp_c"] + prm["geothermal_gradient_c_per_100m"] * depth / 100.0)
        tab.add("reservoir_temp_c", round(t_c, 1), "json" if prm["reservoir_temp_c"] is not None else "calculated",
                "" if prm["reservoir_temp_c"] is not None else
                f"{prm['surface_temp_c']:g} C + {prm['geothermal_gradient_c_per_100m']:g} C/100 m x depth")


def resolve_net_pay(prm, hints, hand, core, area_m2, bg, tab, prefer_hand):
    """Net pay is resolved after the map has been measured, because it can be calibrated
    to a known gas in place:  h = GIIP x Bg / (area x porosity x Sg)."""
    J = {k: v for k, v in hints.items() if not k.startswith("_")}
    H, C = hand or {}, core or {}
    giip_np = None
    if prm.get("giip_bcm"):
        giip_np = prm["giip_bcm"] * 1e9 * bg / (area_m2 * prm["porosity"] * prm["gas_saturation"])
    core_np = None
    if C:
        gross = prm.get("gross_thickness_m") or C["cored_m"]
        core_np = C["ntg"] * gross
    order = ["json", "handout", "giip", "core"] if prefer_hand else ["json", "giip", "handout", "core"]
    v = _choose(tab, "net_pay_m", order,
                {"json": J.get("net_pay_m"), "giip": giip_np, "handout": H.get("net_pay_m"), "core": core_np},
                {"giip": f"calibrated so that the volumetric gas in place = giip_bcm "
                         f"({prm.get('giip_bcm')}) over the measured {area_m2 / 1e6:.1f} km2",
                 "core": "UPPER LIMIT: net-to-gross x cored interval (the cores span more than the pay)"})
    if v is None:
        sys.exit("Net pay is unknown. Give net_pay_m or giip_bcm in the settings JSON, or a "
                 "field-description .docx / core analysis.")
    prm["net_pay_m"] = v


# --------------------------------------------------------------------------
# forecast (tank model) and economics
# --------------------------------------------------------------------------
def run_forecast(G, pvt, p_i, ratio, q_w0, n_wells, rf, months, q_min=0.0,
                 rate_cap=None):
    """Monthly forecast. Returns dict of arrays. Volumes in std m3."""
    pz_i = p_i / pvt.z(p_i)
    c = q_w0 / (p_i ** 2 * (1 - ratio ** 2))
    gp, p, rows = 0.0, p_i, []
    for _ in range(months):
        pz = pz_i * (1.0 - gp / G)
        p = pvt.p_from_pz(pz)
        q = n_wells * c * p ** 2 * (1 - ratio ** 2)
        if rate_cap:
            q = min(q, rate_cap)
        if q < q_min:
            break
        vol = q * DAYS_PER_MONTH
        limit = rf * G - gp
        if vol >= limit:
            vol = max(limit, 0.0)
            gp += vol
            rows.append((p, vol / DAYS_PER_MONTH, vol, gp))
            break
        gp += vol
        rows.append((p, q, vol, gp))
    if not rows:
        rows = [(p_i, 0.0, 0.0, 0.0)]
    a = np.array(rows)
    return {"p": a[:, 0], "q": a[:, 1], "vol": a[:, 2], "gp": a[:, 3]}


def npv_musd(fc, n_wells, cgr, prm):
    vol = fc["vol"]
    m = np.arange(1, len(vol) + 1)
    disc = (1 + prm["discount_rate"]) ** (-m / 12.0)
    revenue = (vol / 1000.0 * prm["gas_price_usd_per_1000m3"]
               + vol * cgr * prm["condensate_price_usd_per_m3"]) / 1e6
    opex = np.full(len(vol), prm["opex_musd_per_year"] / 12.0)
    capex = n_wells * prm["well_cost_musd"] + prm["fixed_capex_musd"]
    return float(((revenue - opex) * disc).sum() - capex)


# --------------------------------------------------------------------------
# well placement  (v6: coverage of undrained field area)
# --------------------------------------------------------------------------
def _downsample(a, step):
    h, w = a.shape[0] // step * step, a.shape[1] // step * step
    return a[:h, :w].reshape(h // step, step, w // step, step).mean(axis=(1, 3))


def _upsample(a, step, shape):
    big = np.kron(a.astype(np.uint8), np.ones((step, step), np.uint8)).astype(bool)
    out = np.zeros(shape, bool)
    h, w = min(shape[0], big.shape[0]), min(shape[1], big.shape[1])
    out[:h, :w] = big[:h, :w]
    return out


def extract_faults(rgb, min_px=60, color=None, tol=80):
    """Faults = thick dark strokes (thin contour lines are removed by the opening).
    If `color` (R,G,B) is given, faults are the lines of that colour instead - use this when
    the map draws faults thin or in a colour (click a fault line to copy its colour)."""
    if color is not None:
        lines = np.abs(rgb.astype(int) - np.array(color, int)).sum(axis=2) <= tol
        lines = ndi.binary_dilation(lines, iterations=1)
        lab, n = ndi.label(lines, structure=np.ones((3, 3)))
        if n == 0:
            return np.zeros(lines.shape, bool)
        sizes = ndi.sum(lines, lab, range(1, n + 1))
        return np.isin(lab, np.where(sizes >= max(10, min_px // 2))[0] + 1)
    dark = rgb.astype(int).sum(axis=2) < 250
    thick = ndi.binary_opening(dark, structure=np.ones((3, 3)))
    lab, n = ndi.label(ndi.binary_dilation(thick), structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(dark.shape, bool)
    sizes = ndi.sum(thick, lab, range(1, n + 1))
    return np.isin(lab, np.where(sizes >= min_px)[0] + 1) & thick


class FieldModel:
    """Working-resolution (about 25 m) rasters: field, faults, fault blocks, depth, gas column."""

    def __init__(self, rgb, mask, m_per_px, prm, depth_pts=None, gwc_m=None, faults=None):
        self.D = max(1, int(round(25.0 / m_per_px)))
        self.m_per_px = m_per_px
        D, self.S = self.D, m_per_px * self.D
        S = self.S
        self.shape_full = mask.shape
        self.gd = _downsample(mask.astype(float), D) > 0.5
        if faults is None:
            faults = (extract_faults(rgb, int(prm["fault_min_pixels"]), prm.get("fault_color"),
                                     prm.get("fault_color_tol", 80))
                      if prm["use_faults"] else np.zeros(mask.shape, bool))
        self.fd = _downsample(faults.astype(float), D) > 0
        self.H, self.W = self.gd.shape
        self.yy, self.xx = np.mgrid[0:self.H, 0:self.W]
        # fault blocks: separate with a stand-off, then give every field pixel to its nearest block
        big, lab = [], None
        if self.fd.any():
            core = self.gd & ~ndi.binary_dilation(
                self.fd, iterations=max(1, int(round(prm["fault_standoff_m"] / S))))
            lab, n = ndi.label(core)
            areas = ndi.sum(core, lab, range(1, n + 1)) * S * S / 1e6 if n else []
            big = [i + 1 for i, a in enumerate(areas) if a > 0.3]
        if big:
            order = {b: k + 1 for k, b in enumerate(sorted(big, key=lambda b: -areas[b - 1]))}
            idx = ndi.distance_transform_edt(~np.isin(lab, big), return_distances=False,
                                             return_indices=True)
            nearest = lab[idx[0], idx[1]]
            self.block = np.where(self.gd, np.vectorize(lambda v: order.get(v, 0))(nearest), 0)
        else:
            self.block = np.where(self.gd, 1, 0)
        self.d_edge = ndi.distance_transform_edt(self.gd) * S
        self.d_fault = (ndi.distance_transform_edt(~self.fd) * S if self.fd.any()
                        else np.full(self.gd.shape, 1e9))
        # depth surface (thin-plate spline through the labels) and gas column
        self.depth = self.column = None
        if depth_pts and len(depth_pts) >= 4:
            P = np.array([[p[0], p[1]] for p in depth_pts], float)
            Z = np.array([p[2] for p in depth_pts], float)
            if len(P) > 400:                               # a contour network: many points, local fit
                _, ui = np.unique(np.round(P).astype(int), axis=0, return_index=True)
                P, Z = P[ui], Z[ui]
                rbf = RBFInterpolator(P, Z, kernel="thin_plate_spline", smoothing=2.0, neighbors=80)
            else:                                          # a few hand-read labels: smooth global fit
                rbf = RBFInterpolator(P, Z, kernel="thin_plate_spline", smoothing=50.0)
            pts = np.c_[self.xx.ravel() * D + D / 2, self.yy.ravel() * D + D / 2]
            self.depth = np.clip(rbf(pts).reshape(self.H, self.W), Z.min() - 10, Z.max() + 40)
            if gwc_m is not None:
                self.column = gwc_m - self.depth
        self.area_km2 = float(self.gd.sum() * S * S / 1e6)

    def set_gwc(self, gwc_m):
        self.column = gwc_m - self.depth

    def disc(self, r_m):
        k = int(np.ceil(r_m / self.S))
        y, x = np.mgrid[-k:k + 1, -k:k + 1]
        return (x * x + y * y <= (r_m / self.S) ** 2).astype(float)

    def circle(self, px, py, r_m):
        return ((self.xx - px / self.D) ** 2 + (self.yy - py / self.D) ** 2) <= (r_m / self.S) ** 2

    def at(self, arr, px, py):
        i = min(max(int(py / self.D), 0), self.H - 1)
        j = min(max(int(px / self.D), 0), self.W - 1)
        return arr[i, j]


# --------------------------------------------------------------------------
# contour lines and their depth numbers (v8)
# --------------------------------------------------------------------------
def ocr_engine():
    """First OCR package that works: rapidocr (pip only) or Tesseract. Returns (name, engine)."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        return "rapidocr", RapidOCR()
    except Exception:
        pass
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return "tesseract", pytesseract
    except Exception:
        pass
    return None, None


def _ocr_image(name, engine, arr):
    """Run OCR on one picture -> [(cx, cy, text, confidence 0-1)] in that picture's pixels."""
    out = []
    if name == "rapidocr":
        res, _ = engine(arr)
        for box, txt, conf in (res or []):
            c = np.mean(np.array(box, float), axis=0)
            out.append((float(c[0]), float(c[1]), str(txt), float(conf)))
    else:
        d = engine.image_to_data(Image.fromarray(arr), output_type=engine.Output.DICT,
                                 config="--psm 11 -c tessedit_char_whitelist=0123456789")
        for i, txt in enumerate(d["text"]):
            try:
                cf = float(d["conf"][i])
            except ValueError:
                continue
            if str(txt).strip() and cf > 0:
                out.append((d["left"][i] + d["width"][i] / 2.0, d["top"][i] + d["height"][i] / 2.0,
                            str(txt).strip(), cf / 100.0))
    return out


def read_depth_numbers(rgb, box, name, engine, scale=3.0, step=30, progress=True):
    """Read 3-4 digit numbers inside box=(x0,y0,x1,y1). Contour labels are printed along the
    lines at every angle, so the picture is enlarged and turned by `step` degrees many times.
    Returns [(x, y, value, confidence)] in the pixels of the full picture."""
    x0, y0, x1, y1 = box
    crop = Image.fromarray(rgb[y0:y1, x0:x1])
    big = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.BICUBIC)
    bw, bh = big.size
    hits = []
    angles = list(range(0, 360, step))
    for k, ang in enumerate(angles):
        rot = big.rotate(ang, expand=True, fillcolor=(255, 255, 255)) if ang else big
        rw, rh = rot.size
        th = math.radians(ang)
        for cx, cy, txt, conf in _ocr_image(name, engine, np.array(rot)):
            t = txt.replace(" ", "")
            mt = re.fullmatch(r"[-\u2212\u2013\u2014_.]?(\d{3,4})", t)       # '-2860' (depth below MSL) counts too
            if not mt:
                continue
            t = mt.group(1)
            dxp, dyp = cx - rw / 2.0, cy - rh / 2.0                    # back to the unrotated picture
            dx = math.cos(th) * dxp - math.sin(th) * dyp
            dy = math.sin(th) * dxp + math.cos(th) * dyp
            hits.append((x0 + (bw / 2.0 + dx) / scale, y0 + (bh / 2.0 + dy) / scale, int(t), conf))
        if progress:
            print(f"   reading numbers: angle {ang:3d} deg  ({k + 1}/{len(angles)})", flush=True)
    return hits


def vote_numbers(hits, interval, radius=30.0, min_votes=2, window=400.0):
    """Merge the readings from all angles. A number is kept if it was read in at least `min_votes`
    angles, is a multiple of the contour interval, and is within `window` m of the typical depth."""
    clusters = []
    for x, y, v, c in sorted(hits, key=lambda h: -h[3]):
        for cl in clusters:
            if math.hypot(cl["x"] - x, cl["y"] - y) < radius:
                cl["v"].append((v, c))
                break
        else:
            clusters.append({"x": x, "y": y, "v": [(v, c)]})
    out = []
    for cl in clusters:
        score = {}
        for v, c in cl["v"]:
            score[v] = score.get(v, 0.0) + c
        v = max(score, key=score.get)
        out.append({"x": cl["x"], "y": cl["y"], "depth": float(v), "score": score[v], "n": len(cl["v"])})
    out = [o for o in out if o["n"] >= min_votes]
    if interval:
        out = [o for o in out if abs(o["depth"] / interval - round(o["depth"] / interval)) < 1e-6]
    if out:
        med = float(np.median([o["depth"] for o in out]))
        out = [o for o in out if abs(o["depth"] - med) <= window]
    return out


def detect_contours(rgb, bold, dark_thr=170, min_len=40):
    """Contour lines = thin dark lines. Everything bold (faults, field outline, bold text) is
    taken out first via the `bold` mask, and so are small specks (digits). Returns a label image
    (0 = background, otherwise the number of the contour piece) and the list of piece numbers."""
    dark = rgb.max(axis=2) < dark_thr          # max channel: bright green / red / cyan are not 'dark'
    thin = dark & ~ndi.binary_dilation(bold, iterations=2)
    lab, n = ndi.label(thin, structure=np.ones((3, 3)))
    if n == 0:
        return lab, np.array([], int)
    sizes = ndi.sum(thin, lab, range(1, n + 1))
    keep = np.where(sizes >= min_len)[0] + 1
    return np.where(np.isin(lab, keep), lab, 0), keep


def assign_depths(lab, labels, interval, max_dist=30.0):
    """Give every contour piece the depth of the number printed next to it."""
    dist, idx = ndi.distance_transform_edt(lab == 0, return_indices=True)
    votes = {}
    for L in labels:
        xi, yi = int(round(L["x"])), int(round(L["y"]))
        L["comp"] = 0
        if not (0 <= yi < lab.shape[0] and 0 <= xi < lab.shape[1]) or dist[yi, xi] > max_dist:
            continue
        comp = int(lab[idx[0][yi, xi], idx[1][yi, xi]])
        L["comp"] = comp
        votes.setdefault(comp, []).append(L["depth"])
    depth_of, ambiguous = {}, 0
    tol = 0.5 * (interval or 20.0)
    for comp, vs in votes.items():
        if max(vs) - min(vs) <= tol:
            depth_of[comp] = float(np.median(vs))
            continue
        vals, cnt = np.unique(vs, return_counts=True)
        j = int(cnt.argmax())
        if cnt[j] >= 2 and cnt[j] > 0.5 * len(vs):
            depth_of[comp] = float(vals[j])
        else:
            ambiguous += 1                    # two different depths on one piece: not trusted
    return depth_of, {"numbers_on_a_line": sum(1 for L in labels if L["comp"]),
                      "ambiguous_pieces": ambiguous}


def contour_points(lab, depth_of, per_piece=60, stride=8):
    """Sample points along every labelled contour piece -> [(x, y, depth)]."""
    slices = ndi.find_objects(lab)
    pts = []
    for comp, d in depth_of.items():
        sl = slices[comp - 1] if comp - 1 < len(slices) else None
        if sl is None:
            continue
        ys, xs = np.nonzero(lab[sl] == comp)
        n = len(ys)
        if n == 0:
            continue
        m = int(min(per_piece, max(6, n // stride)))
        for i in np.linspace(0, n - 1, m).astype(int):
            pts.append((float(xs[i] + sl[1].start), float(ys[i] + sl[0].start), float(d)))
    return pts


def draw_contour_debug(rgb, bold, lab, depth_of, labels, path):
    """Picture of what was found: red = bold lines (faults, outline, bold text), coloured = contour
    lines with their depth, grey = contour lines without a number, black numbers = what was read."""
    img = (rgb.astype(float) * 0.30 + 255 * 0.70).astype(np.uint8)
    img[bold] = (215, 0, 0)
    lut = np.zeros((int(lab.max()) + 1, 3), np.uint8)
    lut[1:] = (150, 150, 150)
    if depth_of:
        ds = np.array(list(depth_of.values()), float)
        lo, hi = float(ds.min()), float(ds.max())
        cmap = plt.get_cmap("turbo")
        for comp, d in depth_of.items():
            c = cmap((d - lo) / max(hi - lo, 1.0))
            lut[comp] = [int(255 * c[0]), int(255 * c[1]), int(255 * c[2])]
    sel = lab > 0
    img[sel] = lut[lab[sel]]
    if depth_of:                                            # make the labelled lines easier to see
        lab_l = np.where(np.isin(lab, list(depth_of)), lab, 0)
        lab_d = ndi.grey_dilation(lab_l, size=(3, 3))
        img[lab_d > 0] = lut[lab_d[lab_d > 0]]
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for L in labels:
        dr.rectangle([L["x"] - 14, L["y"] - 6, L["x"] + 14, L["y"] + 6], fill=(255, 255, 255))
        dr.text((L["x"] - 12, L["y"] - 5), str(int(L["depth"])), fill=(0, 0, 0))
    dr.text((20, 20), "red = bold lines (faults / outline / bold text)   coloured = contour lines with depth   "
                      "grey = contour lines without a number", fill=(0, 0, 0))
    im.save(path)


def estimate_gwc(fm):
    """Gas-water contact estimate = the deep end of the depth surface along the field edge (90th percentile)."""
    edge = fm.gd & ~ndi.binary_erosion(fm.gd, iterations=2)
    return float(np.percentile(fm.depth[edge], 90))


def auto_contour_pipeline(rgb, mask, faults_full, map_path, prm, out_dir, a):
    """Detect the thin contour lines, read the depth numbers (OCR) and return depth points
    [(x, y, depth)] for the whole labelled contour network, plus a summary dict."""
    interval = float(a.contour_interval or prm["contour_interval_m"])
    name, eng = ocr_engine()
    if name is None:
        print("\nAUTOMATIC CONTOURS need an OCR package. Install one of:\n"
              "    python -m pip install rapidocr-onnxruntime      (easiest, nothing else needed)\n"
              "    Tesseract + python -m pip install pytesseract\n"
              "Continuing WITHOUT depth information (answer 'click' or 'type' at the depth step instead).")
        return [], None
    bold = extract_faults(rgb, int(prm["fault_min_pixels"])) | faults_full
    lab, kept = detect_contours(rgb, bold, prm["contour_dark_threshold"], int(prm["contour_min_length_px"]))
    print(f"\nContour detection: {len(kept)} thin contour lines found; {int(bold.sum())} bold pixels "
          f"(faults, field outline, bold text) were kept apart from them.")
    ys, xs = np.nonzero(mask)
    m = 250
    box = (int(max(0, xs.min() - m)), int(max(0, ys.min() - m)),
           int(min(rgb.shape[1], xs.max() + m)), int(min(rgb.shape[0], ys.max() + m)))
    cache = Path(map_path).with_name(Path(map_path).stem + "_numbers.json")
    key = {"box": list(box), "scale": float(a.ocr_scale), "step": int(a.ocr_angle_step),
           "engine": name, "size": [int(rgb.shape[0]), int(rgb.shape[1])]}
    hits = None
    if cache.exists() and not a.redo_ocr:
        try:
            c = json.loads(cache.read_text())
            if c.get("key") == key:
                hits = [tuple(h) for h in c["hits"]]
                print(f"   (using the stored number readings in {cache.name}; add --redo-ocr to read again)")
        except Exception:
            hits = None
    if hits is None:
        print(f"   reading the depth numbers with {name} - about a minute, the picture is turned "
              f"{360 // int(a.ocr_angle_step)} times ...")
        hits = read_depth_numbers(rgb, box, name, eng, float(a.ocr_scale), int(a.ocr_angle_step))
        try:
            cache.write_text(json.dumps({"key": key, "hits": hits}))
        except Exception:
            pass
    labels = vote_numbers(hits, interval)
    depth_of, info = assign_depths(lab, labels, interval, prm["contour_label_max_dist_px"])
    pts = contour_points(lab, depth_of) + [(L["x"], L["y"], L["depth"]) for L in labels]
    summary = {"contour_lines": int(len(kept)), "numbers_read": len(labels),
               "numbers_on_a_line": info["numbers_on_a_line"], "lines_with_depth": len(depth_of),
               "ambiguous_lines_dropped": info["ambiguous_pieces"], "interval_m": interval,
               "ocr_engine": name}
    if depth_of:
        ds = list(depth_of.values())
        summary["depth_range_m"] = [min(ds), max(ds)]
    print(f"   numbers read: {len(labels)} (multiples of {interval:g} m, read at 2+ angles); "
          f"contour lines with a depth: {len(depth_of)} of {len(kept)}"
          + (f"; {info['ambiguous_pieces']} dropped because two different numbers sat on one line"
             if info["ambiguous_pieces"] else ""))
    if depth_of:
        print(f"   depth range of the labelled contours: {min(depth_of.values()):.0f} - "
              f"{max(depth_of.values()):.0f} m")
    else:
        print("   WARNING: no contour line got a depth - check contours.png; lower "
              "contour_dark_threshold / contour_min_length_px or give depth labels by hand.")
    try:
        draw_contour_debug(rgb, bold, lab, depth_of, labels, out_dir / "contours.png")
        rows = [{"depth_m": depth_of.get(c), "line_id": int(c), "pixels": int((lab == c).sum())}
                for c in kept[:2000] if c in depth_of]
        pd.DataFrame(rows).to_csv(out_dir / "contours.csv", index=False)
        print(f"   picture of the result: {out_dir / 'contours.png'}")
    except Exception as e:
        print(f"   (could not write contours.png: {e})")
    return pts, summary


def drained_mask(fm, pts, radius_m, leaky):
    dr = np.zeros((fm.H, fm.W), bool)
    for (x, y) in pts:
        c = fm.circle(x, y, radius_m)
        if not leaky:
            c &= fm.block == fm.at(fm.block, x, y)
        dr |= c
    return dr


def plan_wells(fm, seeds_px, radius_m, leaky, platform_px, reach_m, prm, max_new=40):
    """Greedy coverage placement. Returns (list of new wells, drained mask of seeds+new)."""
    D, S = fm.D, fm.S
    ok = fm.gd & (fm.d_edge >= prm["edge_buffer_m"])
    if fm.fd.any():
        ok &= fm.d_fault >= prm["fault_standoff_m"]
    if fm.column is not None and prm["min_gas_column_m"] > 0:
        ok &= fm.column >= prm["min_gas_column_m"]
    if platform_px is not None and reach_m:
        dp = np.hypot(fm.xx * D + D / 2 - platform_px[0],
                      fm.yy * D + D / 2 - platform_px[1]) * fm.m_per_px
        ok &= dp <= reach_m
    kern = fm.disc(radius_m)
    drained = drained_mask(fm, seeds_px, radius_m, leaky)
    placed = [tuple(s) for s in seeds_px]
    new = []
    for _ in range(max_new):
        sp_ok = ok.copy()
        if placed and prm["min_well_spacing_m"]:
            wm = np.zeros((fm.H, fm.W), bool)
            for (px, py) in placed:
                wm[min(int(py / D), fm.H - 1), min(int(px / D), fm.W - 1)] = True
            sp_ok &= ndi.distance_transform_edt(~wm) * S >= prm["min_well_spacing_m"]
        und = (fm.gd & ~drained).astype(float)
        best = None
        for k in ([None] if leaky else range(1, int(fm.block.max()) + 1)):
            u = und if k is None else und * (fm.block == k)
            cov = fftconvolve(u, kern, mode="same") * S * S / 1e6
            sel = sp_ok if k is None else sp_ok & (fm.block == k)
            bonus = (0.1 * (fm.column - prm["min_gas_column_m"]) / 110.0
                     if fm.column is not None else 0.0)             # small preference for the crest
            score = np.where(sel, cov + bonus, -1.0)
            j = np.unravel_index(score.argmax(), score.shape)
            if score[j] > -1.0 and (best is None or score[j] > best[0]):
                best = (score[j], j, cov[j])
        if best is None or best[2] < prm["min_gain_km2"]:
            break
        j = best[1]
        cx, cy = j[1] * D + D / 2, j[0] * D + D / 2
        dd = fm.circle(cx, cy, radius_m)
        gain = dd & ~drained & (fm.gd if leaky else fm.block == fm.block[j])
        drained |= gain
        placed.append((cx, cy))
        item = {"x": float(cx), "y": float(cy), "gain_km2": float(gain.sum() * S * S / 1e6)}
        if fm.depth is not None:
            item["depth_m"] = float(fm.depth[j])
            item["column_m"] = float(fm.column[j])
        new.append(item)
    return new, drained


def constraint_funnel(fm, prm, platform_px, reach_m):
    """How much field area is left after each placement rule (km2) - shows which rule kills the search."""
    S = fm.S
    ok = fm.gd.copy()
    steps = [("inside the field", float(ok.sum() * S * S / 1e6))]
    ok &= fm.d_edge >= prm["edge_buffer_m"]
    steps.append((f"at least {prm['edge_buffer_m']:.0f} m from the field edge", float(ok.sum() * S * S / 1e6)))
    if fm.fd.any():
        ok &= fm.d_fault >= prm["fault_standoff_m"]
        steps.append((f"at least {prm['fault_standoff_m']:.0f} m from a fault", float(ok.sum() * S * S / 1e6)))
    if fm.column is not None and prm["min_gas_column_m"]:
        ok &= fm.column >= prm["min_gas_column_m"]
        steps.append((f"gas column of at least {prm['min_gas_column_m']:.0f} m", float(ok.sum() * S * S / 1e6)))
    if platform_px is not None and reach_m:
        dp = np.hypot(fm.xx * fm.D + fm.D / 2 - platform_px[0],
                      fm.yy * fm.D + fm.D / 2 - platform_px[1]) * fm.m_per_px
        ok &= dp <= reach_m
        steps.append((f"within {reach_m:.0f} m of the platform", float(ok.sum() * S * S / 1e6)))
    return steps


def voronoi_cells(mask, w):
    """Label every field pixel with its nearest well (used for the drainage-cell colours)."""
    fy, fx = np.nonzero(mask)
    cells = -np.ones(mask.shape, int)
    cells[fy, fx] = cdist(np.column_stack([fx, fy]).astype(float), w).argmin(axis=1)
    return cells


def well_stats(w, cells, mask, m_per_px, extra=None, fm=None, platform=None):
    dist_edge = ndi.distance_transform_edt(mask) * m_per_px
    stats = []
    for k, (x, y) in enumerate(w):
        others = np.delete(w, k, axis=0)
        nn = (np.sqrt(((others - w[k]) ** 2).sum(axis=1)).min() * m_per_px
              if len(others) else float("nan"))
        yi = min(max(int(round(y)), 0), mask.shape[0] - 1)
        xi = min(max(int(round(x)), 0), mask.shape[1] - 1)
        s = {
            "well": f"W{k + 1}",
            "x_px": round(float(x), 1), "y_px": round(float(y), 1),
            "x_m_from_left": round(float(x * m_per_px), 0),
            "y_m_from_top": round(float(y * m_per_px), 0),
            "drainage_km2": round(float((cells == k).sum() * m_per_px ** 2 / 1e6), 2),
            "nearest_well_m": round(float(nn), 0) if nn == nn else None,
            "dist_to_edge_m": round(float(dist_edge[yi, xi]), 0),
        }
        if extra is not None and extra[k] is not None:
            s["new_undrained_km2"] = round(extra[k]["gain_km2"], 2)
        if fm is not None and fm.depth is not None:
            s["top_depth_m"] = round(float(fm.at(fm.depth, x, y)), 1)
            s["gas_column_m"] = round(float(fm.at(fm.column, x, y)), 1)
        if platform is not None:
            s["dist_to_platform_m"] = round(float(np.hypot(x - platform[0], y - platform[1]) * m_per_px), 0)
        stats.append(s)
    return stats


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_wells(rgb, mask, cells, w, seeds_n, m_per_px, stats, title, path, extra=None):
    """extra (optional): dict(radius_m, uncovered=bool mask, platform=(x,y), reach_m)."""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(rgb)
    cmap = plt.get_cmap("tab10")
    ov = np.zeros(mask.shape + (4,))
    for k in range(len(w)):
        ov[cells == k] = (*cmap(k % 10)[:3], 0.15)
    ax.imshow(ov)
    if extra and extra.get("uncovered") is not None:
        pu = np.zeros(mask.shape + (4,))
        pu[extra["uncovered"]] = (0.75, 0.0, 0.8, 0.5)
        ax.imshow(pu)
    ax.contour(mask.astype(float), levels=[0.5], colors="k", linewidths=1.0)
    if extra:
        rpx = extra["radius_m"] / m_per_px
        for (x, y) in w:
            ax.add_patch(plt.Circle((x, y), rpx, fill=False, ec="k", lw=1.0, ls="--", alpha=0.7))
        if extra.get("platform") is not None and extra.get("reach_m"):
            ax.add_patch(plt.Circle(extra["platform"], extra["reach_m"] / m_per_px,
                                    fill=False, ec="royalblue", lw=1.5, ls=":"))
    for k, (x, y) in enumerate(w):
        fixed = k < seeds_n
        col = "royalblue" if fixed else "red"
        ax.plot(x, y, "o", ms=30, mfc="none", mec=col, mew=2.5)          # outer ring
        ax.plot(x, y, "o", ms=20, mfc=col, mec="white", mew=1.5)         # filled circle
        ax.text(x, y, str(k + 1), color="white", ha="center", va="center",
                fontsize=10, weight="bold")
    bar_px = 1000 / m_per_px
    ax.plot([25, 25 + bar_px], [25, 25], "k-", lw=4)
    ax.text(25 + bar_px / 2, 45, "1000 m (calibration)", ha="center", fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    handles = [
        plt.Line2D([], [], marker="o", color="w", mfc="red", ms=13, label="proposed well"),
        plt.Line2D([], [], marker="o", color="w", mfc="royalblue", ms=13,
                   label="seed / existing well")]
    if extra:
        handles += [plt.Line2D([], [], color="k", ls="--", label="drainage circle"),
                    plt.Rectangle((0, 0), 1, 1, fc=(0.75, 0.0, 0.8, 0.45),
                                  label="field not reached by any well")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    return fig


def plot_forecast(results, best_n, G, rf, cgr, path):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for n, r in results.items():
        t = np.arange(1, len(r["fc"]["q"]) + 1) / 12
        lw, al = (3, 1) if n == best_n else (1, 0.5)
        lab = f"{n} well{'s' if n > 1 else ''}" + (" (best)" if n == best_n else "")
        axs[0].plot(t, r["fc"]["q"] / 1e6, lw=lw, alpha=al, label=lab)
        axs[1].plot(t, r["fc"]["gp"] / 1e9, lw=lw, alpha=al, label=lab)
    axs[1].axhline(rf * G / 1e9, ls="--", c="grey")
    axs[1].text(0.5, rf * G / 1e9, " recoverable limit", va="bottom", color="grey")
    axs[0].set(xlabel="years from first gas", ylabel="gas rate, million m3/day",
               title="Gas rate")
    axs[1].set(xlabel="years from first gas", ylabel="cumulative gas, billion m3",
               title="Cumulative gas")
    axs[0].legend(fontsize=8)
    ns = list(results)
    npv = [results[n]["npv"] for n in ns]
    cols = ["tab:red" if n == best_n else "tab:blue" for n in ns]
    axs[2].bar(ns, npv, color=cols)
    axs[2].axhline(0, c="k", lw=0.8)
    axs[2].set(xlabel="number of wells", ylabel="NPV, million USD",
               title="NPV vs number of wells")
    fig.suptitle(f"Forecast (condensate = gas x {cgr * 1e6:.1f} m3/million m3)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return fig


def plot_history(hist, model, path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    t = np.arange(len(hist)) / 12
    axs[0].bar(t, hist["q_m3d"] / 1e6, width=1 / 12, color="grey", label="actual (monthly)")
    tm = np.arange(1, len(model["q"]) + 1) / 12
    axs[0].plot(tm, model["q"] / 1e6, "r", lw=2, label="model")
    axs[0].set(xlabel="years", ylabel="million m3/day", title="Rate")
    axs[0].legend()
    axs[1].plot(t, hist["gas"].cumsum() / 1e9, "k", lw=2, label="actual")
    axs[1].plot(tm, model["gp"] / 1e9, "r", lw=2, label="model")
    axs[1].set(xlabel="years", ylabel="billion m3", title="Cumulative gas")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return fig


# --------------------------------------------------------------------------
# files, menus, map loading
# --------------------------------------------------------------------------
def writable(path):
    """Return `path`, or a time-stamped twin if the file is locked (e.g. open in Excel)."""
    path = Path(path)
    try:
        with open(path, "ab"):
            pass
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{time.strftime('%H%M%S')}{path.suffix}")
        print(f"   ({path.name} is open in another program - writing {alt.name} instead)")
        return alt


def open_files(paths):
    """Open the result pictures (and the output folder) with the default viewer."""
    targets = [Path(paths[0]).parent] + [Path(f) for f in paths]
    for f in targets:
        try:
            if hasattr(os, "startfile"):                     # Windows
                os.startfile(str(f.resolve()))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(f)])
            else:
                subprocess.Popen(["xdg-open", str(f)])
        except Exception as e:
            print(f"   (could not open {f.name} automatically: {e})")


IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf"}
HIST_EXT = {".xlsx", ".xls", ".csv"}


def load_map_image(path, dpi, page):
    """Return an RGB array for the map, and the file actually used for pixel
    measurements. A .pdf is rasterized to a PNG first (PIL cannot read PDF
    pages directly) - the rendered PNG is cached next to the PDF so pixel
    coordinates stay valid between runs."""
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        return np.array(Image.open(path).convert("RGB")), path

    try:
        import pymupdf
    except ImportError:
        sys.exit("Reading a .pdf map needs the pymupdf package:\n"
                 "    python -m pip install pymupdf")
    rendered = path.with_name(f"{path.stem}_p{page}_{int(dpi)}dpi.png")
    if not rendered.exists():
        pymupdf.TOOLS.mupdf_display_errors(False)   # silence harmless colour-profile warnings
        doc = pymupdf.open(path)
        if not (1 <= page <= len(doc)):
            sys.exit(f"{path.name} only has {len(doc)} page(s); --pdf-page {page} is out of range.")
        pix = doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
        Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(rendered)
        doc.close()
        print(f"   (rendered {path.name} page {page} at {dpi:g} dpi -> {rendered.name})")
    return np.array(Image.open(rendered).convert("RGB")), rendered


def is_params_file(f):
    """True if a .json file looks like a settings / parameter file for this program."""
    try:
        d = json.loads(Path(f).read_text())
        known = set(DEFAULTS) | set(REQUIRED)
        return isinstance(d, dict) and sum(1 for k in d if k in known or str(k).startswith("_")) >= 2
    except Exception:
        return False


def find_files(exts, valid=None):
    """Files with these extensions in the current folder (and the script's folder)."""
    here = Path(__file__).resolve().parent
    dirs = [Path.cwd()] + ([here] if here != Path.cwd().resolve() else [])
    found, seen = [], set()
    for d in dirs:
        for f in sorted(d.iterdir(), key=lambda q: q.name.lower()):
            if not f.is_file() or f.name.startswith("~$"):
                continue
            if f.stem.lower().startswith(("wells_map", "forecast", "history_match", "summary")):
                continue                                  # this program's own output files
            if f.suffix.lower() not in exts or f.resolve() in seen:
                continue
            if valid and not valid(f):
                continue
            seen.add(f.resolve())
            found.append(f)
    return found


def ask_choice(title, options, default_idx=0):
    """Numbered menu. Enter = default. Returns the index chosen."""
    print(f"\n{title}")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    while True:
        try:
            raw = input(f"Enter number [{default_idx + 1}]: ").strip()
        except EOFError:
            return default_idx
        if raw == "":
            return default_idx
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"  Please type a number from 1 to {len(options)}.")


def pick_file(title, files, preferred, interactive, allow_none=None):
    """Choose one of `files`. Asks only when there is a real choice.
    allow_none = label of an extra 'skip' option (returns None if chosen)."""
    default = next((i for i, f in enumerate(files) if f.name.lower() == preferred), 0)
    names = [f.name for f in files]
    dup = {n for n in names if names.count(n) > 1}
    labels = [str(f) if f.name in dup else f.name for f in files]
    if len(files) == 1 or not interactive:
        print(f"Using {title.lower()}: {labels[default]}")
        return files[default]
    if allow_none:
        labels.append(allow_none)
    idx = ask_choice(f"Which {title.lower()} should be used?", labels, default)
    return None if allow_none and idx == len(files) else files[idx]


def ask_map_image(default_name=None):
    """ALWAYS ask which map picture to use: pick from the files in the folder,
    or type / paste / drag-and-drop a path to any other image or PDF."""
    imgs = find_files(IMG_EXT)
    names = [g.name for g in imgs]
    labels = [str(f) if names.count(f.name) > 1 else f.name for f in imgs]
    labels.append(">> type or paste the path to a different map image")
    default = next((i for i, f in enumerate(imgs)
                    if default_name and f.name.lower() == Path(default_name).name.lower()), 0)
    idx = ask_choice("MAP PICTURE - which map image should be used?", labels, default)
    if idx < len(imgs):
        return imgs[idx]
    while True:
        raw = input("Path to the map image (you can drag the file into this window): ")
        p = Path(raw.strip().strip('"').strip("'")).expanduser()
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            return p
        print(f"  Not found, or not an image/PDF ({', '.join(sorted(IMG_EXT))}). Try again.")


def ask_number(prompt, default=None):
    while True:
        shown = f" [{default:g}]" if default is not None else ""
        try:
            raw = input(f"{prompt}{shown}: ").strip()
        except EOFError:
            if default is None:
                sys.exit("Input needed but none given.")
            return default
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Please type a number.")


def ask_optional_number(prompt, default=None):
    """Number or Enter for 'none' (or for the default, if one is shown). Returns float or None."""
    while True:
        shown = f" [{default:g}]" if default is not None else ""
        try:
            raw = input(f"{prompt}{shown}: ").strip()
        except EOFError:
            return default
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Please type a number (or just press Enter).")


def parse_xy(text):
    if not text:
        return []
    return [tuple(float(v) for v in item.split(",")) for item in text.split(";") if item.strip()]


def parse_xyz(text):
    """'x,y,depth;x,y,depth' -> [(x, y, depth), ...]"""
    if not text:
        return []
    return [tuple(float(v) for v in item.split(",")[:3]) for item in text.split(";") if item.strip()]


def seeds_to_str(v):
    """JSON gives [[x,y],...]; the command line gives "x,y;x,y"."""
    if isinstance(v, list):
        return ";".join(f"{p[0]},{p[1]}" for p in v)
    return v


# --------------------------------------------------------------------------
# click-on-the-map tools (replaces measuring in Paint)
# --------------------------------------------------------------------------
def can_click():
    return matplotlib.get_backend().lower() != "agg"


def click_points(rgb, title, n):
    """Show the image and collect n clicks (n=-1: as many as you like, press
    Enter when done). Returns a list of (x, y) pixel positions."""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=11)
    try:
        fig.canvas.manager.set_window_title("reservoir_planner - click on the image")
    except Exception:
        pass
    fig.tight_layout()
    pts = plt.ginput(n=n, timeout=0, show_clicks=True)
    plt.close(fig)
    return pts


def choose_scale(rgb, saved_px, saved_m, saved_label):
    """Interactive map-scale step - ALWAYS asked. Returns (scalebar_px, scalebar_m, changed)."""
    options, kinds = [], []
    if can_click():
        options.append("Click the two ends of the scale bar on the map (recommended)")
        kinds.append("click")
    options += ["Type the scale-bar length in pixels (measured in Paint)",
                "Type metres-per-pixel directly"]
    kinds += ["px", "mpp"]
    if saved_px and saved_m:
        options.append(f"Reuse the saved scale: {saved_px:g} px = {saved_m:g} m ({saved_label})")
        kinds.append("saved")
    print("\nThe scale of the map is needed to turn pixels into metres (field area, well spacing).")
    kind = kinds[ask_choice("MAP SCALE - how do you want to give the scale of this map?",
                            options, 0)]
    if kind == "saved":
        return saved_px, saved_m, False
    if kind == "click":
        print("\nA picture window will open. Find the scale bar printed on the map.")
        pts = click_points(rgb, "Click 3 times: a first (throwaway) click to get lined up, "
                                "then the LEFT end of the scale bar, then the RIGHT end. "
                                "Only the last 2 clicks are measured.", n=3)
        if len(pts) < 3:
            sys.exit("Cancelled - fewer than 3 clicks were made.")
        (x0, y0), (x1, y1) = pts[-2:]
        px = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        print(f"  -> {px:.1f} pixels between your two clicks.")
        m = ask_number("  What distance does that bar represent, in METRES "
                       "(read it off the map, e.g. 5 km = 5000)")
        return px, m, True
    if kind == "px":
        px = ask_number("Length of the scale bar in PIXELS")
        m = ask_number("Length of that scale bar in METRES (e.g. 10 km = 10000)")
        return px, m, True
    mpp = ask_number("Metres per pixel")
    return 1.0, mpp, True            # 1 px = mpp metres


def choose_wells(rgb, saved_str, saved_label):
    """Interactive existing-wells step. Returns (seed string 'x,y;x,y', changed)."""
    options, kinds = [], []
    if saved_str:
        options.append(f"Use the saved wells: {saved_str} ({saved_label})")
        kinds.append("saved")
    if can_click():
        options.append("Click the existing wells on the map")
        kinds.append("click")
    options += ["Type pixel coordinates  x,y;x,y", "No existing wells"]
    kinds += ["type", "none"]
    default = 0 if saved_str else len(kinds) - 1
    kind = kinds[ask_choice("EXISTING WELLS - are there wells already drilled in this field?",
                            options, default)]
    if kind == "saved":
        return saved_str, False
    if kind == "click":
        print("\nClick each existing well, then press ENTER in the picture window to finish "
              "(press Enter right away if there are none).")
        pts = click_points(rgb, "Click each EXISTING well. Press ENTER when done.", n=-1)
        return ";".join(f"{x:.1f},{y:.1f}" for x, y in pts), True
    if kind == "type":
        return input("Existing wells as pixel coordinates 'x,y;x,y': ").strip(), True
    return "", True


def choose_depth(rgb, saved_pts, saved_gwc, saved_label, saved_auto=False):
    """Interactive depth-label step. Returns (points [(x,y,depth)], gwc_depth or None, changed).
    Depth labels give the gas column above the gas-water contact for every spot on the map."""
    options, kinds = [], []
    if saved_pts:
        options.append(f"Use the saved depth labels ({len(saved_pts)} points, contact "
                       f"{saved_gwc if saved_gwc is not None else '?'} m; {saved_label})")
        kinds.append("saved")
    options.append("AUTOMATIC: detect the thin contour lines and read their depth numbers (OCR; "
                   "about a minute the first time)")
    kinds.append("auto")
    if can_click():
        options.append("Click depth labels on the map and type their depths")
        kinds.append("click")
    options += ["Type points as  x,y,depth;x,y,depth", "Skip - no depth information"]
    kinds += ["type", "none"]
    default = (kinds.index("auto") if saved_auto else 0 if saved_pts else len(kinds) - 1)
    kind = kinds[ask_choice("DEPTH LABELS - do you want to use depth labels from the map "
                            "(needed for the gas-column rule)?", options, default)]
    if kind == "none":
        return [], None, True
    if kind == "auto":
        return "auto", None, True
    if kind == "saved":
        return saved_pts, saved_gwc, False
    pts = []
    if kind == "click":
        print("\nClick the position of each depth label (a number printed on a contour) - 12 to "
              "25 labels spread over the field work well - then press ENTER in the picture "
              "window. Tip: use the magnifier in the window toolbar to zoom, then click.")
        clicks = click_points(rgb, "Click depth labels (contour numbers). Press ENTER when done.", n=-1)
        for i, (x, y) in enumerate(clicks, 1):
            d = ask_number(f"  Depth printed at click {i} (x={x:.0f}, y={y:.0f}) in metres")
            pts.append((float(x), float(y), d))
    else:
        pts = parse_xyz(input("Depth points 'x,y,depth;x,y,depth': ").strip())
    if len(pts) < 4:
        print("  Need at least 4 depth points - continuing without depth information.")
        return [], None, True
    gwc = ask_number("Depth of the gas-water contact in metres (same depth system as the labels)")
    return pts, gwc, True


def choose_faults(rgb, saved):
    """How are faults drawn on this map? Returns '' (thick black, default), 'none', or 'R,G,B'."""
    options, kinds = [], []
    if saved:
        options.append(f"Use the saved fault colour ({saved})")
        kinds.append("saved")
    options.append("Faults are THICK BLACK lines (default - e.g. the NAM L15 map)")
    kinds.append("black")
    if can_click():
        options.append("Click on a fault line to copy its colour (use this if faults are thin or coloured)")
        kinds.append("click")
    options += ["Type the fault colour as R,G,B", "Ignore faults on this map"]
    kinds += ["type", "none"]
    kind = kinds[ask_choice("FAULTS - how are faults drawn on this map?", options, 0)]
    if kind == "saved":
        return saved, False
    if kind == "black":
        return "", saved not in (None, "")
    if kind == "none":
        return "none", True
    if kind == "type":
        return input("Fault colour as R,G,B (e.g. 90,30,120): ").strip(), True
    print("\nA picture window will open. Click ON a fault line (zoom in first with the magnifier "
          "in the toolbar if the line is thin).")
    pts = click_points(rgb, "Click once on a fault line.", n=1)
    if not pts:
        return "", False
    x, y = int(round(pts[0][0])), int(round(pts[0][1]))
    patch = rgb[max(0, y - 4):y + 5, max(0, x - 4):x + 5].reshape(-1, 3).astype(int)
    med = np.median(patch, axis=0)
    c = patch[np.abs(patch - med).sum(axis=1).argmax()]          # the pixel that differs most = the line
    print(f"  -> fault colour {int(c[0])},{int(c[1])},{int(c[2])}")
    return f"{int(c[0])},{int(c[1])},{int(c[2])}", True


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", help="map image (png/jpg/pdf); if omitted you choose from a menu")
    ap.add_argument("--pdf-dpi", type=float, default=200.0,
                    help="resolution to render a .pdf map at (default 200)")
    ap.add_argument("--pdf-page", type=int, default=1,
                    help="page number to use if the .pdf has more than one page (default 1)")
    ap.add_argument("--params", help="reservoir data (core analysis) JSON; if omitted you choose from a menu")
    ap.add_argument("--history",
                    help="production history .xlsx/.csv ('none' to skip); if omitted you choose from a menu")
    ap.add_argument("--history-wells", type=int, default=1,
                    help="how many wells produced the history (default 1)")
    ap.add_argument("--core", help="core-analysis spreadsheet (depth, porosity, permeability); 'none' to skip")
    ap.add_argument("--field-description", help="project handout .docx with the reservoir table and the gas composition; 'none' to skip")
    ap.add_argument("--reservoir", help="which reservoir (row) of the handout table to use, e.g. L15b-A")
    ap.add_argument("--prefer-handout", action="store_true",
                    help="use the handout values before the values calculated from the cores")
    ap.add_argument("--save-params", help="write all calculated parameters to this JSON file")
    ap.add_argument("--m-per-px", type=float, help="map scale, metres per pixel")
    ap.add_argument("--scalebar-px", type=float, help="length of the scale bar in pixels")
    ap.add_argument("--scalebar-m", type=float, help="length of the scale bar in metres")
    ap.add_argument("--field-color", help="R,G,B of the field polygon (default: any green)")
    ap.add_argument("--color-tol", type=int, default=None)
    ap.add_argument("--merge-disconnected-pieces", action="store_true",
                    help="keep every same-coloured blob, not just the largest - only use "
                         "this if the field's own polygon is genuinely split into pieces")
    ap.add_argument("--seed-wells",
                    help='fixed wells in pixel coords "x,y;x,y" (e.g. an existing well)')
    ap.add_argument("--platform", help='platform position in pixel coords "x,y" '
                                       '(default: the first existing well)')
    ap.add_argument("--reach", type=float, help="maximum horizontal reach of wells from the platform, metres")
    ap.add_argument("--radius", type=float, help="drainage radius in metres (overrides the history calibration)")
    ap.add_argument("--sealing", action="store_true",
                    help="treat faults as sealing: each well only drains its own fault block")
    ap.add_argument("--depth-points",
                    help='depth labels as pixel coords and depth "x,y,depth;x,y,depth"')
    ap.add_argument("--gwc-depth", type=float, help="depth of the gas-water contact, metres")
    ap.add_argument("--fault-color", help='colour of the fault lines "R,G,B" (default: thick black lines); '
                                          '"none" ignores faults')
    ap.add_argument("--auto-contours", action="store_true",
                    help="detect the contour lines and read their depth numbers (OCR) instead of "
                         "clicking depth labels")
    ap.add_argument("--contour-interval", type=float, help="contour spacing in metres (default 20)")
    ap.add_argument("--ocr-scale", type=float, default=3.0, help="enlargement used before reading numbers")
    ap.add_argument("--ocr-angle-step", type=int, default=30, help="turn the picture in steps of this many degrees")
    ap.add_argument("--redo-ocr", action="store_true", help="read the numbers again (ignore the stored readings)")
    ap.add_argument("--n-wells", type=int, help="force the number of wells")
    ap.add_argument("--well-rate-mm3d", type=float,
                    help="initial rate per well, million m3/day (overrides history/Darcy)")
    ap.add_argument("--rate-cap-mm3d", type=float, help="facility limit, million m3/day")
    ap.add_argument("--npv-tolerance", type=float,
                    help="recommend the FEWEST wells whose NPV is within this fraction of the "
                         "maximum NPV (default 0.02 = 2%%; 0 = strictly the maximum)")
    ap.add_argument("--no-match-history", dest="match_history", action="store_false",
                    help="do NOT scale the gas in place to reproduce the history")
    ap.add_argument("--no-save", action="store_true",
                    help="do not offer to save map/scale/wells into the params JSON")
    ap.add_argument("--no-ask", action="store_true",
                    help="never ask questions: use the command line / saved values / defaults")
    ap.add_argument("--no-show", action="store_true",
                    help="do not pop up the picture windows (files are still saved)")
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()
    interactive = not a.no_ask   # always ask unless --no-ask
    be = matplotlib.get_backend()
    if be.lower() == "agg":
        print("Graphics: no window system found (backend 'Agg'). Pictures are SAVED to the output "
              "folder and opened with your image viewer; clicking on the map is not available "
              "(type pixel coordinates instead).\n          To get windows: python -m pip install pyqt5")
    else:
        print(f"Graphics: backend '{be}' - picture windows will open.")
    print(f"[reservoir_planner v8 - contours + parameters calculated from the data files, wells placed by coverage of undrained field area]  "
          f"running: {Path(__file__).resolve()}")

    # ---- 1. settings (params) file - optional ----------------------------------
    if a.params is None:
        js = find_files({".json"}, valid=is_params_file)
        if js:
            a.params = str(pick_file("Settings (.json)", js, "params_example.json", interactive))
        else:
            print("No settings .json file found - using the built-in defaults and the data files.")
    if a.params and not Path(a.params).exists():
        sys.exit(f"params file not found: {a.params}")
    hints = json.loads(Path(a.params).read_text()) if a.params else {}
    params_path = Path(a.params) if a.params else Path("params_derived.json")
    if a.history is None and "_history" in hints:
        a.history = hints["_history"]
    for opt, key in (("core", "_core"), ("field_description", "_field_description"),
                     ("reservoir", "_reservoir")):
        if getattr(a, opt) is None and key in hints:
            setattr(a, opt, hints[key])

    # ---- 2. map picture: ALWAYS ASKED when running interactively ------------
    json_map = hints.get("_map")
    if a.map is None:
        if interactive:
            a.map = str(ask_map_image(json_map))
        elif json_map:
            a.map = json_map
        else:
            imgs = find_files(IMG_EXT)
            if not imgs:
                sys.exit(f"No image files found in {Path.cwd()}")
            a.map = str(pick_file("Map image", imgs, "map.png", False))
    if not Path(a.map).exists():
        sys.exit(f"map file not found: {a.map}\n(current folder: {Path.cwd()})")
    rgb, map_used = load_map_image(a.map, a.pdf_dpi, a.pdf_page)
    print(f"\nUsing map: {map_used.name}  ({rgb.shape[1]} x {rgb.shape[0]} px)")

    # values saved in the JSON belong to ONE specific map picture:
    # use them only if the chosen picture is that same map
    map_name = Path(a.map).name.lower()
    same_map = bool(json_map) and map_name in (Path(json_map).name.lower(),
                                               Path(map_used).name.lower())
    is_l15_map = map_name == "map.png"
    saved_px = saved_m = None
    saved_seeds = None
    saved_depth, saved_gwc, saved_fault, saved_auto = None, None, None, False
    saved_label = "from the params file"
    if same_map:
        saved_px, saved_m = hints.get("_scalebar_px"), hints.get("_scalebar_m")
        saved_seeds = seeds_to_str(hints.get("_seed_wells"))
        saved_depth = [tuple(p) for p in hints.get("_depth_points", [])] or None
        saved_gwc = hints.get("_gwc_depth_m")
        saved_fault = hints.get("_fault_color")
        saved_auto = bool(hints.get("_auto_contours"))
        if a.field_color is None and hints.get("_field_color") is not None:
            fcol = hints["_field_color"]
            a.field_color = ",".join(str(int(v)) for v in fcol) if isinstance(fcol, list) else fcol
        if a.color_tol is None and hints.get("_color_tol") is not None:
            a.color_tol = hints["_color_tol"]
    elif is_l15_map:                                  # built-in values for the original map
        saved_px, saved_m, saved_seeds, saved_label = 64.0, 1000.0, "256,167", "built-in for map.png"
    if a.color_tol is None:
        a.color_tol = 60
    changed = False

    # ---- 3. production history ----------------------------------------------
    if a.history is None:
        hs = find_files(HIST_EXT, valid=lambda f: not is_core_file(f))
        if not hs:
            print("No production-history file (.xlsx/.csv) found - running without history.")
        else:
            h = pick_file("Production history", hs, "l15-02.xlsx", interactive,
                          allow_none="No production history")
            a.history = str(h) if h else None
    elif str(a.history).lower() == "none":
        a.history = None
    if a.history and not Path(a.history).exists():
        sys.exit(f"history file not found: {a.history}")

    # ---- 3b. core analysis + field description (the parameters are calculated from these) --
    if a.core is None:
        cs = find_files(HIST_EXT, valid=is_core_file)
        if cs:
            c_ = pick_file("Core analysis", cs, "core_analyses_l15-02.xlsx", interactive,
                           allow_none="No core analysis")
            a.core = str(c_) if c_ else None
    elif str(a.core).lower() == "none":
        a.core = None
    if a.core and not Path(a.core).exists():
        sys.exit(f"core-analysis file not found: {a.core}")
    if a.field_description is None:
        ds = find_files({".docx"})
        if ds:
            d_ = pick_file("Field description (.docx)", ds, "short_description_of_field.docx",
                           interactive, allow_none="No field description")
            a.field_description = str(d_) if d_ else None
    elif str(a.field_description).lower() == "none":
        a.field_description = None
    if a.field_description and not Path(a.field_description).exists():
        sys.exit(f"field-description file not found: {a.field_description}")

    # ---- 4. scale of THIS map -----------------------------------------------
    if a.m_per_px or (a.scalebar_px and a.scalebar_m):
        print("\nMap scale was given on the command line (--m-per-px / --scalebar-px / "
              "--scalebar-m), so it is NOT asked. Remove those options to be asked.")
    if a.m_per_px is None and not (a.scalebar_px and a.scalebar_m):
        if interactive:
            px, m, ch = choose_scale(rgb, saved_px, saved_m, saved_label)
            a.scalebar_px, a.scalebar_m = px, m
            changed = changed or ch
        elif saved_px and saved_m:
            a.scalebar_px, a.scalebar_m = saved_px, saved_m
        else:
            sys.exit("Give the map scale: --m-per-px  or  --scalebar-px + --scalebar-m "
                     "(or run without --no-ask to click it on the map)")

    # ---- 5. existing wells on THIS map --------------------------------------
    if a.seed_wells is None:
        if interactive:
            a.seed_wells, ch = choose_wells(rgb, saved_seeds, saved_label)
            changed = changed or ch
        else:
            a.seed_wells = saved_seeds or ""

    # ---- 6. depth labels on THIS map ----------------------------------------
    auto_contours = False
    if a.depth_points:
        depth_pts, gwc_m = parse_xyz(a.depth_points), a.gwc_depth
    elif a.auto_contours:
        auto_contours, depth_pts, gwc_m = True, [], saved_gwc
    elif interactive:
        depth_pts, gwc_m, ch = choose_depth(rgb, saved_depth, saved_gwc, saved_label, saved_auto)
        changed = changed or ch
        if isinstance(depth_pts, str) and depth_pts == "auto":
            auto_contours, depth_pts, gwc_m = True, [], None
    elif saved_auto and not saved_depth:
        auto_contours, depth_pts, gwc_m = True, [], saved_gwc
    else:
        depth_pts, gwc_m = (saved_depth or []), saved_gwc
    if a.gwc_depth is not None:
        gwc_m = a.gwc_depth

    # ---- 6b. how faults are drawn on THIS map -------------------------------------
    if a.fault_color is None:
        if interactive:
            a.fault_color, ch = choose_faults(rgb, saved_fault)
            changed = changed or ch
        else:
            a.fault_color = saved_fault or ""

    # ---- inputs: settings + parameters CALCULATED from the data files -----------
    prm = dict(DEFAULTS)
    prm.update({k: v for k, v in hints.items() if not k.startswith("_")})
    core = read_core(a.core, prm["net_phi_cutoff"], prm["net_k_cutoff_md"]) if a.core else None
    hand_sel, hand_name = None, None
    if a.field_description:
        hand = read_field_description(a.field_description)
        if hand and hand["reservoirs"]:
            names = list(hand["reservoirs"])
            want = a.reservoir or prm.get("name")
            idx = next((i for i, n_ in enumerate(names)
                        if want and n_.lower() == str(want).lower()), None)
            if idx is None:
                dflt = next((i for i, n_ in enumerate(names) if n_.lower() == "l15b-a"), 0)
                idx = (ask_choice("RESERVOIR - which row of the field-description table is this field?",
                                  names, dflt) if interactive and len(names) > 1 else dflt)
            hand_name = names[idx]
            hand_sel = dict(hand["reservoirs"][hand_name])
            hand_sel["gas_rows"] = hand["gas_rows"]
            if "name" not in hints:
                prm["name"] = hand_name
            print(f"\nField description: {Path(a.field_description).name} -> reservoir {hand_name}")
    if core:
        print(f"Core analysis: {Path(a.core).name} -> {core['n']} samples, "
              f"{core['top_m']:.0f}-{core['bottom_m']:.0f} m")
    ptab = ParamTable()
    resolve_params(prm, hints, core, hand_sel, a.prefer_handout, ptab)
    miss = [k for k in REQUIRED if k != "net_pay_m" and prm.get(k) is None]
    if miss:
        sys.exit(f"Missing parameters: {miss}\nGive a core-analysis file and/or a field-description "
                 "(.docx), or put them in the settings JSON.")
    if a.m_per_px:
        m_per_px = a.m_per_px
    else:
        m_per_px = a.scalebar_m / a.scalebar_px
    print(f"Map scale: {m_per_px:.2f} m per pixel")
    if a.sealing:
        prm["leaky_faults"] = False
    if str(a.fault_color).lower() == "none":
        prm["use_faults"] = False
    elif a.fault_color:
        prm["fault_color"] = [int(v) for v in str(a.fault_color).split(",")]

    # ---- 7. platform reach (optional) -----------------------------------------
    seeds = parse_xy(a.seed_wells)
    platform = parse_xy(a.platform)[0] if a.platform else (seeds[0] if seeds else None)
    if a.reach is None:
        if interactive and platform is not None:
            print("\nPLATFORM REACH - wells are drilled from the platform (the first existing well "
                  "unless --platform is given); they cannot go beyond its horizontal reach.")
            a.reach = ask_optional_number("Maximum horizontal reach from the platform in metres "
                                          "(Enter = keep the value shown / no limit)",
                                          prm["platform_reach_m"])
        else:
            a.reach = prm["platform_reach_m"]

    # offer to remember the map / scale / wells / depth for next time
    if interactive and changed and not a.no_save:
        ans = input(f"\nSave this map, scale, wells and depth labels into {params_path.name} so you "
                    "do not have to repeat the clicking next time? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            d = json.loads(params_path.read_text()) if params_path.exists() else {}
            d["_core"], d["_field_description"], d["_reservoir"] = a.core, a.field_description, a.reservoir
            d["_fault_color"] = a.fault_color or None
            d["_map"] = map_used.name if map_used.parent.resolve() == Path.cwd().resolve() else str(map_used)
            d["_scalebar_px"] = round(float(a.scalebar_px), 2) if a.scalebar_px else None
            d["_scalebar_m"] = float(a.scalebar_m) if a.scalebar_m else None
            d["_seed_wells"] = [list(p) for p in seeds]
            d["_depth_points"] = [[round(p[0], 1), round(p[1], 1), p[2]] for p in depth_pts]
            d["_auto_contours"] = bool(auto_contours)
            d["_gwc_depth_m"] = gwc_m
            if a.m_per_px:
                d["_scalebar_px"], d["_scalebar_m"] = 1.0, round(float(a.m_per_px), 4)
            params_path.write_text(json.dumps(d, indent=2))
            print(f"   saved into {params_path.name}")

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    P = {n: writable(out / n) for n in ("wells_map.png", "forecast.png", "history_match.png",
                                        "forecast.csv", "summary.json")}

    fc_color = tuple(int(v) for v in a.field_color.split(",")) if a.field_color else None
    mask, raw_px = segment_field(rgb, fc_color, a.color_tol, a.merge_disconnected_pieces)
    area_m2 = float(mask.sum()) * m_per_px ** 2
    print(f"Mapped field: {area_m2 / 1e6:.2f} km2  ({int(mask.sum())} pixels = "
          f"{100.0 * mask.sum() / mask.size:.1f}% of the picture, {m_per_px:.3f} m per pixel)")
    if (mask.shape[0] // max(1, int(round(25.0 / m_per_px))) < 20
            or mask.shape[1] // max(1, int(round(25.0 / m_per_px))) < 20):
        sys.exit(f"\nThe map scale ({m_per_px:.4f} m per pixel) is impossible for this picture: the whole "
                 f"map would be only {mask.shape[1] * m_per_px:.0f} m wide. Run again and give the scale-bar "
                 f"length in METRES (5 km = 5000) next to the length you measured in PIXELS on this picture.")
    if area_m2 / 1e6 < 1.0 or area_m2 / 1e6 > 2000.0:
        print("WARNING: that field size looks wrong. The usual reason is the map scale: when you give the "
              "length of the scale bar it must be in METRES (5 km = 5000), and the pixels must be "
              "measured on THIS picture. If the size is right, the green area may not be the field "
              "(e.g. the legend swatch) - use --field-color R,G,B or check the picture.")

    # ---- gas in place ---------------------------------------------------------
    t_k = (prm["reservoir_temp_c"] if prm["reservoir_temp_c"] is not None else
           prm["surface_temp_c"] + prm["geothermal_gradient_c_per_100m"] * prm["depth_m"] / 100.0) + 273.15
    p_i, p_wf = prm["reservoir_pressure_bar"], prm["operating_pressure_bar"]
    pvt = GasPVT(t_k, prm["gas_specific_gravity"], p_i)
    bg = pvt.bg(p_i)
    resolve_net_pay(prm, hints, hand_sel, core, area_m2, bg, ptab, a.prefer_handout)
    hcpv = area_m2 * prm["net_pay_m"] * prm["porosity"] * prm["gas_saturation"]
    giip_vol = hcpv / bg
    giip = prm["giip_bcm"] * 1e9 if prm["giip_bcm"] else giip_vol
    rf = prm["recovery_factor"]
    ratio = p_wf / p_i
    if not 0 < ratio < 1:
        sys.exit("operating_pressure_bar must be below reservoir_pressure_bar")
    life_m = int(prm["field_life_years"] * 12)

    # ---- per-well initial rate ------------------------------------------------
    hist = None
    cgr = prm["cgr_m3_per_million_m3"] * 1e-6
    rate_source = ""
    z_i = pvt.z(p_i)
    mu = gas_viscosity_cp(p_i, t_k, prm["gas_specific_gravity"], z_i)

    def darcy(n):                      # upper-bound deliverability
        re = math.sqrt(area_m2 / n / math.pi)
        kf = prm["permeability_md"] * (prm["net_pay_m"] * 3.28084)
        dp2 = (p_i * 14.5038) ** 2 - (p_wf * 14.5038) ** 2
        den = 1422 * (t_k * 1.8) * mu * z_i * (
            math.log(re / prm["wellbore_radius_m"]) - 0.75 + prm["skin"])
        return kf * dp2 / den * 28.3168            # m3/day

    if a.history:
        hist, q_hist, cgr = load_history(a.history, a.history_wells)
        h5 = hist.tail(60)
        ptab.add("cgr_m3_per_million_m3", round(cgr * 1e6, 2), "history",
                f"condensate / gas over the whole history (last 5 years: "
                f"{h5['cond'].sum() / max(h5['gas'].sum(), 1.0) * 1e6:.1f})",
                {"default": prm["cgr_m3_per_million_m3"]})
    else:
        ptab.add("cgr_m3_per_million_m3", prm["cgr_m3_per_million_m3"], "json" if
                "cgr_m3_per_million_m3" in hints else "default", "no production history given")
    ptab.show()
    if prm["operating_pressure_bar"] / prm["reservoir_pressure_bar"] > 0.6:
        print(f"\nWARNING: the operating (flowing) pressure is "
              f"{prm['operating_pressure_bar'] / prm['reservoir_pressure_bar']:.0%} of the reservoir "
              f"pressure, so the tank model can only recover a small part of the gas. A late-life "
              f"flowing pressure of 20-40% is more usual: set operating_pressure_bar in the JSON.")
    if a.save_params:
        keep = ("net_pay_m", "porosity", "gas_saturation", "depth_m", "reservoir_pressure_bar",
                "operating_pressure_bar", "permeability_md", "gas_specific_gravity",
                "gas_price_usd_per_1000m3", "recovery_factor", "cgr_m3_per_million_m3", "name")
        Path(a.save_params).write_text(json.dumps(
            {**{k: prm[k] for k in keep if prm.get(k) is not None},
             "_parameter_sources": {r["parameter"]: f"{r['source']}: {r['how']}".strip(": ")
                                    for r in ptab.as_list()}}, indent=2, default=float))
        print(f"   calculated parameters written to {a.save_params}")
    if a.well_rate_mm3d:
        q_w0_fn = lambda n: a.well_rate_mm3d * 1e6
        rate_source = "user-given rate"
    elif hist is not None:
        q_w0_fn = lambda n: q_hist
        rate_source = "measured from history (first 3 producing years)"
    else:
        q_w0_fn = darcy
        rate_source = "Darcy deliverability (UPPER BOUND - real wells give less)"
    q_min = (prm["min_rate_mm3_per_day"] * 1e6 if prm["min_rate_mm3_per_day"]
             else prm["opex_musd_per_year"] * 1e6 /
             (prm["gas_price_usd_per_1000m3"] / 1000.0 * 365.25))
    if a.rate_cap_mm3d is None:
        if interactive:
            print("\nPLANT / EXPORT CAPACITY - without a capacity limit the model keeps rewarding "
                  "extra wells, because more wells just produce the same gas faster.")
            a.rate_cap_mm3d = ask_optional_number(
                "Maximum plant/export rate in million m3/day (Enter = no limit)")
        elif prm.get("rate_cap_mm3d"):
            a.rate_cap_mm3d = float(prm["rate_cap_mm3d"])
    cap = a.rate_cap_mm3d * 1e6 if a.rate_cap_mm3d else None
    tol = a.npv_tolerance if a.npv_tolerance is not None else float(prm["npv_tolerance"])

    # ---- history match --------------------------------------------------------
    hm = None
    figs = []
    g_used, g_scale = giip, 1.0
    match = a.match_history and bool(prm["match_history"])
    if hist is not None:
        n_hm = a.history_wells
        months_h = len(hist)

        def model_gp(G):
            return run_forecast(G, pvt, p_i, ratio, q_w0_fn(n_hm), n_hm, 1.0, months_h)

        def err(scale):
            m = model_gp(giip * scale)["gp"]
            act = (hist["gas"].cumsum().values / 1.0)[:len(m)]
            return float(((m - act) ** 2).mean())

        if match:                                 # history-matched gas in place
            res = minimize_scalar(err, bounds=(0.05, 3.0), method="bounded")
            g_scale = float(res.x)
            g_used = giip * g_scale
        m = model_gp(g_used)
        hm = {"model": m,
              "model_cum_bcm": float(m["gp"][-1] / 1e9),
              "actual_cum_bcm": float(hist["gas"].sum() / 1e9),
              "months": months_h}
        figs.append(plot_history(hist, m, P["history_match.png"]))

    # ---- drainage radius (from the production history) --------------------------
    eur = None
    gas_per_km2 = giip / (area_m2 / 1e6)                       # std m3 per km2
    if a.radius:
        radius_m, radius_src = a.radius, "--radius"
    elif prm["drainage_radius_m"]:
        radius_m, radius_src = float(prm["drainage_radius_m"]), "drainage_radius_m in the JSON"
    else:
        if hist is not None:
            eur = history_eur(hist, prm["eur_abandon_mm3_month"], a.history_wells)
        if eur:
            drained_gas = eur["selected_mm3"] * 1e6 / rf
            radius_m = math.sqrt(drained_gas / gas_per_km2 * 1e6 / math.pi)
            radius_src = (f"production history: EUR {eur['selected_mm3']:.0f} million m3 per well "
                          f"({eur['low_mm3']:.0f}-{eur['high_mm3']:.0f}), recovery factor {rf:.0%}")
        else:
            radius_m, radius_src = 800.0, "DEFAULT 800 m (no usable history - set drainage_radius_m)"
    print(f"Drainage radius: {radius_m:.0f} m  [{radius_src}]")

    # ---- place wells ------------------------------------------------------------
    faults_full = (extract_faults(rgb, int(prm["fault_min_pixels"]), prm.get("fault_color"),
                                  prm.get("fault_color_tol", 80))
                   if prm["use_faults"] else np.zeros(mask.shape, bool))
    contour_info = None
    if auto_contours:
        depth_pts, contour_info = auto_contour_pipeline(rgb, mask, faults_full, map_used, prm, out, a)
    fm = FieldModel(rgb, mask, m_per_px, prm, depth_pts, gwc_m, faults=faults_full)
    if fm.depth is not None and fm.column is None:
        est = estimate_gwc(fm)
        if interactive:
            gwc_m = ask_number("\nDepth of the gas-water contact in metres (Enter = the estimate from the "
                               "deep end of the field edge)", round(est))
        else:
            gwc_m = est
            print(f"   gas-water contact: ESTIMATED {est:.0f} m from the deep end of the field edge "
                  f"(give --gwc-depth to set it)")
        fm.set_gwc(gwc_m)
    if fm.column is None and prm["min_gas_column_m"]:
        print("   (no depth labels: the gas-column rule is not applied)")
    if fm.column is not None and float(np.median(fm.column[fm.gd])) < 0:
        print(f"\nWARNING: the gas column comes out NEGATIVE over most of the field (median "
              f"{float(np.median(fm.column[fm.gd])):.0f} m): the gas-water contact ({gwc_m:g} m) is "
              f"shallower than the depth labels. Check the depth labels and the contact depth - "
              f"they must be in the same depth system.")
    if fm.fd.any():
        print(f"   fault lines cover {100.0 * (fm.fd & fm.gd).sum() / max(fm.gd.sum(), 1):.1f}% of the field; "
              f"{fm.block.max()} fault block(s); faults treated as "
              f"{'LEAKY' if prm['leaky_faults'] else 'SEALING'}")
    else:
        print("   (no faults found on the map)")
    new, _ = plan_wells(fm, seeds, radius_m, prm["leaky_faults"], platform, a.reach, prm)
    if not new:
        print("\nNo new well fits with the current rules. Field area left after each rule:")
        for label, km2 in constraint_funnel(fm, prm, platform, a.reach):
            print(f"   {km2:7.2f} km2  {label}")
        print(f"   (map scale {m_per_px:.2f} m per pixel, mapped field {fm.area_km2:.2f} km2, "
              f"drainage circle {radius_m:.0f} m = {math.pi * radius_m ** 2 / 1e6:.2f} km2, "
              f"minimum gain per well {prm['min_gain_km2']} km2)")
    if not new and not seeds:               # nothing at all: relax the rules one by one
        ladder = []
        if fm.column is not None and prm["min_gas_column_m"]:
            ladder.append(("the gas-column rule was dropped", {"min_gas_column_m": 0.0}))
        ladder += [("the fault stand-off was reduced to 30 m", {"fault_standoff_m": 30.0}),
                   ("faults were ignored", {"fault_standoff_m": 0.0}),
                   ("the edge buffer was reduced to 50 m", {"edge_buffer_m": 50.0}),
                   ("edge buffer, spacing and minimum gain were relaxed (gain 0.05 km2)",
                    {"edge_buffer_m": 0.0, "min_well_spacing_m": 0.0, "min_gain_km2": 0.05})]
        prm_r, used = dict(prm), []
        for text, change in ladder:
            prm_r.update(change)
            used.append(text)
            new, _ = plan_wells(fm, seeds, radius_m, prm["leaky_faults"], platform, a.reach, prm_r)
            if new:
                print("\nWARNING: wells were found only after relaxing the rules - " + "; ".join(used) + ".")
                print("         Check wells_map.png, and set these values in the JSON if you accept them.")
                break
    allw = [tuple(s) for s in seeds] + [(d["x"], d["y"]) for d in new]
    extra_all = [None] * len(seeds) + new
    if not allw:
        sys.exit(f"\nNo well could be placed even after relaxing the rules.\n"
                 f"Key numbers: mapped field {fm.area_km2:.2f} km2, {m_per_px:.3f} m per pixel, "
                 f"drainage circle {radius_m:.0f} m ({math.pi * radius_m ** 2 / 1e6:.2f} km2).\n"
                 "If the mapped field is far from the real size, the scale is wrong. The usual causes:\n"
                 "  * wrong map scale  -> check 'mapped field' above against the real field size\n"
                 "  * the field colour was not found -> use --field-color R,G,B\n"
                 "  * the fault detector took thick contour lines or text for faults -> set "
                 '"use_faults": false in the JSON\n'
                 "  * depth labels / gas-water contact typed in different depth systems "
                 '-> check them, or answer "Skip" at the depth step\n'
                 "  * the drainage circle is smaller than the minimum gain -> set drainage_radius_m "
                 "or min_gain_km2 in the JSON")
    if "max_wells" in hints:
        print("\nNote: 'max_wells' in the params file is ignored - there is no fixed limit "
              "any more (the search stops by itself).")
    if a.n_wells and a.n_wells > len(allw):
        print(f"\nNote: only {len(allw)} wells reach at least {prm['min_gain_km2']} km2 of undrained "
              f"area; lower min_gain_km2 in the JSON to place more.")

    # ---- sweep number of wells ------------------------------------------------
    results = {}
    n = min(a.n_wells, len(allw)) if a.n_wells else 1
    best_npv, since_best, bad_spacing, stop_reason = -float("inf"), 0, 0, ""
    while True:
        if n > len(allw):
            stop_reason = (f"no further well adds at least {prm['min_gain_km2']} km2 of undrained "
                           f"field area")
            break
        w = np.array(allw[:n], float)
        fc = run_forecast(g_used, pvt, p_i, ratio, q_w0_fn(n), n, rf, life_m, q_min, cap)
        cells = voronoi_cells(mask, w)
        st = well_stats(w, cells, mask, m_per_px, extra_all[:n], fm, platform)
        nn = min((s_["nearest_well_m"] for s_ in st if s_["nearest_well_m"]), default=1e9)
        npv = npv_musd(fc, n, cgr, prm)
        ok = nn >= prm["min_well_spacing_m"]
        results[n] = {"fc": fc, "npv": npv, "w": w, "cells": cells,
                      "stats": st, "spacing_ok": ok}
        bad_spacing = 0 if ok else bad_spacing + 1
        if ok and npv > best_npv:
            best_npv, since_best = npv, 0
        else:
            since_best += 1
        if a.n_wells:                   # user forced one number of wells
            break
        if bad_spacing >= 2:
            stop_reason = "wells no longer fit with the minimum spacing"
            break
        if since_best >= 5:
            stop_reason = "NPV has not improved for 5 more wells"
            break
        n += 1
    feas = [k for k in results if results[k]["spacing_ok"]] or list(results)
    best_max_n = max(feas, key=lambda k: results[k]["npv"])       # strictly highest NPV
    npv_max = results[best_max_n]["npv"]
    threshold = npv_max - tol * abs(npv_max)
    best_flat_n = min(k for k in feas if results[k]["npv"] >= threshold)  # fewest wells, ~same NPV
    best = min(a.n_wells, len(allw)) if a.n_wells else best_flat_n

    # ---- outputs --------------------------------------------------------------
    b = results[best]
    drained_best = drained_mask(fm, [tuple(p) for p in b["w"]], radius_m, prm["leaky_faults"])
    uncovered = _upsample(fm.gd & ~drained_best, fm.D, mask.shape) & mask
    uncovered_km2 = float(uncovered.sum() * m_per_px ** 2 / 1e6)
    figs.insert(0, plot_wells(rgb, mask, b["cells"], b["w"], min(len(seeds), best), m_per_px,
                              b["stats"], f"{prm['name']}: {best} wells, area "
                              f"{area_m2 / 1e6:.1f} km2", P["wells_map.png"],
                              extra={"radius_m": radius_m, "uncovered": uncovered,
                                     "platform": platform, "reach_m": a.reach}))
    figs.insert(1, plot_forecast(results, best, g_used, rf, cgr, P["forecast.png"]))

    fc = b["fc"]
    n_m = len(fc["vol"])
    yr = (np.arange(n_m) // 12) + 1
    tab = pd.DataFrame({"year": yr, "gas_million_m3": fc["vol"] / 1e6,
                        "condensate_m3": fc["vol"] * cgr,
                        "pressure_bar": fc["p"], "cum_gas_bcm": fc["gp"] / 1e9,
                        "rate_million_m3_d": fc["q"] / 1e6})
    yearly = tab.groupby("year").agg(gas_million_m3=("gas_million_m3", "sum"),
                                     condensate_m3=("condensate_m3", "sum"),
                                     avg_rate_million_m3_d=("rate_million_m3_d", "mean"),
                                     end_pressure_bar=("pressure_bar", "last"),
                                     cum_gas_bcm=("cum_gas_bcm", "last")).round(3)
    yearly.to_csv(P["forecast.csv"])

    summary = {
        "map_file": map_used.name,
        "parameters": ptab.as_list(),
        "field_area_km2": round(area_m2 / 1e6, 2),
        "area_pixels_before_hole_filling": raw_px, "metres_per_pixel": round(m_per_px, 2),
        "reservoir_temp_c": round(t_k - 273.15, 1), "Z_initial": round(z_i, 3),
        "Bg": round(bg, 5), "gas_viscosity_cp": round(mu, 4),
        "GIIP_volumetric_bcm": round(giip_vol / 1e9, 2),
        "GIIP_basis_bcm": round(giip / 1e9, 2),
        "GIIP_used_bcm": round(g_used / 1e9, 2), "history_match_scale": round(g_scale, 2),
        "recoverable_bcm": round(rf * g_used / 1e9, 2),
        "initial_rate_per_well_million_m3_d": round(q_w0_fn(best) / 1e6, 3),
        "rate_source": rate_source, "CGR_m3_per_million_m3": round(cgr * 1e6, 2),
        "economic_limit_million_m3_d": round(q_min / 1e6, 3),
        "well_location_method": {
            "drainage_radius_m": round(radius_m, 0), "radius_source": radius_src,
            "history_eur": eur, "faults": "leaky" if prm["leaky_faults"] else "sealing",
            "fault_blocks": int(fm.block.max()), "depth_points_used": len(depth_pts),
            "contours": contour_info,
            "gwc_depth_m": gwc_m, "platform_px": platform, "platform_reach_m": a.reach,
            "min_gain_km2": prm["min_gain_km2"], "uncovered_field_km2": round(uncovered_km2, 2)},
        "recommended_wells": best,
        "max_npv_wells": best_max_n, "npv_tolerance": tol,
        "rate_cap_million_m3_d": a.rate_cap_mm3d,
        "NPV_musd_by_wells": {n: round(r["npv"], 1) for n, r in results.items()},
        "recovery_bcm_by_wells": {n: round(r["fc"]["gp"][-1] / 1e9, 2) for n, r in results.items()},
        "wells": b["stats"],
        "history_check": None if hm is None else {
            k: round(hm[k], 3) if isinstance(hm[k], float) else hm[k]
            for k in ("model_cum_bcm", "actual_cum_bcm", "months")},
    }
    P["summary.json"].write_text(json.dumps(summary, indent=2, default=float))

    # ---- console report -------------------------------------------------------
    print(f"\n=== {prm['name']} ===")
    print(f"Field area           : {area_m2 / 1e6:.2f} km2  ({m_per_px:.1f} m/pixel)")
    print(f"Reservoir T / Z / Bg : {t_k - 273.15:.0f} C / {z_i:.3f} / {bg:.5f}")
    print(f"GIIP                 : {giip / 1e9:.2f} Bcm" +
          (" (operator value)" if prm["giip_bcm"] else " (volumetric)") +
          (f"   -> history-matched: {g_used / 1e9:.2f} Bcm (x{g_scale:.2f})"
           if match and hist is not None else ""))
    print(f"Recoverable (RF {rf:.0%}): {rf * g_used / 1e9:.2f} Bcm")
    print(f"Initial rate / well  : {q_w0_fn(best) / 1e6:.3f} million m3/d [{rate_source}]")
    if hm:
        print(f"History check        : model {hm['model_cum_bcm']:.2f} Bcm vs actual "
              f"{hm['actual_cum_bcm']:.2f} Bcm over {hm['months']} months")
    print("\n wells | recovery Bcm | NPV MUSD | gain vs n-1 | min spacing ok")
    prev = None
    for n, r in results.items():
        gain = "     -     " if prev is None else f"{r['npv'] - prev:+10.1f} "
        prev = r["npv"]
        mark = ""
        if n == best:
            mark = "  <-- recommended"
        if n == best_max_n and n != best:
            mark = "  <-- highest NPV"
        elif n == best_max_n:
            mark += " (also highest NPV)"
        print(f"  {n:>3}  |    {r['fc']['gp'][-1] / 1e9:5.2f}     | {r['npv']:8.1f} | {gain} | "
              f"{'yes' if r['spacing_ok'] else 'NO '}{mark}")
    if not a.n_wells:
        print(f"\nRecommended = fewest wells whose NPV is within {tol:.0%} of the maximum "
              f"({npv_max:.1f} MUSD at {best_max_n} wells). The curve is usually very flat near "
              f"the top, so extra wells there add cost and risk but almost no value.\n"
              f"Change this with --npv-tolerance (0 = strictly the highest NPV).")
    if stop_reason:
        print(f"\nSearch stopped at {max(results)} wells: {stop_reason}.")
        if best == max(k for k in results if results[k]["spacing_ok"]) and "spacing" in stop_reason:
            print("NOTE: the best NPV is at the spacing limit - the field cannot take more wells "
                  "at this minimum spacing (min_well_spacing_m in the JSON).")
    print(f"\nWell locations ({best} wells; W1-W{len(seeds)} existing):" if seeds
          else f"\nWell locations ({best} wells):")
    for s in b["stats"]:
        line = (f"  {s['well']}: x={s['x_px']:.0f}px y={s['y_px']:.0f}px | "
                f"drainage {s['drainage_km2']} km2 | nearest well {s['nearest_well_m']} m | "
                f"edge {s['dist_to_edge_m']} m")
        if "new_undrained_km2" in s:
            line += f" | new area reached {s['new_undrained_km2']} km2"
        if "gas_column_m" in s:
            line += f" | gas column {s['gas_column_m']} m"
        if "dist_to_platform_m" in s:
            line += f" | {s['dist_to_platform_m']:.0f} m from platform"
        print(line)
    print(f"Field area not reached by any well (purple on the map): {uncovered_km2:.2f} km2")
    print(f"\nIf no picture window opened, open these files yourself:")
    print(f"    {P['wells_map.png'].resolve()}")
    print(f"    {P['forecast.png'].resolve()}")
    print(f"\nFiles written to {out.resolve()}:")
    print(f"MAP WITH WELL CIRCLES: {P['wells_map.png'].resolve()}")
    files = [P["wells_map.png"], P["forecast.png"]]
    if hm:
        files.append(P["history_match.png"])
    for f in files + [P["forecast.csv"], P["summary.json"]]:
        print("   ", f.name)
    if not a.no_show:
        if matplotlib.get_backend().lower() == "agg":     # no window system available
            open_files(files)
        else:
            print("\nShowing the pictures - close the windows to finish.")
            plt.show()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        sys.exit(1)