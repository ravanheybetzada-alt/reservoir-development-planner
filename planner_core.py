"""
planner_core.py - the calculation behind the web app (no questions, no print-outs, no windows).

analyse(cfg) takes ONE dictionary with everything the user chose (files, scale, wells, depth,
settings) and returns ONE dictionary with the results (tables, figures, warnings, numbers).
It re-uses the functions of reservoir_planner.py, so the web app and the command-line program
give the same answers.
"""
import math
import os
import types
from pathlib import Path

os.environ.setdefault("RP_HEADLESS", "1")
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import reservoir_planner as rp


def analyse(cfg):
    out = Path(cfg.get("out_dir") or "out")
    out.mkdir(parents=True, exist_ok=True)
    warn = []
    # ---- map, scale, field -----------------------------------------------------------
    rgb, map_used = rp.load_map_image(cfg["map_path"], cfg.get("pdf_dpi", 200.0), cfg.get("pdf_page", 1))
    m_per_px = float(cfg["m_per_px"])
    fc = tuple(int(v) for v in cfg["field_color"]) if cfg.get("field_color") else None
    mask, raw_px = rp.segment_field(rgb, fc, 60, False)
    area_m2 = float(mask.sum()) * m_per_px ** 2
    if area_m2 / 1e6 < 1.0 or area_m2 / 1e6 > 2000.0:
        warn.append(f"The mapped field is {area_m2 / 1e6:.2f} km2 - that looks wrong. Check the scale "
                    "(scale-bar length in METRES) and the field colour.")
    if (mask.shape[0] // max(1, int(round(25.0 / m_per_px))) < 20
            or mask.shape[1] // max(1, int(round(25.0 / m_per_px))) < 20):
        raise ValueError(f"The map scale ({m_per_px:.4f} m per pixel) is impossible for this picture.")
    # ---- settings ----------------------------------------------------------------------
    user = {k: v for k, v in (cfg.get("settings") or {}).items() if v is not None}
    prm = dict(rp.DEFAULTS)
    prm.update(user)
    prm["leaky_faults"] = bool(cfg.get("leaky", True))
    if cfg.get("fault_mode") == "none":
        prm["use_faults"] = False
    elif cfg.get("fault_mode") == "color" and cfg.get("fault_color"):
        prm["fault_color"] = [int(v) for v in cfg["fault_color"]]
    reach = cfg.get("reach_m") or prm["platform_reach_m"]
    seeds = [tuple(map(float, p)) for p in (cfg.get("seeds") or [])]
    platform = tuple(cfg["platform"]) if cfg.get("platform") else (seeds[0] if seeds else None)
    # ---- parameters: calculated from the data files ----------------------------------
    core = rp.read_core(cfg["core_path"], prm["net_phi_cutoff"], prm["net_k_cutoff_md"]) if cfg.get("core_path") else None
    hand_sel, hand_name = None, None
    if cfg.get("handout_path"):
        hand = rp.read_field_description(cfg["handout_path"])
        if hand and hand["reservoirs"]:
            names = list(hand["reservoirs"])
            hand_name = cfg.get("reservoir") if cfg.get("reservoir") in names else names[0]
            hand_sel = dict(hand["reservoirs"][hand_name])
            hand_sel["gas_rows"] = hand["gas_rows"]
            if "name" not in user:
                prm["name"] = hand_name
    ptab = rp.ParamTable()
    rp.resolve_params(prm, user, core, hand_sel, bool(cfg.get("prefer_handout")), ptab)
    miss = [k for k in rp.REQUIRED if k != "net_pay_m" and prm.get(k) is None]
    if miss:
        raise ValueError(f"Missing parameters: {miss}. Upload a core analysis and/or the field description, "
                         "or type them in the Parameters tab.")
    t_k = (prm["reservoir_temp_c"] if prm["reservoir_temp_c"] is not None else
           prm["surface_temp_c"] + prm["geothermal_gradient_c_per_100m"] * prm["depth_m"] / 100.0) + 273.15
    p_i, p_wf = prm["reservoir_pressure_bar"], prm["operating_pressure_bar"]
    pvt = rp.GasPVT(t_k, prm["gas_specific_gravity"], p_i)
    bg = pvt.bg(p_i)
    rp.resolve_net_pay(prm, user, hand_sel, core, area_m2, bg, ptab, bool(cfg.get("prefer_handout")))
    hcpv = area_m2 * prm["net_pay_m"] * prm["porosity"] * prm["gas_saturation"]
    giip_vol = hcpv / bg
    giip = prm["giip_bcm"] * 1e9 if prm["giip_bcm"] else giip_vol
    rf = prm["recovery_factor"]
    ratio = p_wf / p_i
    if not 0 < ratio < 1:
        raise ValueError("The operating pressure must be below the reservoir pressure.")
    if ratio > 0.6:
        warn.append(f"The operating pressure is {ratio:.0%} of the reservoir pressure, so the tank model can "
                    "only recover a small part of the gas. A late-life flowing pressure of 20-40% is more usual.")
    life_m = int(prm["field_life_years"] * 12)
    # ---- production history, rate per well ----------------------------------------
    hist, q_hist = None, None
    cgr = prm["cgr_m3_per_million_m3"] * 1e-6
    z_i = pvt.z(p_i)
    mu = rp.gas_viscosity_cp(p_i, t_k, prm["gas_specific_gravity"], z_i)

    def darcy(n):
        re = math.sqrt(area_m2 / n / math.pi)
        kf = prm["permeability_md"] * (prm["net_pay_m"] * 3.28084)
        dp2 = (p_i * 14.5038) ** 2 - (p_wf * 14.5038) ** 2
        den = 1422 * (t_k * 1.8) * mu * z_i * (math.log(re / prm["wellbore_radius_m"]) - 0.75 + prm["skin"])
        return kf * dp2 / den * 28.3168

    n_hist = int(cfg.get("history_wells", 1))
    if cfg.get("history_path"):
        hist, q_hist, cgr = rp.load_history(cfg["history_path"], n_hist)
        ptab.add("cgr_m3_per_million_m3", round(cgr * 1e6, 2), "history", "condensate / gas over the whole history")
    if cfg.get("well_rate_mm3d"):
        q_w0_fn, rate_source = (lambda n: cfg["well_rate_mm3d"] * 1e6), "user-given rate"
    elif hist is not None:
        q_w0_fn, rate_source = (lambda n: q_hist), "measured from the production history"
    else:
        q_w0_fn, rate_source = darcy, "Darcy deliverability (UPPER BOUND)"
        warn.append("No production history: the well rate is the Darcy upper bound and the drainage radius is a default.")
    q_min = (prm["min_rate_mm3_per_day"] * 1e6 if prm["min_rate_mm3_per_day"] else
             prm["opex_musd_per_year"] * 1e6 / (prm["gas_price_usd_per_1000m3"] / 1000.0 * 365.25))
    cap = float(cfg["rate_cap_mm3d"]) * 1e6 if cfg.get("rate_cap_mm3d") else None
    tol = float(cfg.get("npv_tolerance", prm["npv_tolerance"]))
    # ---- history match (optional) ---------------------------------------------------
    g_used, g_scale, hm = giip, 1.0, None
    if hist is not None:
        months_h = len(hist)
        model_gp = lambda G: rp.run_forecast(G, pvt, p_i, ratio, q_w0_fn(n_hist), n_hist, 1.0, months_h)
        if cfg.get("match_history"):
            def err(scale):
                m = model_gp(giip * scale)["gp"]
                return float(((m - hist["gas"].cumsum().values[:len(m)]) ** 2).mean())
            g_scale = float(minimize_scalar(err, bounds=(0.05, 3.0), method="bounded").x)
            g_used = giip * g_scale
        mdl = model_gp(g_used)
        hm = {"model": mdl, "model_cum_bcm": float(mdl["gp"][-1] / 1e9),
              "actual_cum_bcm": float(hist["gas"].sum() / 1e9), "months": months_h}
    # ---- drainage radius ----------------------------------------------------------------
    eur = None
    gas_per_km2 = giip / (area_m2 / 1e6)
    if cfg.get("radius_m"):
        radius_m, radius_src = float(cfg["radius_m"]), "typed in"
    elif prm["drainage_radius_m"]:
        radius_m, radius_src = float(prm["drainage_radius_m"]), "settings"
    else:
        eur = rp.history_eur(hist, prm["eur_abandon_mm3_month"], n_hist) if hist is not None else None
        if eur:
            radius_m = math.sqrt(eur["selected_mm3"] * 1e6 / rf / gas_per_km2 * 1e6 / math.pi)
            radius_src = (f"production history: EUR {eur['selected_mm3']:.0f} million m3 per well "
                          f"({eur['low_mm3']:.0f}-{eur['high_mm3']:.0f}), recovery factor {rf:.0%}")
        else:
            radius_m, radius_src = 800.0, "default 800 m (no usable history)"
            warn.append("Drainage radius is the 800 m default - give a production history or type a radius.")
    # ---- faults, contours, depth surface --------------------------------------------------
    faults_full = (rp.extract_faults(rgb, int(prm["fault_min_pixels"]), prm.get("fault_color"),
                                     prm.get("fault_color_tol", 80)) if prm["use_faults"] else np.zeros(mask.shape, bool))
    depth_pts, contour_info, gwc_m = list(cfg.get("depth_pts") or []), None, cfg.get("gwc_m")
    if cfg.get("depth_mode") == "auto":
        ns = types.SimpleNamespace(contour_interval=cfg.get("contour_interval"), ocr_scale=3.0,
                                   ocr_angle_step=30, redo_ocr=bool(cfg.get("redo_ocr")))
        depth_pts, contour_info = rp.auto_contour_pipeline(rgb, mask, faults_full, Path(cfg["map_path"]), prm, out, ns)
        if contour_info is None:
            warn.append("Automatic contours need an OCR package: python -m pip install rapidocr-onnxruntime")
    fm = rp.FieldModel(rgb, mask, m_per_px, prm, depth_pts, gwc_m, faults=faults_full)
    if fm.depth is not None and fm.column is None:
        gwc_m = rp.estimate_gwc(fm)
        fm.set_gwc(gwc_m)
        warn.append(f"The gas-water contact was ESTIMATED as {gwc_m:.0f} m (deep end of the field edge). "
                    "Type the real value in the Depth tab if you know it.")
    if fm.column is not None and float(np.median(fm.column[fm.gd])) < 0:
        warn.append("The gas column is NEGATIVE over most of the field: the gas-water contact is shallower "
                    "than the depth labels. Check both.")
    if fm.column is None:
        warn.append("No depth information: the gas-column rule is not applied.")
    # ---- place wells ---------------------------------------------------------------------
    new, _ = rp.plan_wells(fm, seeds, radius_m, prm["leaky_faults"], platform, reach, prm)
    funnel = rp.constraint_funnel(fm, prm, platform, reach) if not new else None
    if not new and not seeds:
        ladder = []
        if fm.column is not None and prm["min_gas_column_m"]:
            ladder.append(("the gas-column rule was dropped", {"min_gas_column_m": 0.0}))
        ladder += [("the fault stand-off was reduced to 30 m", {"fault_standoff_m": 30.0}),
                   ("faults were ignored", {"fault_standoff_m": 0.0}),
                   ("the edge buffer was reduced to 50 m", {"edge_buffer_m": 50.0}),
                   ("edge buffer, spacing and minimum gain were relaxed", {"edge_buffer_m": 0.0, "min_well_spacing_m": 0.0, "min_gain_km2": 0.05})]
        prm_r, used = dict(prm), []
        for text, change in ladder:
            prm_r.update(change)
            used.append(text)
            new, _ = rp.plan_wells(fm, seeds, radius_m, prm["leaky_faults"], platform, reach, prm_r)
            if new:
                warn.append("Wells were found only after relaxing the rules: " + "; ".join(used) + ".")
                break
    allw = [tuple(s) for s in seeds] + [(d["x"], d["y"]) for d in new]
    extra_all = [None] * len(seeds) + new
    if not allw:
        raise ValueError(f"No well could be placed (mapped field {fm.area_km2:.2f} km2, {m_per_px:.3f} m per pixel, "
                         f"drainage circle {radius_m:.0f} m). Field area left after each rule: "
                         + "; ".join(f"{k:.2f} km2 {t}" for t, k in (funnel or [])))
    # ---- sweep the number of wells -----------------------------------------------------------
    results, best_npv, since_best, bad_sp, stop_reason = {}, -float("inf"), 0, 0, ""
    forced = cfg.get("n_wells")
    n = min(int(forced), len(allw)) if forced else 1
    while True:
        if n > len(allw):
            stop_reason = f"no further well adds at least {prm['min_gain_km2']} km2 of undrained field area"
            break
        w = np.array(allw[:n], float)
        fc_ = rp.run_forecast(g_used, pvt, p_i, ratio, q_w0_fn(n), n, rf, life_m, q_min, cap)
        cells = rp.voronoi_cells(mask, w)
        st_ = rp.well_stats(w, cells, mask, m_per_px, extra_all[:n], fm, platform)
        nn = min((s_["nearest_well_m"] for s_ in st_ if s_["nearest_well_m"]), default=1e9)
        npv = rp.npv_musd(fc_, n, cgr, prm)
        ok = nn >= prm["min_well_spacing_m"]
        results[n] = {"fc": fc_, "npv": npv, "w": w, "cells": cells, "stats": st_, "spacing_ok": ok}
        bad_sp = 0 if ok else bad_sp + 1
        if ok and npv > best_npv:
            best_npv, since_best = npv, 0
        else:
            since_best += 1
        if forced:
            break
        if bad_sp >= 2:
            stop_reason = "wells no longer fit with the minimum spacing"
            break
        if since_best >= 5:
            stop_reason = "NPV has not improved for 5 more wells"
            break
        n += 1
    feas = [k for k in results if results[k]["spacing_ok"]] or list(results)
    best_max_n = max(feas, key=lambda k: results[k]["npv"])
    npv_max = results[best_max_n]["npv"]
    best_flat = min(k for k in feas if results[k]["npv"] >= npv_max - tol * abs(npv_max))
    best = min(int(forced), len(allw)) if forced else best_flat
    b = results[best]
    drained = rp.drained_mask(fm, [tuple(p) for p in b["w"]], radius_m, prm["leaky_faults"])
    uncovered = rp._upsample(fm.gd & ~drained, fm.D, mask.shape) & mask
    uncovered_km2 = float(uncovered.sum() * m_per_px ** 2 / 1e6)
    # ---- figures and tables ------------------------------------------------------------------
    f_wells = rp.plot_wells(rgb, mask, b["cells"], b["w"], min(len(seeds), best), m_per_px, b["stats"],
                            f"{prm['name']}: {best} wells, area {area_m2 / 1e6:.1f} km2", out / "wells_map.png",
                            extra={"radius_m": radius_m, "uncovered": uncovered, "platform": platform, "reach_m": reach})
    f_fc = rp.plot_forecast(results, best, g_used, rf, cgr, out / "forecast.png")
    f_hm = rp.plot_history(hist, hm["model"], out / "history_match.png") if hm else None
    fc_b = b["fc"]
    yr = (np.arange(len(fc_b["vol"])) // 12) + 1
    tab = pd.DataFrame({"year": yr, "gas_million_m3": fc_b["vol"] / 1e6, "condensate_m3": fc_b["vol"] * cgr,
                        "pressure_bar": fc_b["p"], "cum_gas_bcm": fc_b["gp"] / 1e9, "rate_million_m3_d": fc_b["q"] / 1e6})
    yearly = tab.groupby("year").agg(gas_million_m3=("gas_million_m3", "sum"), condensate_m3=("condensate_m3", "sum"),
                                     avg_rate_million_m3_d=("rate_million_m3_d", "mean"),
                                     end_pressure_bar=("pressure_bar", "last"), cum_gas_bcm=("cum_gas_bcm", "last")).round(3)
    sweep = pd.DataFrame([{"wells": k, "recovery_bcm": round(r["fc"]["gp"][-1] / 1e9, 2), "npv_musd": round(r["npv"], 1),
                           "spacing_ok": r["spacing_ok"],
                           "mark": "recommended" if k == best else ("highest NPV" if k == best_max_n else "")}
                          for k, r in results.items()])
    wells = pd.DataFrame(b["stats"])
    wells.to_csv(out / "wells.csv", index=False)
    yearly.to_csv(out / "forecast.csv")
    summary = {"name": prm["name"], "field_area_km2": round(area_m2 / 1e6, 2), "m_per_px": round(m_per_px, 3),
               "GIIP_bcm": round(giip / 1e9, 2), "GIIP_used_bcm": round(g_used / 1e9, 2),
               "recoverable_bcm": round(rf * g_used / 1e9, 2), "net_pay_m": round(prm["net_pay_m"], 1),
               "drainage_radius_m": round(radius_m), "radius_source": radius_src,
               "history_eur": eur, "recommended_wells": best, "max_npv_wells": best_max_n,
               "uncovered_field_km2": round(uncovered_km2, 2), "stop_reason": stop_reason,
               "rate_source": rate_source, "gwc_m": gwc_m, "contours": contour_info,
               "faults": "leaky" if prm["leaky_faults"] else "sealing", "fault_blocks": int(fm.block.max()),
               "parameters": ptab.as_list()}
    (out / "summary.json").write_text(__import__("json").dumps(summary, indent=2, default=float))
    return {"summary": summary, "wells": wells, "sweep": sweep, "yearly": yearly, "params": ptab, "warnings": warn,
            "figs": {"wells": f_wells, "forecast": f_fc, "history": f_hm}, "out_dir": out, "rgb": rgb,
            "mask": mask, "faults": faults_full, "radius_m": radius_m, "funnel": funnel}


def preview(map_path, m_per_px, field_color=None, fault_mode="black", fault_color=None, min_px=60, pdf_dpi=200.0):
    """Quick look for the web page: field mask and faults on the map (no calculation)."""
    rgb, used = rp.load_map_image(map_path, pdf_dpi, 1)
    fc = tuple(int(v) for v in field_color) if field_color else None
    mask, raw_px = rp.segment_field(rgb, fc, 60, False)
    faults = None
    if fault_mode != "none":
        faults = rp.extract_faults(rgb, min_px, [int(v) for v in fault_color] if fault_mode == "color" and fault_color else None, 80)
    return rgb, mask, faults, float(mask.sum()) * m_per_px ** 2 / 1e6


def run_contours(map_path, m_per_px, field_color=None, fault_mode="black", fault_color=None, interval=20.0,
                 redo=False, out_dir="out", pdf_dpi=200.0):
    """Detect the thin contour lines and read their depth numbers (for the Depth tab).
    Returns (depth points, summary dict or None, path of contours.png)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rgb, used = rp.load_map_image(map_path, pdf_dpi, 1)
    fc = tuple(int(v) for v in field_color) if field_color else None
    mask, _ = rp.segment_field(rgb, fc, 60, False)
    prm = dict(rp.DEFAULTS)
    faults = np.zeros(mask.shape, bool)
    if fault_mode != "none":
        faults = rp.extract_faults(rgb, int(prm["fault_min_pixels"]),
                                   [int(v) for v in fault_color] if fault_mode == "color" and fault_color else None, 80)
    ns = types.SimpleNamespace(contour_interval=interval, ocr_scale=3.0, ocr_angle_step=30, redo_ocr=bool(redo))
    pts, info = rp.auto_contour_pipeline(rgb, mask, faults, Path(map_path), prm, out, ns)
    return pts, info, out / "contours.png"
