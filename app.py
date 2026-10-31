"""
app.py - browser version of the reservoir planner (Streamlit).

Start it with:   python -m streamlit run app.py
(or double-click run_app.bat on Windows). It runs on YOUR computer; nothing is uploaded anywhere.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("RP_HEADLESS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

import planner_core as core

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:                      # the page still works: points are typed instead of clicked
    streamlit_image_coordinates = None

HERE = Path(__file__).resolve().parent
DATA = HERE / "app_data"
DATA.mkdir(exist_ok=True)
SHOW_W = 820                             # width of the clickable pictures on the page

st.set_page_config(page_title="Reservoir planner", layout="wide")
S = st.session_state
for k, v in {"scale_pts": [], "wells": [], "depth_pts": [], "auto_pts": [], "auto_info": None,
             "fault_color": None, "m_per_px": None, "result": None, "pending": None,
             "paths": {}, "example": False}.items():
    S.setdefault(k, v)


# ---------------------------------------------------------------- helpers

SCALE_FILE = DATA / "saved_scales.json"


def load_saved_scale(map_name):
    """The scale last used for a map with this file name (like the saved scale in the VS Code program)."""
    import json
    try:
        return json.loads(SCALE_FILE.read_text()).get(map_name)
    except Exception:
        return None


def save_scale(map_name, px, m):
    import json
    try:
        d = json.loads(SCALE_FILE.read_text()) if SCALE_FILE.exists() else {}
        d[map_name] = {"px": float(px), "m": float(m)}
        SCALE_FILE.write_text(json.dumps(d))
    except Exception:
        pass

def save_upload(f):
    if f is None:
        return None
    p = DATA / f.name
    p.write_bytes(f.getbuffer())
    return str(p)


@st.cache_data(show_spinner="Reading the map ...")
def load_map(path, mtime):
    import reservoir_planner as rp
    rgb, used = rp.load_map_image(path, 200.0, 1)
    return rgb


@st.cache_data(show_spinner="Measuring the field ...")
def field_preview(path, mtime, m_per_px, field_color, fault_mode, fault_color):
    rgb, mask, faults, km2 = core.preview(path, m_per_px, field_color, fault_mode, fault_color)
    return mask, faults, km2


def clickable(arr, key, points=(), labels=None, color=(255, 0, 0)):
    """Show a picture; return the new click as (x, y) in the ORIGINAL picture pixels, or None."""
    pil = Image.fromarray(arr)
    f = pil.width / SHOW_W
    disp = pil.resize((SHOW_W, int(pil.height / f)))
    d = ImageDraw.Draw(disp)
    for i, p in enumerate(points):
        x, y = p[0] / f, p[1] / f
        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=color, width=3)
        d.text((x + 9, y - 8), str(labels[i] if labels else i + 1), fill=color)
    if streamlit_image_coordinates is None:
        st.image(disp)
        st.info("The click component is not installed (python -m pip install streamlit-image-coordinates), "
                "so type the pixel position instead.")
        return None
    v = streamlit_image_coordinates(disp, key=key)
    if not v or S.get(key + "_last") == v:
        return None
    S[key + "_last"] = v
    return (v["x"] * pil.width / v["width"], v["y"] * pil.height / v["height"])


def overlay(arr, mask=None, faults=None):
    img = arr.astype(float).copy()
    if mask is not None:
        img[mask] = img[mask] * 0.55 + np.array([0, 200, 0]) * 0.45
    if faults is not None:
        img[faults] = (230, 0, 0)
    return img.astype(np.uint8)


# ---------------------------------------------------------------- sidebar: files
with st.sidebar:
    st.title("Reservoir planner")
    st.caption("Runs on your computer. Nothing leaves it.")
    ex = [HERE / n for n in ("L15A_ROSLU.pdf", "L15-02.xlsx", "Core_analyses_L15-02.xlsx",
                              "Short_description_of_field.docx")]
    if all(p.exists() for p in ex) and st.button("Load the L15 example"):
        S["paths"] = {"map": str(ex[0]), "history": str(ex[1]), "core": str(ex[2]), "handout": str(ex[3])}
        S["m_per_px"] = 1000.0 / 158.0
        S["wells"] = [(640.0, 613.0)]
        S["example"] = True
        S["scale_pts"] = []
        save_scale("L15A_ROSLU.pdf", 158.0, 1000.0)
        st.rerun()
    st.subheader("Files")
    up = {"map": st.file_uploader("Map picture (png / jpg / pdf)", type=["png", "jpg", "jpeg", "pdf"]),
          "history": st.file_uploader("Production history (xlsx / csv)", type=["xlsx", "xls", "csv"]),
          "core": st.file_uploader("Core analysis (xlsx / csv)", type=["xlsx", "xls", "csv"]),
          "handout": st.file_uploader("Field description (docx)", type=["docx"])}
    for k, f in up.items():
        if f is not None:
            S["paths"][k] = save_upload(f)
    for k, label in (("map", "map"), ("history", "history"), ("core", "core"), ("handout", "description")):
        if S["paths"].get(k):
            st.caption(f"{label}: {Path(S['paths'][k]).name}")
    if st.button("Clear everything"):
        for k in list(S.keys()):
            del S[k]
        st.rerun()

P = S["paths"]
st.title("Reservoir planner: well locations from a map and a production history")
if not P.get("map"):
    st.info("Start by uploading the map picture in the sidebar (or load the L15 example).")
    st.stop()

mpath = P["map"]
try:
    rgb = load_map(mpath, os.path.getmtime(mpath))
except (SystemExit, Exception) as e:          # e.g. a PDF map but pymupdf is not installed
    st.error(f"Could not read the map: {e}")
    st.stop()
tabs = st.tabs(["1. Map & scale", "2. Wells & faults", "3. Depth", "4. Settings", "5. Results"])

# ---------------------------------------------------------------- tab 1: map and scale
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    saved_scale = load_saved_scale(Path(mpath).name)
    with c2:
        st.subheader("Field colour")
        fmode = st.radio("How is the field drawn?", ["Any green (automatic)", "A custom colour"], key="fmode")
        field_color = None
        if fmode == "A custom colour":
            txt = st.text_input("R,G,B", "0,200,0", key="fcolor")
            try:
                field_color = tuple(int(v) for v in txt.split(","))
            except ValueError:
                st.error("Type three numbers like 0,200,0")
        S["field_color"] = field_color
        st.subheader("Scale of the map")
        st.caption(f"The picture used by the program is {rgb.shape[1]} x {rgb.shape[0]} pixels. "
                   "Pixel lengths must be measured on THIS picture.")
        opt_click = "Click the two ends of the scale bar on the map (recommended)"
        opt_px = "Type the scale-bar length in pixels (measured in Paint)"
        opt_mpp = "Type metres per pixel directly"
        options = [opt_click, opt_px, opt_mpp]
        if streamlit_image_coordinates is None:
            options.remove(opt_click)
            st.warning("Clicking needs:  python -m pip install streamlit-image-coordinates  (then restart the app).")
        if saved_scale:
            opt_saved = f"Reuse the saved scale: {saved_scale['px']:g} px = {saved_scale['m']:g} m"
            options.append(opt_saved)
        smode = st.radio("MAP SCALE - how do you want to give the scale of this map?", options,
                         index=len(options) - 1 if saved_scale else 0, key="smode")
        if saved_scale and smode == opt_saved:
            S["m_per_px"] = saved_scale["m"] / saved_scale["px"]
            st.success(f"{saved_scale['px']:g} px = {saved_scale['m']:g} m -> {S['m_per_px']:.3f} m per pixel")
        elif smode == opt_px:
            px = st.number_input("Length of the scale bar in PIXELS", value=float(saved_scale["px"]) if saved_scale else 158.0,
                                 min_value=0.1, format="%.1f", key="px_typed")
            metres = st.number_input("Length of that scale bar in METRES (5 km = 5000)",
                                     value=float(saved_scale["m"]) if saved_scale else 1000.0, min_value=0.1, key="m_typed")
            S["m_per_px"] = metres / px
            save_scale(Path(mpath).name, px, metres)
            st.success(f"{px:g} px = {metres:g} m -> {S['m_per_px']:.3f} m per pixel")
        elif smode == opt_mpp:
            S["m_per_px"] = st.number_input("Metres per pixel", value=float(S["m_per_px"] or 6.0), min_value=0.001,
                                            format="%.4f", key="mpp_typed")
            save_scale(Path(mpath).name, 1.0, S["m_per_px"])
        else:                                              # click the two ends
            st.write(f"Clicks so far: {len(S['scale_pts'])} (need 2: the LEFT end, then the RIGHT end).")
            if st.button("Clear the scale clicks"):
                S["scale_pts"] = []
                S["m_per_px"] = None
                st.rerun()
            if len(S["scale_pts"]) >= 2:
                (x0, y0), (x1, y1) = S["scale_pts"][:2]
                px_meas = float(np.hypot(x1 - x0, y1 - y0))
                px = st.number_input("Scale bar length in pixels (edit it if you measured it more precisely)",
                                     value=round(px_meas, 1), min_value=0.1, format="%.1f", key=f"px_{round(px_meas, 1)}")
                metres = st.number_input("That bar is how many METRES? (5 km = 5000)", value=1000.0, min_value=0.1,
                                         key="m_clicked")
                S["m_per_px"] = metres / px
                save_scale(Path(mpath).name, px, metres)
                st.success(f"{px:g} px = {metres:g} m -> {S['m_per_px']:.3f} m per pixel")
    with c1:
        if smode == opt_click and len(S["scale_pts"]) < 2:
            st.caption("Click the LEFT end and then the RIGHT end of the scale bar printed on the map.")
            c = clickable(rgb, "scale_click", S["scale_pts"], ["a", "b"])
            if c:
                S["scale_pts"].append(c)
                st.rerun()
        elif S["m_per_px"]:
            mask, faults, km2 = field_preview(mpath, os.path.getmtime(mpath), S["m_per_px"], field_color, "none", None)
            st.image(overlay(rgb, mask), caption=f"Mapped field: {km2:.2f} km2 (green)", width=SHOW_W)
            if km2 < 1 or km2 > 2000:
                st.error("That field size looks wrong: check the scale (the bar length must be in METRES, the "
                         "pixel length measured on this picture) and the field colour.")
        else:
            st.image(rgb, width=SHOW_W)

if not S["m_per_px"]:
    for t in tabs[1:]:
        with t:
            st.info("Set the map scale in tab 1 first.")
    st.stop()
M = float(S["m_per_px"])

# ---------------------------------------------------------------- tab 2: wells and faults
with tabs[1]:
    c1, c2 = st.columns([3, 2])
    with c2:
        st.subheader("Existing wells")
        st.caption("Click each well on the picture. The first one is the platform unless you pick another.")
        st.write(f"{len(S['wells'])} well(s).")
        b1, b2 = st.columns(2)
        if b1.button("Remove last well") and S["wells"]:
            S["wells"].pop()
        if b2.button("Clear wells"):
            S["wells"] = []
        platform_idx = 0
        if len(S["wells"]) > 1:
            platform_idx = st.selectbox("Platform", range(len(S["wells"])), format_func=lambda i: f"well {i + 1}")
        reach = st.number_input("Maximum reach from the platform, metres (0 = no limit)", value=3750.0 if S["example"] else 0.0,
                                min_value=0.0, step=250.0)
        st.subheader("Faults")
        faultmode = st.radio("How are faults drawn?", ["Thick black lines", "Click on a fault line (copy its colour)",
                                                       "Ignore faults"], key="faultmode")
        fault_mode = {"Thick black lines": "black", "Ignore faults": "none"}.get(faultmode, "color")
        leaky = st.radio("Do faults let pressure through?", ["Leaky (default)", "Sealing"], key="leaky") == "Leaky (default)"
        if fault_mode == "color":
            st.write("Fault colour:", S["fault_color"] or "not picked yet - click a fault on the picture")
            if st.button("Pick again"):
                S["fault_color"] = None
    with c1:
        pick_fault = fault_mode == "color" and not S["fault_color"]
        if pick_fault:
            st.caption("Click ON a fault line (zoom the browser in if the line is thin).")
            c = clickable(rgb, "fault_click")
            if c:
                x, y = int(c[0]), int(c[1])
                patch = rgb[max(0, y - 4):y + 5, max(0, x - 4):x + 5].reshape(-1, 3).astype(int)
                med = np.median(patch, axis=0)
                col = patch[np.abs(patch - med).sum(axis=1).argmax()]
                S["fault_color"] = tuple(int(v) for v in col)
                st.rerun()
        else:
            st.caption("Click on the picture to add an existing well.")
            c = clickable(rgb, "well_click", S["wells"])
            if c:
                S["wells"].append(c)
                st.rerun()
            fc = S["fault_color"] if fault_mode == "color" else None
            if fault_mode != "none":
                mask, faults, km2 = field_preview(mpath, os.path.getmtime(mpath), M, S["field_color"], fault_mode, fc)
                cov = 100.0 * float((faults & mask).sum()) / max(float(mask.sum()), 1.0)
                st.image(overlay(rgb, None, faults), caption=f"Detected faults in red: {cov:.1f}% of the field",
                         width=SHOW_W)

# ---------------------------------------------------------------- tab 3: depth
with tabs[2]:
    st.subheader("Depth of the top of the reservoir")
    dmode = st.radio("Where does the depth come from?", ["Automatic: read the contour lines and their numbers",
                                                         "Click depth labels by hand", "No depth information"],
                     key="dmode")
    gwc = st.number_input("Depth of the gas-water contact, metres (0 = estimate it)", value=2967.0 if S["example"] else 0.0,
                          min_value=0.0)
    if dmode.startswith("Automatic"):
        interval = st.number_input("Contour interval, metres", value=20.0, min_value=1.0)
        redo = st.checkbox("Read the numbers again (ignore the stored readings)")
        if st.button("Detect the contours and read the numbers"):
            with st.spinner("Reading the depth numbers - about a minute the first time ..."):
                pts, info, img_path = core.run_contours(mpath, M, S["field_color"], "none" if fault_mode == "none" else fault_mode,
                                                        S["fault_color"], interval, redo, DATA / "contours_out")
            S["auto_pts"], S["auto_info"] = pts, info
            S["contour_img"] = str(img_path)
        if S["auto_info"]:
            i = S["auto_info"]
            st.success(f"{i['numbers_read']} numbers read; {i['lines_with_depth']} of {i['contour_lines']} contour lines "
                       f"got a depth ({i.get('depth_range_m', ['?', '?'])[0]:.0f} - {i.get('depth_range_m', ['?', '?'])[1]:.0f} m).")
            if S.get("contour_img") and Path(S["contour_img"]).exists():
                st.image(S["contour_img"], caption="red = bold lines (faults, outline, text); coloured = contour lines "
                                                   "with the depth that was read; grey = lines without a number",
                         )
        elif S["auto_pts"] == [] and st.session_state.get("tried_auto"):
            st.warning("No OCR package found: python -m pip install rapidocr-onnxruntime")
    elif dmode.startswith("Click"):
        c1, c2 = st.columns([3, 2])
        with c1:
            c = clickable(rgb, "depth_click", [(p[0], p[1]) for p in S["depth_pts"]], [int(p[2]) for p in S["depth_pts"]], (0, 0, 255))
            if c:
                S["pending"] = c
        with c2:
            if S["pending"]:
                st.write(f"Selected position: ({S['pending'][0]:.0f}, {S['pending'][1]:.0f})")
                dep = st.number_input("Depth printed at that position, metres", value=2800.0)
                if st.button("Add this depth label"):
                    S["depth_pts"].append((S["pending"][0], S["pending"][1], dep))
                    S["pending"] = None
                    st.rerun()
            st.write(f"{len(S['depth_pts'])} label(s). 12 to 25 labels spread over the field work well.")
            if st.button("Clear depth labels"):
                S["depth_pts"] = []

# ---------------------------------------------------------------- tab 4: settings
with tabs[3]:
    st.caption("Leave a box at 0 to let the program calculate it from the files. A value you type always wins.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Reservoir")
        reservoir = None
        if P.get("handout"):
            import reservoir_planner as rp
            h = rp.read_field_description(P["handout"])
            names = list((h or {}).get("reservoirs", {}))
            if names:
                reservoir = st.selectbox("Row of the field-description table", names, index=names.index("L15b-A") if "L15b-A" in names else 0)
        prefer = st.checkbox("Use the description's values before the core calculation")
        giip = st.number_input("Gas in place, billion m3 (0 = from volumetric)", value=9.07 if S["example"] else 0.0, min_value=0.0)
        reserves = st.number_input("Reserves, billion m3 (with gas in place gives the recovery factor)", value=6.742 if S["example"] else 0.0, min_value=0.0)
        netpay = st.number_input("Net pay, metres (0 = calculate)", value=0.0, min_value=0.0)
        poro = st.number_input("Porosity, % (0 = calculate)", value=0.0, min_value=0.0, max_value=40.0)
        sg = st.number_input("Gas saturation, % (0 = calculate)", value=0.0, min_value=0.0, max_value=100.0)
        pres = st.number_input("Reservoir pressure, bar (0 = calculate)", value=0.0, min_value=0.0)
        opres = st.number_input("Operating (flowing) pressure, bar (0 = 25% of reservoir pressure)", value=0.0, min_value=0.0)
    with c2:
        st.subheader("Economics")
        well_cost = st.number_input("Cost of one well, million USD", value=30.0, min_value=0.0)
        opex = st.number_input("Operating cost, million USD per year", value=4.0, min_value=0.0)
        gprice = st.number_input("Gas price, USD per 1000 m3 (0 = from the description)", value=0.0, min_value=0.0)
        cprice = st.number_input("Condensate price, USD per m3", value=450.0, min_value=0.0)
        disc = st.number_input("Discount rate", value=0.10, min_value=0.0, max_value=1.0)
        life = st.number_input("Field life, years", value=25, min_value=1)
        cap = st.number_input("Plant / export capacity, million m3 per day (0 = no limit)", value=0.0, min_value=0.0)
        tol = st.number_input("NPV tolerance (fewest wells within this fraction of the best NPV)", value=0.02, min_value=0.0, max_value=0.5)
    with c3:
        st.subheader("Well placement")
        radius = st.number_input("Drainage radius, metres (0 = from the production history)", value=0.0, min_value=0.0)
        spacing = st.number_input("Minimum spacing between wells, m", value=500.0, min_value=0.0)
        edge = st.number_input("Distance from the field edge, m", value=150.0, min_value=0.0)
        fstand = st.number_input("Distance from faults, m", value=120.0, min_value=0.0)
        mincol = st.number_input("Minimum gas column, m", value=80.0, min_value=0.0)
        mingain = st.number_input("Stop when a new well adds less than, km2", value=0.5, min_value=0.0)
        hist_wells = st.number_input("Wells that produced the history", value=1, min_value=1)
        match = st.checkbox("Scale the gas in place to match the history (only if the history is the whole field)", value=False)
        nwells = st.number_input("Force the number of wells (0 = automatic)", value=0, min_value=0)

# ---------------------------------------------------------------- tab 5: results
with tabs[4]:
    if st.button("Run the planning", type="primary"):
        settings = {"giip_bcm": giip or None, "reserves_bcm": reserves or None, "net_pay_m": netpay or None,
                    "porosity": poro / 100 if poro else None, "gas_saturation": sg / 100 if sg else None,
                    "reservoir_pressure_bar": pres or None, "operating_pressure_bar": opres or None,
                    "well_cost_musd": well_cost, "opex_musd_per_year": opex, "gas_price_usd_per_1000m3": gprice or None,
                    "condensate_price_usd_per_m3": cprice, "discount_rate": disc, "field_life_years": int(life),
                    "min_well_spacing_m": spacing, "edge_buffer_m": edge, "fault_standoff_m": fstand,
                    "min_gas_column_m": mincol, "min_gain_km2": mingain, "drainage_radius_m": None}
        auto = dmode.startswith("Automatic")
        cfg = dict(map_path=mpath, m_per_px=M, field_color=S["field_color"], seeds=S["wells"],
                   platform=S["wells"][platform_idx] if S["wells"] else None, reach_m=reach or None,
                   fault_mode=fault_mode, fault_color=S["fault_color"], leaky=leaky,
                   depth_mode="points" if (auto and S["auto_pts"]) or dmode.startswith("Click") else "none",
                   depth_pts=S["auto_pts"] if auto else (S["depth_pts"] if dmode.startswith("Click") else []),
                   gwc_m=gwc or None, history_path=P.get("history"), history_wells=int(hist_wells),
                   core_path=P.get("core"), handout_path=P.get("handout"), reservoir=reservoir, prefer_handout=prefer,
                   settings=settings, match_history=match, radius_m=radius or None, n_wells=int(nwells) or None,
                   rate_cap_mm3d=cap or None, npv_tolerance=tol, out_dir=str(DATA / "results"))
        try:
            with st.spinner("Calculating ..."):
                S["result"] = core.analyse(cfg)
        except Exception as e:                                    # show the reason, not a stack trace
            S["result"] = None
            st.error(str(e))
    r = S.get("result")
    if r:
        sm = r["summary"]
        for w in r["warnings"]:
            st.warning(w)
        m = st.columns(6)
        m[0].metric("Mapped field", f"{sm['field_area_km2']} km2")
        m[1].metric("Gas in place", f"{sm['GIIP_bcm']} bcm")
        m[2].metric("Net pay", f"{sm['net_pay_m']} m")
        m[3].metric("Drainage radius", f"{sm['drainage_radius_m']} m")
        m[4].metric("Recommended wells", sm["recommended_wells"])
        m[5].metric("Field not reached", f"{sm['uncovered_field_km2']} km2")
        st.caption(f"Radius from: {sm['radius_source']}")
        st.subheader("Proposed well locations")
        st.pyplot(r["figs"]["wells"])
        st.dataframe(r["wells"])
        st.subheader("Number of wells, recovery and NPV")
        st.dataframe(r["sweep"])
        st.pyplot(r["figs"]["forecast"])
        if r["figs"]["history"] is not None:
            st.subheader("Model against the real production history")
            st.pyplot(r["figs"]["history"])
        with st.expander("Parameters and where each value came from"):
            st.dataframe(pd.DataFrame([{"parameter": x["parameter"], "value": x["value"], "source": x["source"],
                                        "how": x["how"], "other sources": str(x["other_sources"])} for x in r["params"].as_list()]),
                         )
        with st.expander("Yearly production forecast"):
            st.dataframe(r["yearly"])
        d1, d2, d3 = st.columns(3)
        out = r["out_dir"]
        d1.download_button("wells.csv", (out / "wells.csv").read_bytes(), "wells.csv")
        d2.download_button("wells_map.png", (out / "wells_map.png").read_bytes(), "wells_map.png")
        d3.download_button("summary.json", (out / "summary.json").read_bytes(), "summary.json")
