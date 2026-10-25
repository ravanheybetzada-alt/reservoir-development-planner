from pathlib import Path
import json
import subprocess
import sys
import io

import pandas as pd
import streamlit as st
from PIL import Image

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "outputs"
ENGINE = BASE / "reservoir_planner.py"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

st.set_page_config(page_title="Reservoir Development Planner", page_icon="🛢️", layout="wide")
st.title("Reservoir Development Planner")
st.caption("Gas / gas-condensate field screening · local engineering prototype")
st.warning("Engineering screening only. Check source data, map calibration, assumptions and outputs before using results for any real development decision.")

with st.sidebar:
    st.header("Project")
    project_name = st.text_input("Project name", value="My reservoir study")
    st.subheader("Reservoir parameters")
    depth_m = st.number_input("Reservoir depth (m)", min_value=0.0, value=2950.0, step=50.0)
    pressure_bar = st.number_input("Initial reservoir pressure (bar)", min_value=0.0, value=305.0, step=5.0)
    temperature_c = st.number_input("Reservoir temperature (°C)", value=100.0, step=5.0)
    gas_sg = st.number_input("Gas specific gravity (air = 1)", min_value=0.1, value=0.78, step=0.01, format="%.2f")
    porosity_pct = st.number_input("Porosity (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
    gas_sat_pct = st.number_input("Gas saturation (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    permeability_md = st.number_input("Permeability (mD)", min_value=0.0, value=10.0, step=1.0)
    net_pay_m = st.number_input("Net pay (m)", min_value=0.0, value=30.0, step=1.0)
    operating_pressure_bar = st.number_input("Operating / flowing pressure (bar)", min_value=0.0, value=76.25, step=5.0)
    st.caption("These values are starting assumptions and are written into the engine settings. Change them to match supported project data.")

st.subheader("1. Upload the project files")
a, b = st.columns(2)
with a:
    map_file = st.file_uploader("Reservoir map", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "pdf"])
    core_file = st.file_uploader("Core analysis table (optional)", type=["csv", "xlsx", "xls"])
with b:
    history_file = st.file_uploader("Production history (optional)", type=["csv", "xlsx", "xls"])
    handout_file = st.file_uploader("Field description (optional; DOCX)", type=["docx"])

st.subheader("2. Map calibration and well constraints")
st.caption("The original engine uses pixel coordinates. Measure the scale bar on the uploaded map and enter its pixel length and real-world length, or enter metres per pixel directly.")
c1, c2, c3 = st.columns(3)
with c1:
    scale_mode = st.radio("Scale input", ["Scale-bar length", "Metres per pixel"], horizontal=False)
with c2:
    if scale_mode == "Scale-bar length":
        scalebar_px = st.number_input("Scale-bar length (pixels)", min_value=0.01, value=64.0, step=1.0)
    else:
        m_per_px = st.number_input("Metres per pixel", min_value=0.0001, value=15.625, step=0.1, format="%.4f")
with c3:
    if scale_mode == "Scale-bar length":
        scalebar_m = st.number_input("Scale-bar real length (metres)", min_value=0.01, value=1000.0, step=100.0)
    else:
        st.info("Direct scale selected")

c4, c5, c6 = st.columns(3)
with c4:
    field_color = st.text_input("Field polygon RGB (optional)", placeholder="e.g. 0,180,70")
    color_tol = st.number_input("Field colour tolerance", min_value=0, max_value=255, value=60)
with c5:
    seed_wells = st.text_input("Existing wells (pixel x,y; separate with ;)", placeholder="e.g. 256,167")
    platform = st.text_input("Platform position (optional, pixel x,y)", placeholder="x,y")
with c6:
    reach = st.number_input("Platform reach (m; 0 = no limit)", min_value=0.0, value=0.0, step=250.0)

with st.expander("Advanced model / economic assumptions"):
    d1, d2, d3 = st.columns(3)
    with d1:
        recovery_factor = st.slider("Recovery factor", 0.1, 1.0, 0.75, 0.05)
        min_spacing = st.number_input("Minimum well spacing (m)", min_value=1.0, value=500.0, step=50.0)
        edge_buffer = st.number_input("Field-edge buffer (m)", min_value=0.0, value=150.0, step=25.0)
    with d2:
        gas_price = st.number_input("Gas price (USD / 1000 m³)", min_value=0.0, value=200.0, step=10.0)
        condensate_price = st.number_input("Condensate price (USD / m³)", min_value=0.0, value=450.0, step=25.0)
        cgr = st.number_input("Condensate-gas ratio (m³ / million m³)", min_value=0.0, value=10.0, step=1.0)
    with d3:
        well_cost = st.number_input("Well cost (million USD)", min_value=0.0, value=30.0, step=1.0)
        fixed_capex = st.number_input("Fixed CAPEX (million USD)", min_value=0.0, value=0.0, step=1.0)
        opex = st.number_input("Annual OPEX (million USD/year)", min_value=0.0, value=4.0, step=0.5)
    discount_rate = st.number_input("Discount rate", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
    field_life = st.number_input("Field life (years)", min_value=1, max_value=100, value=25, step=1)
    npv_tolerance = st.number_input("NPV tolerance (fraction; 0.02 = 2%)", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
    match_history = st.checkbox("Match GIIP to production history when history is supplied", value=True)
    sealing = st.checkbox("Treat faults as sealing (otherwise leaky)", value=False)
    auto_contours = st.checkbox("Attempt automatic contour-line / depth-label reading (optional OCR dependency)", value=False)
    gwc_depth = st.number_input("Gas-water contact depth (m; 0 = not supplied)", min_value=0.0, value=0.0, step=10.0)
    rate_cap = st.number_input("Facility rate cap (million m³/day; 0 = no limit)", min_value=0.0, value=0.0, step=0.5)

st.subheader("3. Preview files")
if map_file:
    try:
        if map_file.name.lower().endswith('.pdf'):
            st.info(f"PDF map selected: {map_file.name}. The engine will render the first page.")
        else:
            im = Image.open(map_file)
            st.image(im, caption=f"{map_file.name} · {im.width} × {im.height} px", use_container_width=True)
    except Exception as exc:
        st.error(f"Could not preview map: {exc}")

def preview_table(uploaded, label):
    if uploaded is None:
        return
    try:
        suffix = Path(uploaded.name).suffix.lower()
        df = pd.read_csv(uploaded) if suffix == '.csv' else pd.read_excel(uploaded)
        st.markdown(f"**{label}:** {len(df):,} rows × {len(df.columns):,} columns")
        st.dataframe(df.head(100), use_container_width=True)
    except Exception as exc:
        st.error(f"Could not read {label}: {exc}")
preview_table(history_file, "Production history")
preview_table(core_file, "Core analysis")

st.divider()
st.subheader("4. Run the original engineering engine")
st.write("This calls the original `reservoir_planner.py` as a separate process with the uploaded files and settings. It keeps the original calculation logic; terminal prompts and interactive map clicks are disabled for this first browser integration.")
if not ENGINE.exists():
    st.error("The engine file reservoir_planner.py is missing. Put it in the same folder as app.py.")
else:
    st.success("Original engine file found: reservoir_planner.py")

if st.button("Run reservoir analysis", type="primary", disabled=map_file is None):
    try:
        # Save uploads using stable local paths. Original filenames are retained in display only.
        uploaded_paths = {}
        for key, uploaded in [("map", map_file), ("history", history_file), ("core", core_file), ("handout", handout_file)]:
            if uploaded is not None:
                safe_name = {"map":"uploaded_map", "history":"production_history", "core":"core_analysis", "handout":"field_description"}[key] + Path(uploaded.name).suffix.lower()
                p = DATA / safe_name
                p.write_bytes(uploaded.getvalue())
                uploaded_paths[key] = p
        settings = {
            "name": project_name,
            "depth_m": depth_m,
            "reservoir_pressure_bar": pressure_bar,
            "operating_pressure_bar": operating_pressure_bar,
            "reservoir_temp_c": temperature_c,
            "gas_specific_gravity": gas_sg,
            "porosity": porosity_pct / 100.0,
            "gas_saturation": gas_sat_pct / 100.0,
            "permeability_md": permeability_md,
            "net_pay_m": net_pay_m,
            "recovery_factor": recovery_factor,
            "min_well_spacing_m": min_spacing,
            "edge_buffer_m": edge_buffer,
            "gas_price_usd_per_1000m3": gas_price,
            "condensate_price_usd_per_m3": condensate_price,
            "cgr_m3_per_million_m3": cgr,
            "well_cost_musd": well_cost,
            "fixed_capex_musd": fixed_capex,
            "opex_musd_per_year": opex,
            "discount_rate": discount_rate,
            "field_life_years": int(field_life),
            "npv_tolerance": npv_tolerance,
            "match_history": match_history,
            "use_faults": True,
            "leaky_faults": not sealing,
        }
        params_path = DATA / "web_settings.json"
        params_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        cmd = [sys.executable, str(ENGINE), "--map", str(uploaded_paths["map"]), "--params", str(params_path), "--no-ask", "--no-show", "--outdir", str(OUT), "--color-tol", str(int(color_tol))]
        if "history" in uploaded_paths: cmd += ["--history", str(uploaded_paths["history"])]
        else: cmd += ["--history", "none"]
        if "core" in uploaded_paths: cmd += ["--core", str(uploaded_paths["core"])]
        else: cmd += ["--core", "none"]
        if "handout" in uploaded_paths: cmd += ["--field-description", str(uploaded_paths["handout"])]
        else: cmd += ["--field-description", "none"]
        if scale_mode == "Scale-bar length": cmd += ["--scalebar-px", str(scalebar_px), "--scalebar-m", str(scalebar_m)]
        else: cmd += ["--m-per-px", str(m_per_px)]
        if field_color.strip(): cmd += ["--field-color", field_color.strip()]
        if seed_wells.strip(): cmd += ["--seed-wells", seed_wells.strip()]
        if platform.strip(): cmd += ["--platform", platform.strip()]
        if reach > 0: cmd += ["--reach", str(reach)]
        if gwc_depth > 0: cmd += ["--gwc-depth", str(gwc_depth)]
        if rate_cap > 0: cmd += ["--rate-cap-mm3d", str(rate_cap)]
        if auto_contours: cmd += ["--auto-contours"]
        if sealing: cmd += ["--sealing"]
        with st.spinner("Running the reservoir engine. This can take a little while for large maps…"):
            result = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True, timeout=600)
        st.session_state["engine_stdout"] = result.stdout
        st.session_state["engine_stderr"] = result.stderr
        st.session_state["engine_returncode"] = result.returncode
        if result.returncode == 0 and (OUT / "summary.json").exists():
            st.session_state["engine_summary"] = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
            st.success("Analysis completed. Review the assumptions and outputs below.")
        else:
            st.error("The engine did not complete successfully. See the diagnostic output below.")
    except subprocess.TimeoutExpired:
        st.error("The analysis exceeded 10 minutes and was stopped. Try a smaller map or lower-resolution PDF.")
    except Exception as exc:
        st.exception(exc)

if "engine_summary" in st.session_state:
    s = st.session_state["engine_summary"]
    st.subheader("5. Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Field area", f"{s.get('field_area_km2', '—')} km²")
    m2.metric("Recommended wells", str(s.get("recommended_wells", "—")))
    m3.metric("GIIP used", f"{s.get('GIIP_used_bcm', '—')} bcm")
    m4, m5, m6 = st.columns(3)
    m4.metric("Recoverable gas", f"{s.get('recoverable_bcm', '—')} bcm")
    m5.metric("Initial rate / well", f"{s.get('initial_rate_per_well_million_m3_d', '—')} million m³/day")
    m6.metric("Map scale", f"{s.get('metres_per_pixel', '—')} m/pixel")
    if s.get("wells"):
        st.markdown("**Proposed well locations (pixel coordinates)**")
        st.dataframe(pd.DataFrame(s["wells"]), use_container_width=True)
    if (OUT / "wells_map.png").exists():
        st.markdown("**Well placement map**")
        st.image(str(OUT / "wells_map.png"), use_container_width=True)
    if (OUT / "forecast.png").exists():
        st.markdown("**Production and NPV forecast**")
        st.image(str(OUT / "forecast.png"), use_container_width=True)
    if (OUT / "history_match.png").exists():
        st.markdown("**History match**")
        st.image(str(OUT / "history_match.png"), use_container_width=True)
    for filename, label, mime in [("summary.json", "Download summary JSON", "application/json"), ("forecast.csv", "Download forecast CSV", "text/csv"), ("wells_map.png", "Download well map", "image/png"), ("forecast.png", "Download forecast plot", "image/png")]:
        p = OUT / filename
        if p.exists(): st.download_button(label, data=p.read_bytes(), file_name=filename, mime=mime, key=f"dl_{filename}")

with st.expander("Engine log / diagnostics"):
    if "engine_stdout" in st.session_state:
        st.text_area("Standard output", st.session_state["engine_stdout"], height=260)
        if st.session_state.get("engine_stderr"):
            st.text_area("Errors / warnings", st.session_state["engine_stderr"], height=180)
    else:
        st.caption("Run an analysis to see the original engine's log here.")

st.caption("Important limitations: the engine's original click-to-calibrate workflow has not yet been replaced by clickable browser map controls. This version requires scale inputs and pixel coordinates to be entered manually. Review field segmentation, fault detection, and well locations visually before relying on the results.")
