# Reservoir Development Planner — Streamlit integration

## Run locally (Windows PowerShell)

From this folder, run the virtual-environment Python directly (no activation needed):

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m streamlit run app.py
```

Keep `app.py` and `reservoir_planner.py` in the same folder. The app runs the original engineering engine as a subprocess, saves uploaded files under `data/`, and writes generated outputs under `outputs/`.

## Important

This is an initial integration, not a validated engineering product. The original calculations are retained, but the desktop map-clicking workflow is not yet available in the browser; map scale and any existing-well coordinates must be entered manually in pixel units. Confirm field segmentation and map scale, review the generated map and logs, and validate all assumptions/results independently. Economics use placeholder defaults unless replaced with supported project data.
