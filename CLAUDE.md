# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file Streamlit app (`streamlit_app.py`) that counts insects in trap photos using a local YOLOv8 model. The UI is entirely in Thai and is targeted at sugar-factory pest inspection. Users pick a factory/department/location, upload or capture a photo, get an annotated image with counts, then save the record (Excel + a Power Automate webhook).

## Commands

```bash
pip install -r requirements.txt        # Python deps
streamlit run streamlit_app.py          # run locally → http://localhost:8501
```

There is no build, test, or lint setup. On Debian/Streamlit Cloud, `packages.txt` lists the apt packages (`libgl1`, `libglib2.0-0t64`) that OpenCV needs — these are required for `import cv2` to succeed in that environment.

## Architecture & key facts

- **Everything lives in `streamlit_app.py`.** It's one ~700-line script: CSS block, then a two-column layout (left = inputs, right = results). State flows through `st.session_state`; the three-step indicator (`render_steps`) is derived from whether location fields and `raw_predictions` are set.
- **Model:** loaded via `@st.cache_resource` from the checked-in weights file `runs_detect_train-4_weights_best.pt` (52 MB, tracked in git). Two classes: `fly` and `test` (see `Dataset/data.yaml`). `test` is surfaced in the UI as "แมลงอื่นๆ" (other insects).
- **Two-stage confidence:** `model.predict(conf=0.10, iou=0.40, imgsz=1920, max_det=5000)` runs once at a low threshold and stores all raw boxes. The UI slider then *filters* those stored predictions client-side — changing the slider does NOT re-run inference. Keep that split intact when editing detection logic.
- **`LOCATION_DATA`** is a hardcoded factory → department → location dict (Thai strings) driving the cascading selectboxes. Edit this dict to change available locations.
- **Saving** does three things: POSTs the record (with base64 JPEG) to a hardcoded Power Automate URL, appends to the local `insect_analysis_history.xlsx`, and offers an in-memory Excel download. The committed xlsx is real historical data, not a fixture.

## Gotchas

- `roboflow` is imported at the top but unused — the app moved from the Roboflow API to local YOLO (it's also absent from `requirements.txt`). Don't reintroduce a Roboflow dependency.
- `index.html` is an **unrelated** standalone water/wastewater dashboard, not part of the Streamlit app. Ignore it unless explicitly asked.
- Deployment is Streamlit Community Cloud (see `CODEOWNERS`). Recent breakages have been environment-specific: OpenCV needing the right apt package names for the current Debian release. If `cv2` fails to import in the cloud, check `packages.txt` against the deployed Debian version.
- All user-facing strings are Thai; timestamps use the Asia/Bangkok offset (`timezone(timedelta(hours=7))`). Preserve both when touching UI or records.
