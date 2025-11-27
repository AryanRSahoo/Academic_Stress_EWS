# src/streamlit/app.py
import os
import json
from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------
# CONFIG: default project root (change by setting PROJECT_ROOT env var)
# -----------------------
DEFAULT_PROJECT_ROOT = "/Users/aryansahoo/Documents/Academic_Stress_EWS_Project"
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT)

MODELS_DIR = Path(PROJECT_ROOT) / "models"
MODEL_PATH = MODELS_DIR / "logistic_pipeline.joblib"
FEATURES_JSON = MODELS_DIR / "feature_names.json"

# -----------------------
# load model + feature order
# -----------------------
@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: Path, features_path: Path):
    model = joblib.load(model_path)
    if features_path.exists():
        with open(features_path, "r") as fh:
            feature_order = json.load(fh)
    else:
        # fallback: raise helpful error
        raise FileNotFoundError(f"feature_names.json not found at {features_path}")
    return model, feature_order

# UI starts
st.set_page_config(page_title="Academic Stress EWS", layout="wide")

st.markdown(""" 
# Academic Stress EWS  
Professional — Clean — Predictive  
_Please fill all fields and click Predict_
""")

# Sidebar: show project location & load button
with st.sidebar:
    st.write("**Project root**")
    st.code(PROJECT_ROOT)
    st.write("---")
    st.write("Model file:")
    st.code(str(MODEL_PATH))
    st.write("Feature order file:")
    st.code(str(FEATURES_JSON))
    st.write("---")
    if st.button("Reload model"):
        st.rerun()

# load artifacts (fail with clear message if missing)
try:
    MODEL, FEATURE_ORDER = load_artifacts(MODEL_PATH, FEATURES_JSON)
except Exception as e:
    st.error(f"Failed to load model/features: {e}")
    st.stop()

# Ensure FEATURE_ORDER contains all required features (sanity)
REQUIRED = [
    "school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob",
    "reason","guardian","traveltime","studytime","failures","schoolsup","famsup",
    "paid","activities","nursery","higher","internet","romantic","famrel","freetime",
    "goout","Dalc","Walc","health","absences","G1","G2"
]
if not all(f in FEATURE_ORDER for f in REQUIRED):
    st.warning("Loaded feature order does not exactly match expected required features. App will still attempt to proceed using feature_names.json order.")

# -----------------------
# Friendly labels and option maps (must match what pipeline expects)
# -----------------------
# For readability in the UI we display user-friendly strings; the dict values
# are the encoded values that get fed to the pipeline.

FIELD_INFO = {
    "school": ("School", {"GP": "GP", "MS": "MS"}),
    "sex": ("Sex", {"F": "F", "M": "M"}),
    "age": ("Age (years)", None),  # numeric
    "address": ("Address", {"U (urban)": "U", "R (rural)": "R"}),
    "famsize": ("Family Size", {"GT3 (>=3)": "GT3", "LE3 (<3)": "LE3"}),
    "Pstatus": ("Parent status", {"T (together)": "T", "A (apart)": "A"}),
    "Medu": ("Mother's Education", {"0 - None": 0, "1 - Primary": 1, "2 - 5th–9th": 2, "3 - Secondary": 3, "4 - Higher": 4}),
    "Fedu": ("Father's Education", {"0 - None": 0, "1 - Primary": 1, "2 - 5th–9th": 2, "3 - Secondary": 3, "4 - Higher": 4}),
    "Mjob": ("Mother's Job", {"teacher":"teacher","health":"health","services":"services","at_home":"at_home","other":"other"}),
    "Fjob": ("Father's Job", {"teacher":"teacher","health":"health","services":"services","at_home":"at_home","other":"other"}),
    "reason": ("Reason for school", {"home":"home","reputation":"reputation","course":"course","other":"other"}),
    "guardian": ("Guardian", {"mother":"mother","father":"father","other":"other"}),
    "traveltime": ("Travel time", {"1 - <15 min":1,"2 - 15–30 min":2,"3 - 30–60 min":3,"4 - >60 min":4}),
    "studytime": ("Study time", {"1 - <2 hrs":1,"2 - 2–5 hrs":2,"3 - 5–10 hrs":3,"4 - >10 hrs":4}),
    "failures": ("Past failures", {"0":0,"1":1,"2":2,"3":3}),
    "schoolsup": ("School support", {"yes":1,"no":0}),
    "famsup": ("Family support", {"yes":1,"no":0}),
    "paid": ("Extra paid classes", {"yes":1,"no":0}),
    "activities": ("Activities", {"yes":1,"no":0}),
    "nursery": ("Attended nursery", {"yes":1,"no":0}),
    "higher": ("Wants higher edu", {"yes":1,"no":0}),
    "internet": ("Internet at home", {"yes":1,"no":0}),
    "romantic": ("In romantic relationship", {"yes":1,"no":0}),
    "famrel": ("Family relation (1=bad → 5=excellent)", {"1":1,"2":2,"3":3,"4":4,"5":5}),
    "freetime": ("Free time (1→5)", {"1":1,"2":2,"3":3,"4":4,"5":5}),
    "goout": ("Go out with friends (1→5)", {"1":1,"2":2,"3":3,"4":4,"5":5}),
    "Dalc": ("Workday alcohol (1→5)", {"1":1,"2":2,"3":3,"4":4,"5":5}),
    "Walc": ("Weekend alcohol (1→5)", {"1":1,"2":2,"3":3,"4":4,"5":5}),
    "health": ("Health (1→5)", {"1":1,"2":2,"3":3,"4":4,"5":5}),
    "absences": ("Absence count (0–93)", None),
    "G1": ("Grade G1 (0–20)", None),
    "G2": ("Grade G2 (0–20)", None)
}

# -----------------------
# Build form (two column layout)
# -----------------------
st.markdown("## Input features")

with st.form(key="input_form", clear_on_submit=False):
    cols = st.columns((1,1))  # two equal columns
    inputs: Dict[str, Any] = {}
    col_index = 0
    for feat in FEATURE_ORDER:
        # friendly label & map
        if feat not in FIELD_INFO:
            st.warning(f"Missing UI mapping for feature: {feat}")
            label = feat
            opts = None
        else:
            label, opts = FIELD_INFO[feat]

        c = cols[col_index % 2]
        col_index += 1

        if opts is None:
            # numeric input
            if feat in ("age",):
                val = c.number_input(label, min_value=10, max_value=30, value=17, step=1, key=feat)
            elif feat in ("absences",):
                val = c.number_input(label, min_value=0, max_value=93, value=0, step=1, key=feat)
            elif feat in ("G1","G2"):
                val = c.number_input(label, min_value=0, max_value=20, value=10, step=1, key=feat)
            else:
                val = c.number_input(label, value=0, key=feat)
            inputs[feat] = val
        else:
            # dropdown
            opts_list = list(opts.keys())
            sel = c.selectbox(label, opts_list, index=0, key=feat)
            inputs[feat] = opts[sel]  # map displayed label to encoded value

    submitted = st.form_submit_button("Predict")

# -----------------------
# When submitted: validate and predict
# -----------------------
def validate_inputs(inp: Dict[str, Any]) -> List[str]:
    errs = []
    # quick sanity checks
    for f in REQUIRED:
        if f not in inp:
            errs.append(f"Missing {f}")
    return errs

if submitted:
    errs = validate_inputs(inputs)
    if errs:
        st.error("Validation errors: " + "; ".join(errs))
    else:
        # Create DataFrame with exact FEATURE_ORDER so column names line up for ColumnTransformer
        X = pd.DataFrame([inputs], columns=FEATURE_ORDER)

        # Predict
        try:
            prob = None
            try:
                prob = float(MODEL.predict_proba(X)[:,1][0])
            except Exception:
                prob = None
            pred = int(MODEL.predict(X)[0])

            col1, col2 = st.columns((1,1))
            with col1:
                st.metric("Prediction (class)", "High stress (1)" if pred==1 else "Low stress (0)")
            with col2:
                st.metric("Probability", f"{prob:.3f}" if prob is not None else "N/A")

            # Show the input we sent (for reproducibility)
            st.markdown("#### Input (sent to model)")
            st.json(inputs)

            # allow export
            st.download_button("Download input JSON", data=json.dumps(inputs, indent=2), file_name="stress_input.json")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)
