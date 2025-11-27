#!/usr/bin/env python3
"""
Final functional desktop GUI for Academic Stress EWS.
"""

import os
import json
import traceback
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np

MODEL_PATH = "/Users/aryansahoo/Documents/Academic_Stress_EWS_Project/models/logistic_pipeline.joblib"
FEATURES_PATH = "/Users/aryansahoo/Documents/Academic_Stress_EWS_Project/models/feature_names.json"

REQUIRED_FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2"
]

try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at: {MODEL_PATH}\n\n{e}")

if os.path.exists(FEATURES_PATH):
    try:
        with open(FEATURES_PATH, "r") as fh:
            FEATURE_ORDER = json.load(fh)
        if not all(f in FEATURE_ORDER for f in REQUIRED_FEATURES):
            FEATURE_ORDER = REQUIRED_FEATURES.copy()
    except Exception:
        FEATURE_ORDER = REQUIRED_FEATURES.copy()
else:
    FEATURE_ORDER = REQUIRED_FEATURES.copy()


# ---------------- FRIENDLY LABELS & OPTIONS ----------------

FIELD_INFO = [
    ("school", "School", {"GP": "GP", "MS": "MS"}),
    ("sex", "Sex", {"F": "F", "M": "M"}),
    ("age", "Age", None),
    ("address", "Address", {"U (urban)": "U", "R (rural)": "R"}),
    ("famsize", "Family Size", {"GT3 (>=3)": "GT3", "LE3 (<3)": "LE3"}),
    ("Pstatus", "Parent Cohabitation", {"T (together)": "T", "A (apart)": "A"}),
    ("Medu", "Mother's Education", {"0 - None": 0, "1 - Primary": 1, "2 - 5th–9th": 2, "3 - Secondary": 3, "4 - Higher": 4}),
    ("Fedu", "Father's Education", {"0 - None": 0, "1 - Primary": 1, "2 - 5th–9th": 2, "3 - Secondary": 3, "4 - Higher": 4}),
    ("Mjob", "Mother's Job", {"teacher": "teacher", "health": "health", "services": "services", "at_home": "at_home", "other": "other"}),
    ("Fjob", "Father's Job", {"teacher": "teacher", "health": "health", "services": "services", "at_home": "at_home", "other": "other"}),
    ("reason", "Reason for School Choice", {"home": "home", "reputation": "reputation", "course": "course", "other": "other"}),
    ("guardian", "Guardian", {"mother": "mother", "father": "father", "other": "other"}),
    ("traveltime", "Travel Time", {"1 - <15 min": 1, "2 - 15–30 min": 2, "3 - 30–60 min": 3, "4 - >60 min": 4}),
    ("studytime", "Weekly Study Time", {"1 - <2 hrs": 1, "2 - 2–5 hrs": 2, "3 - 5–10 hrs": 3, "4 - >10 hrs": 4}),
    ("failures", "Past Failures", {"0": 0, "1": 1, "2": 2, "3": 3}),
    ("schoolsup", "School Support", {"yes": 1, "no": 0}),
    ("famsup", "Family Support", {"yes": 1, "no": 0}),
    ("paid", "Paid Classes", {"yes": 1, "no": 0}),
    ("activities", "Activities", {"yes": 1, "no": 0}),
    ("nursery", "Nursery", {"yes": 1, "no": 0}),
    ("higher", "Wants Higher Edu", {"yes": 1, "no": 0}),
    ("internet", "Internet", {"yes": 1, "no": 0}),
    ("romantic", "Romantic", {"yes": 1, "no": 0}),
    ("famrel", "Family Relationship", {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}),
    ("freetime", "Free Time", {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}),
    ("goout", "Going Out", {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}),
    ("Dalc", "Workday Alcohol", {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}),
    ("Walc", "Weekend Alcohol", {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}),
    ("health", "Health", {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}),
    ("absences", "Absences", None),
    ("G1", "G1 (0–20)", None),
    ("G2", "G2 (0–20)", None),
]

FIELD_INFO_MAP = {feat: (label, opts) for feat, label, opts in FIELD_INFO}


# ---------------- UI BUILD ----------------

root = tk.Tk()
root.title("Academic Stress EWS")
root.geometry("900x700")
root.configure(bg="white")

main = ttk.Frame(root)
main.pack(fill="both", expand=True, padx=8, pady=8)

left = ttk.Frame(main)
right = ttk.Frame(main, width=300)
left.pack(side="left", fill="both", expand=True)
right.pack(side="right", fill="y")

branding = ttk.Label(left, text="Academic Stress EWS", font=("Helvetica", 18, "bold"))
branding.pack(anchor="nw", pady=(6, 12))


# ---------------- FIXED SCROLLABLE AREA (no black section) ----------------

canvas = tk.Canvas(left, highlightthickness=0, bg="white")
scrollbar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)

form_frame = ttk.Frame(canvas)

canvas.create_window((0, 0), window=form_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=False)   # << The key fix

def _on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))
    canvas.config(height=min(form_frame.winfo_height(), 600))  # dynamically shrink

form_frame.bind("<Configure>", _on_frame_configure)


# ---------------- FORM ELEMENTS ----------------

widget_vars = {}
error_labels = {}

def add_dropdown(parent, feature, label_text, options_map):
    frame = ttk.Frame(parent)
    frame.pack(fill="x", pady=6, padx=6)
    ttk.Label(frame, text=label_text).pack(anchor="w")
    var = tk.StringVar()
    combo = ttk.Combobox(frame, textvariable=var, values=list(options_map.keys()),
                         state="readonly", width=44)
    combo.pack(anchor="w")
    combo.set(list(options_map.keys())[0])
    widget_vars[feature] = ("dropdown", var, options_map)
    err = ttk.Label(frame, text="", foreground="red")
    err.pack(anchor="w")
    error_labels[feature] = err

def add_spinbox(parent, feature, label_text, minval, maxval):
    frame = ttk.Frame(parent)
    frame.pack(fill="x", pady=6, padx=6)
    ttk.Label(frame, text=label_text).pack(anchor="w")
    var = tk.StringVar()
    spin = tk.Spinbox(frame, from_=minval, to=maxval, textvariable=var, width=10)
    spin.pack(anchor="w")
    var.set(str(minval))
    widget_vars[feature] = ("spin", var, (minval, maxval))
    err = ttk.Label(frame, text="", foreground="red")
    err.pack(anchor="w")
    error_labels[feature] = err

for feat in FEATURE_ORDER:
    label, opts = FIELD_INFO_MAP.get(feat, (feat, None))
    if opts is None:
        if feat == "age":
            add_spinbox(form_frame, feat, label, 10, 30)
        elif feat == "absences":
            add_spinbox(form_frame, feat, label, 0, 93)
        elif feat in ("G1", "G2"):
            add_spinbox(form_frame, feat, label, 0, 20)
        else:
            add_spinbox(form_frame, feat, label, 0, 100)
    else:
        add_dropdown(form_frame, feat, label, opts)


# ---------------- OUTPUT PANEL ----------------

res_title = ttk.Label(right, text="Prediction", font=("Helvetica", 14, "bold"))
res_title.pack(pady=(12, 8))

pred_var = tk.StringVar(value="—")
prob_var = tk.StringVar(value="—")

ttk.Label(right, text="Class:").pack(anchor="w", padx=12)
ttk.Label(right, textvariable=pred_var, font=("Helvetica", 12)).pack(anchor="w", padx=12, pady=(0, 8))

ttk.Label(right, text="Probability:").pack(anchor="w", padx=12)
ttk.Label(right, textvariable=prob_var, font=("Helvetica", 12)).pack(anchor="w", padx=12, pady=(0, 12))


# ---------------- LOGIC ----------------

def safe_cast_int(x):
    try:
        return int(float(x))
    except:
        return None

def gather_and_map_inputs():
    errors, mapped = {}, {}
    for feat in FEATURE_ORDER:
        t, var, extra = widget_vars[feat]
        if t == "dropdown":
            val = var.get()
            mapped_val = extra.get(val)
            if mapped_val is None:
                errors[feat] = "Choose a valid option"
            else:
                mapped[feat] = mapped_val

        else:
            val = var.get()
            val_i = safe_cast_int(val)
            if val_i is None:
                errors[feat] = "Enter a number"
            else:
                mapped[feat] = val_i

    if errors:
        return False, errors

    return True, [mapped[f] for f in FEATURE_ORDER]


def predict_action():
    for e in error_labels.values():
        e.config(text="")

    ok, payload = gather_and_map_inputs()
    if not ok:
        for feat, msg in payload.items():
            error_labels[feat].config(text=msg)
        messagebox.showerror("Input Error", "Fix the highlighted fields.")
        return

    try:
        import pandas as pd
        X = pd.DataFrame([payload], columns=FEATURE_ORDER)

        try:
            prob = float(MODEL.predict_proba(X)[:, 1][0])
        except:
            prob = None

        pred = int(MODEL.predict(X)[0])
        pred_var.set("1 (High stress)" if pred == 1 else "0 (Low stress)")
        prob_var.set(f"{prob:.4f}" if prob is not None else "N/A")

    except Exception as e:
        tb = traceback.format_exc()
        messagebox.showerror("Prediction error", f"{e}\n\n{tb}")


ttk.Button(right, text="Predict", command=predict_action).pack(pady=20, padx=12)

def clear_form():
    for feat, (t, var, extra) in widget_vars.items():
        if t == "dropdown":
            var.set(list(extra.keys())[0])
        else:
            var.set(str(extra[0]))
        error_labels[feat].config(text="")

ttk.Button(right, text="Clear", command=clear_form).pack(pady=(0, 6), padx=12)

root.mainloop()
