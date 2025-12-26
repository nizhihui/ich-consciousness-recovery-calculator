# -*- coding: utf-8 -*-
"""
SCI-ready web calculator (Streamlit)
Prediction of 6-Month Postoperative Consciousness Recovery
in Comatose Patients With Spontaneous Intracerebral Hemorrhage (ICH)

This app is intended for research/educational use only.
"""

import math
import streamlit as st

# =========================
# Model coefficients
# =========================
# NOTE: Replace coefficients below if you refit / recalibrate the model.
B0 = 5.706245
B_AGE = -0.116444
B_GCSP = 0.734351
B_VOL = 0.005742
B_IVH = -0.208413
B_VENT = -1.470830
B_MLS = -0.041931
B_GLU = -0.166805

# =========================
# Development cohort ranges
# =========================
# Used ONLY for out-of-range warnings (not for calculations).
DEV_MINMAX = {
    "age": (16, 96),
    "gcsp": (1, 8),
    "volume": (0.0, 200.0),
    "ivh": (0, 4),
    "vent": (0, 1),
    "mls": (0.0, 30.0),
    "glu": (2.0, 30.0),
}

DEV_P1P99 = {
    "age": (25, 88),
    "gcsp": (1, 8),
    "volume": (10.0, 120.0),
    "ivh": (0, 4),
    "mls": (0.0, 20.0),
    "glu": (4.0, 20.0),
}

# =========================
# Helpers
# =========================
def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def predict_probability(age, gcsp, volume, ivh_grade, vent_enlarged, midline_shift, glucose):
    lp = (
        B0
        + B_AGE * age
        + B_GCSP * gcsp
        + B_VOL * volume
        + B_IVH * ivh_grade
        + B_VENT * vent_enlarged
        + B_MLS * midline_shift
        + B_GLU * glucose
    )
    return sigmoid(lp), lp

def range_flag(value, key):
    mn, mx = DEV_MINMAX[key]
    if value < mn or value > mx:
        return "red"
    p1, p99 = DEV_P1P99.get(key, (mn, mx))
    if value < p1 or value > p99:
        return "yellow"
    return None

def risk_band(p):
    # Interpretive bands are for communication only (not a clinical guideline).
    if p < 0.20:
        return "Low"
    if p < 0.50:
        return "Intermediate"
    return "High"

# =========================
# UI
# =========================
st.set_page_config(
    page_title="ICH Consciousness Recovery Calculator (6 months)",
    page_icon="🧠",
    layout="centered",
)

st.title("Prediction of 6‑Month Postoperative Consciousness Recovery")
st.caption(
    "Comatose patients with spontaneous supratentorial intracerebral hemorrhage (GCS ≤ 8). "
    "Web-based calculator derived from a multivariable logistic regression model (development cohort, n = 516)."
)

with st.expander("Important Notice (recommended for manuscript submission)", expanded=True):
    st.markdown(
        """
**Research/Educational Use Only.** This web-based calculator is intended for research and educational purposes and **must not**
be used as a substitute for clinical judgment or to make medical decisions.

**Model provenance.** The model was developed using a **single-center retrospective cohort (n = 516)** and has **not yet**
undergone external validation.

**Generalizability.** External validation and (if needed) recalibration are recommended before use in other settings or populations.

**Out-of-range inputs.** If entered values are outside the development cohort range, predictions may involve extrapolation and
should be interpreted with additional caution.
        """
    )

with st.expander("Predictor Definitions", expanded=False):
    st.markdown(
        """
- **GCS‑Pupils score (GCS‑P):** Combined assessment of Glasgow Coma Scale and pupillary reactivity  
- **Hematoma volume (mL)**  
- **Intraventricular hemorrhage (IVH) grade (0–4):** 0 = none; 1 = <25%; 2 = 25–50%; 3 = 50–75%; 4 = >75% of ventricular volume  
- **Ventricular enlargement:** 1 = yes, 0 = no  
- **Midline shift (mm):** Maximum midline shift  
- **Admission blood glucose (mmol/L)**
        """
    )

st.subheader("Enter preoperative patient information")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=67, step=1)
    gcsp = st.slider("GCS‑Pupils score (1–8)", min_value=1, max_value=8, value=6, step=1)
    volume = st.number_input("Hematoma volume (mL)", min_value=0.0, max_value=300.0, value=40.0, step=1.0)
    ivh = st.selectbox("IVH grade (0–4)", options=[0, 1, 2, 3, 4], index=2)

with col2:
    vent = st.selectbox(
        "Ventricular enlargement",
        options=[0, 1],
        index=0,
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
    )
    mls = st.number_input("Midline shift (mm)", min_value=0.0, max_value=60.0, value=8.0, step=0.5)
    glu = st.number_input("Admission blood glucose (mmol/L)", min_value=0.0, max_value=60.0, value=9.0, step=0.1)

# Range warnings
warns = []
for k, v in [
    ("age", age),
    ("gcsp", gcsp),
    ("volume", volume),
    ("ivh", ivh),
    ("vent", vent),
    ("mls", mls),
    ("glu", glu),
]:
    f = range_flag(v, k)
    if f == "red":
        mn, mx = DEV_MINMAX[k]
        warns.append(("red", f"**{k}** is outside the development cohort range ({mn}–{mx})."))
    elif f == "yellow":
        p1, p99 = DEV_P1P99.get(k, DEV_MINMAX[k])
        warns.append(("yellow", f"**{k}** is outside the typical range (approx. P1–P99: {p1}–{p99})."))

for level, msg in warns:
    if level == "red":
        st.error("Extrapolation warning: " + msg)
    else:
        st.warning("Caution: " + msg)

st.divider()

p, lp = predict_probability(age, gcsp, volume, ivh, vent, mls, glu)

st.markdown("### Result")
st.metric("Estimated probability of consciousness recovery at 6 months", f"{p*100:.1f}%")

band = risk_band(p)
st.caption(
    f"Interpretive band (for communication only): **{band}**. "
    "This estimate is based on a multivariable logistic regression model incorporating "
    "age, GCS‑Pupils score, hematoma volume, IVH grade, ventricular enlargement, midline shift, and admission blood glucose."
)

st.markdown("### Copy-ready statement (for manuscript/clinical communication)")
copy_text = (
    f"Predicted probability of 6-month postoperative consciousness recovery: {p*100:.1f}% "
    f"(logistic regression model; predictors: age={age} years, GCS-P={gcsp}, hematoma volume={volume:.1f} mL, "
    f"IVH grade={ivh}, ventricular enlargement={vent}, midline shift={mls:.1f} mm, admission glucose={glu:.1f} mmol/L). "
    "For research/educational use only; not a substitute for clinical judgment. External validation recommended."
)
st.code(copy_text, language="text")

with st.expander("Model specification (for reproducibility)", expanded=False):
    st.markdown(
        f"""
The model uses the following form:

- Linear predictor: **LP = β0 + Σ(βi · Xi)**
- Probability: **P = 1 / (1 + exp(−LP))**

Coefficients used in this app:

- Intercept (β0): `{B0}`
- Age (years): `{B_AGE}`
- GCS‑Pupils score: `{B_GCSP}`
- Hematoma volume (mL): `{B_VOL}`
- IVH grade (0–4): `{B_IVH}`
- Ventricular enlargement (0/1): `{B_VENT}`
- Midline shift (mm): `{B_MLS}`
- Admission blood glucose (mmol/L): `{B_GLU}`
        """
    )
    st.caption("Tip: If you refit or recalibrate the model, update the coefficients above and redeploy.")

st.divider()
st.caption(
    "Model reporting follows the TRIPOD statement for prediction model development. "
    "© 2025. For academic use only."
)
