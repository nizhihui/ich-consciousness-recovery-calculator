# -*- coding: utf-8 -*-
"""
SCI-ready web calculator (Streamlit) — with built-in Publication (static) view
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

def label_yesno01(x: int) -> str:
    return "Yes (1)" if x == 1 else "No (0)"


# =========================
# UI
# =========================
st.set_page_config(
    page_title="ICH Consciousness Recovery Calculator (6 months)",
    page_icon="🧠",
    layout="centered",
)

# ---- Header ----
st.title("Prediction of 6‑Month Postoperative Consciousness Recovery")
st.caption(
    "Comatose patients with spontaneous supratentorial intracerebral hemorrhage (GCS ≤ 8). "
    "Web-based calculator derived from a multivariable logistic regression model (development cohort, n = 516)."
)

# ---- Publication / Screenshot mode toggle (TOP, always visible) ----
publication_view = st.toggle(
    "Publication (static) view for figures",
    value=False,
    help=(
        "Enable this mode to generate a clean, non-interactive layout suitable for screenshots in manuscripts. "
        "Interactive controls, warnings, and long text blocks are minimized."
    ),
)

# Keep the page short in publication view: collapse long sections by default
notice_expanded = False if publication_view else True
defs_expanded = False

with st.expander("Important Notice (recommended for manuscript submission)", expanded=notice_expanded):
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

with st.expander("Predictor Definitions", expanded=defs_expanded):
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

# ---- Inputs ----
# Use session_state defaults so switching modes doesn't reset user-entered values
def _get_default(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# defaults chosen only for demonstration; user can change in interactive mode
d_age = _get_default("age", 67)
d_gcsp = _get_default("gcsp", 6)
d_vol = _get_default("volume", 40.0)
d_ivh = _get_default("ivh", 2)
d_vent = _get_default("vent", 0)
d_mls = _get_default("mls", 10.0)
d_glu = _get_default("glu", 9.4)

if publication_view:
    # Static: show values as text (no sliders/inputs)
    age = int(d_age)
    gcsp = int(d_gcsp)
    volume = float(d_vol)
    ivh = int(d_ivh)
    vent = int(d_vent)
    mls = float(d_mls)
    glu = float(d_glu)

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Age (years):** {age}")
        st.markdown(f"**GCS‑Pupils score (1–8):** {gcsp}")
        st.markdown(f"**Hematoma volume (mL):** {volume:.1f}")
        st.markdown(f"**IVH grade (0–4):** {ivh}")
    with right:
        st.markdown(f"**Ventricular enlargement:** {label_yesno01(vent)}")
        st.markdown(f"**Midline shift (mm):** {mls:.1f}")
        st.markdown(f"**Admission blood glucose (mmol/L):** {glu:.1f}")

else:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=int(d_age), step=1, key="age")
        gcsp = st.slider("GCS‑Pupils score (1–8)", min_value=1, max_value=8, value=int(d_gcsp), step=1, key="gcsp")
        volume = st.number_input(
            "Hematoma volume (mL)", min_value=0.0, max_value=300.0, value=float(d_vol), step=1.0, key="volume"
        )
        ivh = st.selectbox("IVH grade (0–4)", options=[0, 1, 2, 3, 4], index=int(d_ivh), key="ivh")

    with col2:
        vent = st.selectbox(
            "Ventricular enlargement",
            options=[0, 1],
            index=int(d_vent),
            key="vent",
            format_func=label_yesno01,
        )
        mls = st.number_input("Midline shift (mm)", min_value=0.0, max_value=60.0, value=float(d_mls), step=0.5, key="mls")
        glu = st.number_input(
            "Admission blood glucose (mmol/L)", min_value=0.0, max_value=60.0, value=float(d_glu), step=0.1, key="glu"
        )

# ---- Range warnings (suppressed in publication view) ----
if not publication_view:
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

# ---- Prediction ----
p, lp = predict_probability(age, gcsp, volume, ivh, vent, mls, glu)

# ---- Results ----
st.markdown("### Result")

if publication_view:
    st.markdown(f"**Estimated probability of consciousness recovery at 6 months: {p*100:.1f}%**")
    st.caption("Based on a multivariable logistic regression model.")
else:
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

st.divider()
st.caption(
    "Model reporting follows the TRIPOD statement for prediction model development. "
    "© 2025. For academic use only."
)
