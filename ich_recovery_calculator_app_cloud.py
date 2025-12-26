
import math
import streamlit as st

# =========================
# ICH Consciousness Recovery Calculator (6 months)
# Based on multivariable logistic regression (development cohort n=516)
# Outcome: Recovery of consciousness at 6 months (follow commands) = 1
# =========================

st.set_page_config(page_title="ICH 6个月意识恢复预测（在线计算器）", layout="centered")

st.title("自发性 ICH（术前昏迷，GCS≤8）术后 6 个月意识恢复预测")
st.caption("基于多因素 Logistic 回归（开发队列 n=516）。结局：6个月“能遵嘱”=恢复。")

with st.expander("⚠️ 重要声明（建议投稿版本保留）", expanded=True):
    st.markdown("""
- 本工具用于**科研/教学/临床辅助沟通**，不构成医疗建议或替代临床决策。  
- 模型基于**单中心回顾性开发队列**（n=516）；在外部人群使用前建议进行**外部验证与（必要时）重新标定**。  
- 若输入值**超出开发队列范围**，预测可能存在外推风险，页面会提示。  
""")

with st.expander("变量定义（与你数据字典一致）", expanded=False):
    st.markdown("""
- **GCS-Pupils score（GCS-P）**：综合 GCS 与瞳孔反射的评分  
- **Hematoma volume（mL）**：血肿体积（你的表中为 Hematoma size）  
- **IVH 分级（0–4）**：0=未破入；1=<25%；2=25–50%；3=50–75%；4=>75%  
- **Ventricle enlargement**：1=脑室扩大；0=无  
- **Midline shift（mm）**：最大中线移位  
- **Blood glucose（mmol/L）**：入院血糖  
""")

# ---- Development cohort ranges (observed) ----
RANGES = {'Age': {'max': 89.0, 'median': 57.0, 'min': 17.0, 'p1': 32.3, 'p99': 84.0},
 'Blood_glucose': {'max': 30.3, 'median': 8.8, 'min': 1.9, 'p1': 5.015, 'p99': 20.62500000000001},
 'GCS_P': {'max': 8.0, 'median': 6.0, 'min': 1.0, 'p1': 1.0, 'p99': 8.0},
 'Hematoma_volume': {'max': 173.4671222,
                     'median': 56.41256875,
                     'min': 25.063675,
                     'p1': 26.7790338,
                     'p99': 143.13293718000008},
 'IVH_grade': {'max': 4.0, 'median': 1.0, 'min': 0.0, 'p1': 0.0, 'p99': 4.0},
 'Midline_shift': {'max': 22.8, 'median': 10.2, 'min': 0.0, 'p1': 3.5, 'p99': 19.968500000000006},
 'Ventricle_enlargement': {'max': 1.0, 'median': 0.0, 'min': 0.0, 'p1': 0.0, 'p99': 1.0}}

def warn_out_of_range(name: str, value: float):
    r = RANGES[name]
    if value < r["min"] or value > r["max"]:
        st.error(f"⚠️ {name} = {value} 超出开发队列观测范围 [{r['min']}, {r['max']}]，存在外推风险。")
    elif value < r["p1"] or value > r["p99"]:
        st.warning(f"提示：{name} = {value} 位于开发队列的极端区间（<P1 或 >P99），预测不确定性可能更高。")

# ---- Inputs ----
st.subheader("输入患者术前信息")

age = st.number_input("年龄（years）", min_value=0.0, max_value=120.0, value=float(RANGES["Age"]["median"]), step=1.0)
gcs_p = st.slider("GCS-Pupils score（1–8）", min_value=1, max_value=8, value=int(RANGES["GCS_P"]["median"]), step=1)

hematoma_vol = st.number_input("血肿体积（mL）", min_value=0.0, max_value=300.0, value=float(RANGES["Hematoma_volume"]["median"]), step=1.0)
ivh_grade = st.slider("破入脑室分级（0–4）", min_value=0, max_value=4, value=int(RANGES["IVH_grade"]["median"]), step=1)

vent_enl = st.radio("脑室扩大（Ventricle enlargement）", options=[0, 1], index=int(RANGES["Ventricle_enlargement"]["median"]),
                    format_func=lambda x: "否（0）" if x == 0 else "是（1）")

mls = st.number_input("最大中线移位（mm）", min_value=0.0, max_value=40.0, value=float(RANGES["Midline_shift"]["median"]), step=0.1)
glu = st.number_input("入院血糖（mmol/L）", min_value=0.0, max_value=60.0, value=float(RANGES["Blood_glucose"]["median"]), step=0.1)

# ---- Range warnings ----
st.subheader("输入范围检查")
warn_out_of_range("Age", float(age))
warn_out_of_range("GCS_P", float(gcs_p))
warn_out_of_range("Hematoma_volume", float(hematoma_vol))
warn_out_of_range("IVH_grade", float(ivh_grade))
warn_out_of_range("Ventricle_enlargement", float(vent_enl))
warn_out_of_range("Midline_shift", float(mls))
warn_out_of_range("Blood_glucose", float(glu))

# ---- Coefficients (from fitted model) ----
B0 = 5.706245
B_AGE = -0.116444
B_GCSP = 0.734351
B_HEM = 0.005742
B_IVH = -0.208413
B_VENT = -1.470830
B_MLS = -0.041931
B_GLU = -0.166805

logit = (
    B0
    + B_AGE * float(age)
    + B_GCSP * float(gcs_p)
    + B_HEM * float(hematoma_vol)
    + B_IVH * float(ivh_grade)
    + B_VENT * float(vent_enl)
    + B_MLS * float(mls)
    + B_GLU * float(glu)
)
p = 1 / (1 + math.exp(-logit))

st.divider()
st.subheader("预测结果")
st.metric("6 个月意识恢复概率（能遵嘱）", f"{p*100:.1f}%")

# Risk bands (editable)
if p >= 0.70:
    band = "较高概率恢复"
    st.success(f"风险分层：{band}")
elif p >= 0.40:
    band = "中等概率恢复"
    st.warning(f"风险分层：{band}")
else:
    band = "较低概率恢复"
    st.error(f"风险分层：{band}")

st.subheader("一键生成可复制结果语句")
statement = (
    f"ICH术后6个月意识恢复预测（Logistic回归）：预测概率={p*100:.1f}%，风险分层：{band}。"
    f"输入：年龄={age:.0f}岁，GCS-P={gcs_p}，血肿体积={hematoma_vol:.1f}mL，"
    f"IVH分级={ivh_grade}，脑室扩大={vent_enl}，中线移位={mls:.1f}mm，血糖={glu:.1f}mmol/L。"
)
st.text_area("复制下面这段文字（Cmd/Ctrl+C）", value=statement, height=120)

with st.expander("计算细节（用于科研复现/补充材料）", expanded=False):
    st.code(
        f"Intercept (β0) = {B0}\n"
        f"logit = β0 + β_age*Age + β_gcsp*GCS-P + β_hem*HematomaVol + β_ivh*IVH + β_vent*VentricleEnl + β_mls*MLS + β_glu*Glucose\n\n"
        f"logit = {logit:.4f}\n"
        f"P = 1 / (1 + exp(-logit)) = {p:.4f}"
    )
    st.markdown("""
**投稿建议**：在 Supplementary Material 提供此公式与系数表，并在正文声明“外部验证前仅作科研用途”。  
""")
