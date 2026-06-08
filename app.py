import streamlit as st
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# CSS — Subtle, Clean, Professional
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Lora:wght@500;600&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* Background — soft slate, not harsh black */
.stApp {
    background-color: #f1f5f9;
    min-height: 100vh;
}

.block-container {
    padding: 2.5rem 3.5rem !important;
    max-width: 1200px !important;
}

/* Hero */
.hero {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border-left: 4px solid #4f7cac;
}
.hero-title {
    font-family: 'Lora', serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #64748b;
    font-size: 0.97rem;
    font-weight: 400;
    margin: 0;
}

/* Section heading */
.section-heading {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e2e8f0;
}

/* Input labels */
.stSlider label,
.stSelectbox label {
    color: #475569 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: #4f7cac !important;
}
.stSlider [data-baseweb="thumb"] {
    background: #ffffff !important;
    border: 2px solid #4f7cac !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.15) !important;
}

/* Selectboxes */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
}

/* Primary button */
.stButton > button {
    background: #4f7cac !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    height: 3em !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 1px 3px rgba(79,124,172,0.3) !important;
    transition: background 0.2s ease !important;
}
.stButton > button:hover {
    background: #3b6899 !important;
}

/* Reset button */
.reset-btn > button {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: #64748b !important;
    box-shadow: none !important;
}
.reset-btn > button:hover {
    border-color: #cbd5e1 !important;
    color: #475569 !important;
    background: #f8fafc !important;
}

/* Divider */
hr { border-color: #e2e8f0 !important; margin: 1.5rem 0 !important; }

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.75rem 2rem;
    border-top: 3px solid #4f7cac;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Lora', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #1e293b;
    line-height: 1.15;
}
.metric-sub {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}

.risk-high { border-top-color: #e05252; }
.risk-high .metric-value { color: #c0392b; }

.risk-low  { border-top-color: #4caf84; }
.risk-low  .metric-value { color: #2e7d5e; }

/* Recommendation cards */
.tip-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
}
.tip-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 0.3rem;
}
.tip-body {
    font-size: 0.8rem;
    color: #64748b;
    line-height: 1.55;
}

/* General text */
p, label { color: #475569 !important; }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load Models
# -----------------------------
def load_model(file):
    if not os.path.exists(file):
        st.error(f"Missing file: `{file}`. Ensure all model files are present.")
        st.stop()
    return pickle.load(open(file, "rb"))

reg    = load_model("reg_model.pkl")
clf    = load_model("clf_model.pkl")
scaler = load_model("scaler.pkl")


# -----------------------------
# Hero
# -----------------------------
st.markdown("""
<div class="hero">
    <p class="hero-title">MediPredict AI</p>
    <p class="hero-sub">Predict estimated medical costs and disease risk based on your health profile.</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Section 1 — Personal Details
# -----------------------------
st.markdown('<div class="section-heading">Personal Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age      = st.slider("Age", 0, 100, 25)
with col2:
    bmi      = st.slider("BMI", 10.0, 50.0, 22.0, step=0.1)
with col3:
    children = st.slider("Number of Children", 0, 5, 0)

col4, col5, col6 = st.columns(3)
with col4:
    gender   = st.selectbox("Gender",         ["Male", "Female"])
with col5:
    smoker   = st.selectbox("Smoker",          ["No", "Yes"])
with col6:
    activity = st.selectbox("Activity Level",  ["Low", "Medium", "High"])

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Section 2 — Coverage & Location
# -----------------------------
st.markdown('<div class="section-heading">Coverage & Location</div>', unsafe_allow_html=True)

col7, col8, _ = st.columns([1, 1, 1])
with col7:
    insurance = st.selectbox("Insurance Plan", ["Basic", "Premium"])
with col8:
    city      = st.selectbox("City Type",      ["Urban", "Semi-Urban", "Rural"])

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Section 3 — Medical History
# -----------------------------
st.markdown('<div class="section-heading">Medical History</div>', unsafe_allow_html=True)

col9, col10, col11, col12 = st.columns(4)
with col9:
    diabetes     = st.selectbox("Diabetes",      [0, 1], format_func=lambda x: "Yes" if x else "No")
with col10:
    hypertension = st.selectbox("Hypertension",  [0, 1], format_func=lambda x: "Yes" if x else "No")
with col11:
    heart        = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
with col12:
    asthma       = st.selectbox("Asthma",        [0, 1], format_func=lambda x: "Yes" if x else "No")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Encoding
# -----------------------------
gender_enc    = 1 if gender == "Male" else 0
smoker_enc    = 1 if smoker == "Yes" else 0
activity_enc  = {"Low": 0, "Medium": 1, "High": 2}[activity]
insurance_enc = {"Basic": 0, "Premium": 1}[insurance]
city_enc      = {"Urban": 0, "Semi-Urban": 1, "Rural": 2}[city]

# -----------------------------
# Action Buttons
# -----------------------------
btn1, btn2, btn3, _ = st.columns([2, 1.2, 0.9, 1.5])

with btn1:
    predict_clicked = st.button("Run Full Prediction", use_container_width=True)
with btn2:
    cost_only = st.button("Estimate Cost Only", use_container_width=True)
with btn3:
    st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    reset = st.button("Reset", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# Prediction
# -----------------------------
def get_prediction():
    data = np.array([[
        age, gender_enc, bmi, children, smoker_enc,
        activity_enc, insurance_enc, city_enc,
        diabetes, hypertension, heart, asthma
    ]])
    scaled  = scaler.transform(data)
    cost    = reg.predict(scaled)[0]
    disease = clf.predict(scaled)[0]
    return cost, disease, scaled


if predict_clicked or cost_only:
    cost, disease, _ = get_prediction()

    st.markdown("---")
    st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)

    # Metric cards
    if predict_clicked:
        mc1, mc2 = st.columns(2)
    else:
        mc1, mc2 = st.columns([1, 1])

    with mc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated Annual Medical Cost</div>
            <div class="metric-value">&#8377; {cost:,.0f}</div>
            <div class="metric-sub">Based on your demographic and health profile</div>
        </div>
        """, unsafe_allow_html=True)

    if predict_clicked:
        with mc2:
            rc    = "risk-high" if disease == 1 else "risk-low"
            rlbl  = "High Risk" if disease == 1 else "Low Risk"
            rsub  = "We recommend consulting a specialist." if disease == 1 else "Your profile indicates a low disease risk."
            st.markdown(f"""
            <div class="metric-card {rc}">
                <div class="metric-label">Disease Risk Assessment</div>
                <div class="metric-value">{rlbl}</div>
                <div class="metric-sub">{rsub}</div>
            </div>
            """, unsafe_allow_html=True)

    # Charts
    st.markdown("<br>", unsafe_allow_html=True)

    BG   = "#ffffff"
    AX   = "#f8fafc"
    GRID = "#e2e8f0"
    TEXT = "#64748b"
    BLUE = "#4f7cac"

    ch1, ch2 = st.columns(2)

    # Chart 1 — Profile snapshot
    with ch1:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(AX)

        features = ["Age", "BMI", "Children", "Activity"]
        vals     = [age, bmi, children, activity_enc]
        palette  = ["#4f7cac", "#6b9bc4", "#84b4d4", "#a8cce0"]

        bars = ax.bar(features, vals, color=palette, width=0.5, zorder=3, edgecolor='none')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(round(v, 1)), ha='center', va='bottom',
                    color='#334155', fontsize=9, fontweight='600')

        ax.set_title("Patient Profile Overview", color=TEXT, fontsize=10, pad=10, fontweight='600')
        ax.tick_params(colors=TEXT, labelsize=8.5)
        ax.spines[:].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.3 + 2)
        ax.yaxis.set_visible(False)
        ax.grid(axis='y', color=GRID, linestyle='-', linewidth=0.8, zorder=0)
        fig.tight_layout()
        st.pyplot(fig)

    # Chart 2 — Condition flags
    with ch2:
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
        fig2.patch.set_facecolor(BG)
        ax2.set_facecolor(AX)

        conditions  = ["Diabetes", "Hypertension", "Heart Disease", "Asthma", "Smoker"]
        cond_vals   = [diabetes, hypertension, heart, asthma, smoker_enc]
        bar_colors  = ["#c0392b" if v else "#4caf84" for v in cond_vals]

        hbars = ax2.barh(conditions, cond_vals, color=bar_colors, height=0.4, zorder=3, edgecolor='none')
        for bar, v in zip(hbars, cond_vals):
            lbl = "Present" if v else "Absent"
            ax2.text(v + 0.03, bar.get_y() + bar.get_height()/2,
                     lbl, va='center', color='#334155', fontsize=8.5, fontweight='500')

        ax2.set_title("Medical Condition Status", color=TEXT, fontsize=10, pad=10, fontweight='600')
        ax2.set_xlim(0, 1.5)
        ax2.tick_params(colors=TEXT, labelsize=8.5)
        ax2.spines[:].set_visible(False)
        ax2.xaxis.set_visible(False)

        legend_els = [
            mpatches.Patch(color='#c0392b', label='Present'),
            mpatches.Patch(color='#4caf84', label='Absent')
        ]
        ax2.legend(handles=legend_els, loc='lower right',
                   facecolor='#f8fafc', edgecolor='#e2e8f0',
                   labelcolor=TEXT, fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)

    # Charts 3 & 4 — full prediction only
    if predict_clicked:
        st.markdown("<br>", unsafe_allow_html=True)
        dc1, dc2 = st.columns(2)

        with dc1:
            fig3, ax3 = plt.subplots(figsize=(5, 3.8))
            fig3.patch.set_facecolor(BG)
            ax3.set_facecolor(BG)

            base_c   = age * 100
            bmi_c    = bmi * 80
            smk_c    = 4000 if smoker_enc else 0
            cond_c   = (diabetes + hypertension + heart + asthma) * 1200
            rem_c    = max(cost - base_c - bmi_c - smk_c - cond_c, 500)

            sizes  = [base_c, bmi_c, smk_c, cond_c, rem_c]
            labels = ["Age", "BMI", "Smoking", "Conditions", "Base"]
            colors = ["#4f7cac", "#84b4d4", "#e05252", "#f0a04b", "#4caf84"]

            wedges, _, autotexts = ax3.pie(
                sizes, labels=None, autopct='%1.0f%%', startangle=140,
                colors=colors, explode=[0.03]*5,
                wedgeprops=dict(width=0.55, edgecolor='white', linewidth=1.5),
                pctdistance=0.78,
                textprops=dict(color='#334155', fontsize=8, fontweight='600')
            )
            ax3.set_title("Cost Factor Breakdown", color=TEXT, fontsize=10, pad=10, fontweight='600')
            ax3.legend(wedges, labels, loc="lower center",
                       bbox_to_anchor=(0.5, -0.15), ncol=3,
                       facecolor='#f8fafc', edgecolor='#e2e8f0',
                       labelcolor=TEXT, fontsize=7.5)
            fig3.tight_layout()
            st.pyplot(fig3)

        with dc2:
            fig4, ax4 = plt.subplots(figsize=(5, 3.8))
            fig4.patch.set_facecolor(BG)
            ax4.set_facecolor(AX)

            coverage_pct  = 60 if insurance_enc == 0 else 85
            out_of_pocket = cost * (1 - coverage_pct / 100)

            cats  = ["Total Cost", "Covered", "Out of Pocket"]
            vals4 = [cost, cost * coverage_pct / 100, out_of_pocket]
            cols4 = [BLUE, "#4caf84", "#e05252"]

            bars4 = ax4.bar(cats, vals4, color=cols4, width=0.45, zorder=3, edgecolor='none')
            for bar, v in zip(bars4, vals4):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + cost * 0.01,
                         f"\u20b9{v:,.0f}", ha='center', va='bottom',
                         color='#334155', fontsize=8, fontweight='600')

            ax4.set_title(f"Coverage Breakdown  ({coverage_pct}% covered)", color=TEXT,
                          fontsize=10, pad=10, fontweight='600')
            ax4.tick_params(colors=TEXT, labelsize=8.5)
            ax4.spines[:].set_visible(False)
            ax4.yaxis.set_visible(False)
            ax4.set_ylim(0, max(vals4) * 1.25)
            ax4.grid(axis='y', color=GRID, linestyle='-', linewidth=0.8, zorder=0)
            fig4.tight_layout()
            st.pyplot(fig4)

    # Recommendations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-heading">Recommendations</div>', unsafe_allow_html=True)

    tips = []
    if smoker_enc:   tips.append(("Smoking Cessation",    "Smoking is a significant driver of both medical cost and disease risk. Consider cessation programmes."))
    if bmi > 30:     tips.append(("Weight Management",    f"A BMI of {bmi:.1f} is above the healthy range. Dietary adjustments and regular exercise can help."))
    if age > 50:     tips.append(("Routine Screenings",   "Annual health screenings are strongly advised for individuals above 50."))
    if diabetes:     tips.append(("Diabetes Care",        "Maintain consistent blood glucose monitoring and adhere to your prescribed medication schedule."))
    if hypertension: tips.append(("Blood Pressure Control","Reduce sodium intake, manage stress, and follow your physician's treatment plan."))
    if heart:        tips.append(("Cardiac Health",       "Maintain a low-sodium, heart-healthy diet and engage in moderate physical activity as advised."))
    if asthma:       tips.append(("Respiratory Care",     "Keep rescue medication accessible and avoid known environmental triggers."))
    if not tips:     tips.append(("Healthy Profile",      "Your current profile indicates low risk. Maintain a balanced diet, regular exercise, and annual check-ups."))

    tip_cols = st.columns(min(len(tips), 3))
    for i, (title, body) in enumerate(tips):
        with tip_cols[i % 3]:
            st.markdown(f"""
            <div class="tip-card">
                <div class="tip-title">{title}</div>
                <div class="tip-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.balloons()

# Footer
st.markdown("""
<br><br>
<div style="text-align:center; color:#cbd5e1; font-size:0.72rem; letter-spacing:1px;">
    MEDIPREDICT AI &nbsp;&middot;&nbsp; FOR EDUCATIONAL USE ONLY &nbsp;&middot;&nbsp; NOT A SUBSTITUTE FOR MEDICAL ADVICE
</div>
""", unsafe_allow_html=True)
