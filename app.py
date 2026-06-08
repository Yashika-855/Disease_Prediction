import streamlit as st
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

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
# Gorgeous CSS Theme
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

/* ---- Background ---- */
.stApp {
    background: linear-gradient(135deg, #020817 0%, #0c1445 40%, #0a1628 100%);
    min-height: 100vh;
}

/* ---- Remove default padding ---- */
.block-container { padding: 2rem 3rem 3rem 3rem !important; max-width: 1300px !important; }

/* ---- Headings ---- */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; letter-spacing: -1px; }
h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }

/* ---- Hero Banner ---- */
.hero-banner {
    background: linear-gradient(120deg, #0f3460 0%, #16213e 50%, #1a1a2e 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 24px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,0.07) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.75rem;
    font-weight: 300;
    letter-spacing: 0.3px;
}
.hero-badges {
    display: flex;
    gap: 0.7rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    border-radius: 100px;
    padding: 0.3rem 1rem;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}
.badge.purple { background: rgba(129,140,248,0.12); border-color: rgba(129,140,248,0.3); color: #818cf8; }
.badge.pink   { background: rgba(192,132,252,0.12); border-color: rgba(192,132,252,0.3); color: #c084fc; }

/* ---- Section Cards ---- */
.section-card {
    background: linear-gradient(145deg, rgba(15,23,42,0.9), rgba(30,41,59,0.6));
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ---- Sliders ---- */
.stSlider > div > div > div > div { background: linear-gradient(90deg, #06b6d4, #6366f1) !important; }
.stSlider [data-baseweb="thumb"] { background: white !important; border: 3px solid #38bdf8 !important; }
.stSlider label { color: #cbd5e1 !important; font-weight: 500; font-size: 0.92rem; }

/* ---- Selectboxes ---- */
.stSelectbox label { color: #cbd5e1 !important; font-weight: 500; font-size: 0.92rem; }
.stSelectbox > div > div {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #06b6d4 0%, #6366f1 50%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    height: 3.5em !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.55) !important;
}

/* Reset button variant */
.reset-btn > button {
    background: rgba(239,68,68,0.15) !important;
    border: 1px solid rgba(239,68,68,0.4) !important;
    color: #f87171 !important;
    box-shadow: none !important;
}
.reset-btn > button:hover { background: rgba(239,68,68,0.25) !important; box-shadow: none !important; }

/* ---- Metric Cards ---- */
.metric-card {
    background: linear-gradient(145deg, rgba(6,182,212,0.12), rgba(99,102,241,0.08));
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #06b6d4, #6366f1);
    border-radius: 18px 18px 0 0;
}
.metric-label { color: #64748b; font-size: 0.8rem; letter-spacing: 1.5px; text-transform: uppercase; font-weight: 600; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; color: #38bdf8; line-height: 1.1; margin: 0.3rem 0; }
.metric-sub   { color: #475569; font-size: 0.8rem; }

.risk-high {
    background: linear-gradient(145deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border-color: rgba(239,68,68,0.4);
}
.risk-high::before { background: linear-gradient(90deg, #ef4444, #dc2626); }
.risk-high .metric-value { color: #f87171; }

.risk-low {
    background: linear-gradient(145deg, rgba(34,197,94,0.15), rgba(22,163,74,0.08));
    border-color: rgba(34,197,94,0.4);
}
.risk-low::before { background: linear-gradient(90deg, #22c55e, #16a34a); }
.risk-low .metric-value { color: #4ade80; }

/* ---- Divider ---- */
hr { border-color: rgba(56,189,248,0.12) !important; }

/* ---- General text ---- */
p, label { color: #94a3b8 !important; }

/* ---- Info box ---- */
.info-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.info-pill {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 12px;
    padding: 0.9rem 1.3rem;
    flex: 1;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.info-pill .icon { font-size: 1.5rem; }
.info-pill .text { font-size: 0.82rem; color: #64748b; line-height: 1.4; }
.info-pill .text strong { color: #94a3b8; display: block; font-size: 0.87rem; margin-bottom: 2px; }

/* ---- Step indicator ---- */
.step-row { display: flex; gap: 0.5rem; margin-bottom: 1.8rem; align-items: center; }
.step-dot {
    width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, #06b6d4, #6366f1);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.85rem; color: white;
    flex-shrink: 0;
}
.step-line { flex: 1; height: 1px; background: rgba(56,189,248,0.15); }
.step-label { color: #475569; font-size: 0.78rem; letter-spacing: 1px; text-transform: uppercase; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load Models
# -----------------------------
def load_model(file):
    if not os.path.exists(file):
        st.error(f"❌ Missing file: `{file}` — please ensure all model files are present.")
        st.stop()
    return pickle.load(open(file, "rb"))

reg    = load_model("reg_model.pkl")
clf    = load_model("clf_model.pkl")
scaler = load_model("scaler.pkl")


# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">MediPredict AI</p>
    <p class="hero-sub">Intelligent health cost estimation & disease risk assessment powered by machine learning</p>
    <div class="hero-badges">
        <span class="badge">🤖 ML-Powered</span>
        <span class="badge purple">📊 Real-time Analysis</span>
        <span class="badge pink">🔒 Privacy First</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Info Pills
st.markdown("""
<div class="info-row">
    <div class="info-pill">
        <span class="icon">💊</span>
        <div class="text"><strong>Medical Cost Prediction</strong>Estimate annual healthcare expenditure based on your profile</div>
    </div>
    <div class="info-pill">
        <span class="icon">🧬</span>
        <div class="text"><strong>Disease Risk Scoring</strong>Assess likelihood of developing conditions using clinical indicators</div>
    </div>
    <div class="info-pill">
        <span class="icon">⚡</span>
        <div class="text"><strong>Instant Results</strong>Get predictions in seconds with detailed visual breakdown</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Step 1 — Personal Info
# -----------------------------
st.markdown("""
<div class="step-row">
    <div class="step-dot">1</div>
    <div class="step-line"></div>
    <span class="step-label">Personal Information</span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age      = st.slider("🎂 Age", 0, 100, 25)
with col2:
    bmi      = st.slider("⚖️ BMI", 10.0, 50.0, 22.0, step=0.1)
with col3:
    children = st.slider("👨‍👩‍👧 Children", 0, 5, 0)

col4, col5, col6 = st.columns(3)
with col4:
    gender   = st.selectbox("👤 Gender",   ["Male", "Female"])
with col5:
    smoker   = st.selectbox("🚬 Smoker",   ["No", "Yes"])
with col6:
    activity = st.selectbox("🏃 Activity Level", ["Low", "Medium", "High"])

st.markdown("---")

# -----------------------------
# Step 2 — Coverage & Location
# -----------------------------
st.markdown("""
<div class="step-row">
    <div class="step-dot">2</div>
    <div class="step-line"></div>
    <span class="step-label">Coverage & Location</span>
</div>
""", unsafe_allow_html=True)

col7, col8 = st.columns(2)
with col7:
    insurance = st.selectbox("🛡️ Insurance Plan", ["Basic", "Premium"])
with col8:
    city      = st.selectbox("🏙️ City Type",       ["Urban", "Semi-Urban", "Rural"])

st.markdown("---")

# -----------------------------
# Step 3 — Medical History
# -----------------------------
st.markdown("""
<div class="step-row">
    <div class="step-dot">3</div>
    <div class="step-line"></div>
    <span class="step-label">Medical History</span>
</div>
""", unsafe_allow_html=True)

col9, col10, col11, col12 = st.columns(4)
with col9:
    diabetes      = st.selectbox("🩸 Diabetes",      [0, 1])
with col10:
    hypertension  = st.selectbox("💓 Hypertension",  [0, 1])
with col11:
    heart         = st.selectbox("❤️ Heart Disease", [0, 1])
with col12:
    asthma        = st.selectbox("🫁 Asthma",        [0, 1])

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
btn_col1, btn_col2, btn_col3 = st.columns([3, 1, 1])

with btn_col1:
    predict_clicked = st.button("🚀 Run Full Prediction Analysis", use_container_width=True)
with btn_col2:
    cost_only = st.button("💰 Cost Only", use_container_width=True)
with btn_col3:
    with st.container():
        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        reset = st.button("🔄 Reset", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction Logic
# -----------------------------
def get_prediction():
    input_data = np.array([[
        age, gender_enc, bmi, children, smoker_enc,
        activity_enc, insurance_enc, city_enc,
        diabetes, hypertension, heart, asthma
    ]])
    input_scaled = scaler.transform(input_data)
    cost    = reg.predict(input_scaled)[0]
    disease = clf.predict(input_scaled)[0]
    return cost, disease, input_scaled

if predict_clicked or cost_only:
    cost, disease, input_scaled = get_prediction()

    st.markdown("---")
    st.markdown("""
    <div class="step-row">
        <div class="step-dot">✓</div>
        <div class="step-line"></div>
        <span class="step-label">Prediction Results</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Metric Cards ---
    if predict_clicked:
        m1, m2 = st.columns(2)
    else:
        m1, _ = st.columns([1, 1])

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💰 Estimated Annual Cost</div>
            <div class="metric-value">₹ {cost:,.0f}</div>
            <div class="metric-sub">Based on your health & demographic profile</div>
        </div>
        """, unsafe_allow_html=True)

    if predict_clicked:
        with m2:
            risk_class = "risk-high" if disease == 1 else "risk-low"
            risk_icon  = "⚠️" if disease == 1 else "✅"
            risk_label = "HIGH RISK" if disease == 1 else "LOW RISK"
            risk_sub   = "Consult a specialist soon" if disease == 1 else "Keep up the healthy habits!"
            st.markdown(f"""
            <div class="metric-card {risk_class}">
                <div class="metric-label">{risk_icon} Disease Risk Level</div>
                <div class="metric-value">{risk_label}</div>
                <div class="metric-sub">{risk_sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # --- Charts ---
    st.markdown("<br>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    # Chart 1: Feature Overview Radar-style Bar
    with chart_col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')

        features = ["Age", "BMI", "Children", "Activity"]
        raw_vals = [age, bmi, children, activity_enc]
        colors   = ["#38bdf8", "#818cf8", "#c084fc", "#34d399"]

        bars = ax.bar(features, raw_vals, color=colors, width=0.55, zorder=3,
                      edgecolor='none', linewidth=0)
        for bar, val in zip(bars, raw_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(round(val, 1)), ha='center', va='bottom',
                    color='white', fontsize=10, fontweight='bold')

        ax.set_title("Patient Profile Snapshot", color='#94a3b8', fontsize=11,
                     pad=12, fontweight='600')
        ax.tick_params(colors='#64748b', labelsize=9)
        ax.spines[:].set_visible(False)
        ax.set_ylim(0, max(raw_vals) * 1.25 + 2)
        ax.yaxis.set_visible(False)
        ax.grid(axis='y', color='#1e293b', linestyle='--', alpha=0.5, zorder=0)
        fig.tight_layout()
        st.pyplot(fig)

    # Chart 2: Risk Factor Heatmap
    with chart_col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#0f172a')
        ax2.set_facecolor('#0f172a')

        conditions  = ["Diabetes", "Hypertension", "Heart", "Asthma", "Smoker"]
        cond_values = [diabetes, hypertension, heart, asthma, smoker_enc]
        bar_colors  = ["#ef4444" if v else "#22c55e" for v in cond_values]

        hbars = ax2.barh(conditions, cond_values, color=bar_colors, height=0.45,
                         zorder=3, edgecolor='none')
        for bar, val in zip(hbars, cond_values):
            label = "Present" if val else "Absent"
            ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                     label, va='center', color='white', fontsize=9, fontweight='600')

        ax2.set_title("Medical Condition Flags", color='#94a3b8', fontsize=11,
                      pad=12, fontweight='600')
        ax2.set_xlim(0, 1.4)
        ax2.tick_params(colors='#64748b', labelsize=9)
        ax2.spines[:].set_visible(False)
        ax2.xaxis.set_visible(False)

        legend_elements = [
            mpatches.Patch(color='#ef4444', label='Present'),
            mpatches.Patch(color='#22c55e', label='Absent')
        ]
        ax2.legend(handles=legend_elements, loc='lower right',
                   facecolor='#1e293b', edgecolor='none',
                   labelcolor='#94a3b8', fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)

    # --- Cost Breakdown Donut ---
    if predict_clicked:
        st.markdown("<br>", unsafe_allow_html=True)
        d_col1, d_col2 = st.columns([1, 1])

        with d_col1:
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            fig3.patch.set_facecolor('#0f172a')
            ax3.set_facecolor('#0f172a')

            # Simulated cost breakdown
            base         = age * 100
            bmi_contrib  = bmi * 80
            smoker_cost  = 4000 if smoker_enc else 0
            disease_cost = (diabetes + hypertension + heart + asthma) * 1200
            remainder    = max(cost - base - bmi_contrib - smoker_cost - disease_cost, 500)

            sizes  = [base, bmi_contrib, smoker_cost, disease_cost, remainder]
            labels = ["Age Factor", "BMI Factor", "Smoking", "Conditions", "Base Premium"]
            colors = ["#38bdf8", "#818cf8", "#ef4444", "#f97316", "#22c55e"]
            explode = [0.04]*5

            wedges, texts, autotexts = ax3.pie(
                sizes, labels=None, autopct='%1.0f%%', startangle=140,
                colors=colors, explode=explode,
                wedgeprops=dict(width=0.55, edgecolor='#0f172a', linewidth=2),
                pctdistance=0.78,
                textprops=dict(color='white', fontsize=8, fontweight='bold')
            )
            ax3.set_title("Cost Breakdown Estimate", color='#94a3b8', fontsize=11,
                          pad=12, fontweight='600')
            legend = ax3.legend(wedges, labels, loc="lower center",
                                bbox_to_anchor=(0.5, -0.18), ncol=3,
                                facecolor='#1e293b', edgecolor='none',
                                labelcolor='#94a3b8', fontsize=7.5)
            fig3.tight_layout()
            st.pyplot(fig3)

        with d_col2:
            # Insurance vs Actual Cost gauge-style
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            fig4.patch.set_facecolor('#0f172a')
            ax4.set_facecolor('#0f172a')

            coverage_pct = 60 if insurance_enc == 0 else 85
            out_of_pocket = cost * (1 - coverage_pct/100)

            cats   = ["Total Cost", "Covered", "Out of Pocket"]
            vals   = [cost, cost * coverage_pct/100, out_of_pocket]
            cols   = ["#38bdf8", "#22c55e", "#f97316"]

            bars2 = ax4.bar(cats, vals, color=cols, width=0.5, zorder=3, edgecolor='none')
            for bar, val in zip(bars2, vals):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + cost*0.01,
                         f"₹{val:,.0f}", ha='center', va='bottom',
                         color='white', fontsize=8, fontweight='bold')

            ax4.set_title(f"Coverage Breakdown ({coverage_pct}% covered)",
                          color='#94a3b8', fontsize=11, pad=12, fontweight='600')
            ax4.tick_params(colors='#64748b', labelsize=9)
            ax4.spines[:].set_visible(False)
            ax4.yaxis.set_visible(False)
            ax4.set_ylim(0, max(vals) * 1.25)
            fig4.tight_layout()
            st.pyplot(fig4)

    # --- Recommendations ---
    st.markdown("---")
    st.markdown("""
    <div class="step-row">
        <div class="step-dot">💡</div>
        <div class="step-line"></div>
        <span class="step-label">Personalised Recommendations</span>
    </div>
    """, unsafe_allow_html=True)

    tips = []
    if smoker_enc: tips.append(("🚭", "Quit Smoking", "Smoking significantly raises costs & disease risk. Seek cessation support."))
    if bmi > 30:   tips.append(("🥗", "Manage BMI", f"Your BMI of {bmi:.1f} is above healthy range. A dietitian can help."))
    if age > 50:   tips.append(("🩺", "Annual Check-ups", "Regular screenings are critical after 50. Schedule today."))
    if diabetes:   tips.append(("💉", "Diabetes Management", "Monitor blood sugar and follow your medication plan closely."))
    if heart:      tips.append(("❤️", "Cardiac Care", "Maintain a heart-healthy lifestyle — low sodium, moderate exercise."))
    if not tips:   tips.append(("🌟", "Great Profile!", "You appear to be in good health. Keep it up!"))

    tip_cols = st.columns(min(len(tips), 3))
    for i, (icon, title, body) in enumerate(tips):
        with tip_cols[i % 3]:
            st.markdown(f"""
            <div class="section-card" style="text-align:center; padding:1.5rem;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-family:'Syne',sans-serif; font-weight:700; color:#e2e8f0; font-size:0.95rem; margin-bottom:0.4rem;">{title}</div>
                <div style="color:#64748b; font-size:0.82rem; line-height:1.5;">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.balloons()

# Footer
st.markdown("""
<br><br>
<div style="text-align:center; color:#1e293b; font-size:0.75rem; letter-spacing:1px;">
    MEDIPREDICT AI &nbsp;·&nbsp; FOR EDUCATIONAL USE ONLY &nbsp;·&nbsp; NOT A MEDICAL DIAGNOSIS
</div>
""", unsafe_allow_html=True)
