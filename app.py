import streamlit as st
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --ink:    #0b0f1a;
    --paper:  #111827;
    --card:   #161d2e;
    --border: rgba(255,255,255,0.07);
    --gold:   #d4a943;
    --teal:   #2ec4b6;
    --rose:   #e05c7a;
    --muted:  #8892a4;
    --text:   #e8edf5;
}

* { font-family: 'Outfit', sans-serif; box-sizing: border-box; }

.stApp {
    background: var(--ink);
    background-image:
        radial-gradient(ellipse 80% 60% at 10% -10%, rgba(46,196,182,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 110%, rgba(212,169,67,0.06) 0%, transparent 60%);
    min-height: 100vh;
}

.block-container { padding: 2rem 2rem 4rem 2rem !important; max-width: 760px !important; }

/* ── Progress Bar ── */
.prog-wrap {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 2.5rem;
}
.prog-step {
    width: 28px; height: 28px; border-radius: 50%;
    border: 2px solid rgba(255,255,255,0.1);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 600; color: var(--muted);
    background: var(--card);
    transition: all .3s;
    flex-shrink: 0;
}
.prog-step.done  { background: var(--teal); border-color: var(--teal); color: #0b0f1a; }
.prog-step.active { background: var(--gold); border-color: var(--gold); color: #0b0f1a; }
.prog-line { flex: 1; height: 1px; background: rgba(255,255,255,0.08); }
.prog-line.done { background: var(--teal); }

/* ── Slide Card ── */
.slide-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 2.8rem 3rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
    position: relative; overflow: hidden;
}
.slide-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--teal), var(--gold));
}

.slide-label {
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 3px; text-transform: uppercase;
    color: var(--teal); margin-bottom: 0.6rem;
}
.slide-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem; font-weight: 900;
    color: var(--text); line-height: 1.2;
    margin: 0 0 0.4rem 0;
}
.slide-title em {
    font-style: italic; color: var(--gold);
}
.slide-desc {
    color: var(--muted); font-size: 0.88rem;
    margin-bottom: 2rem; font-weight: 300;
}

/* ── Inputs ── */
.stSlider label, .stSelectbox label, .stRadio label {
    color: var(--text) !important; font-weight: 500 !important; font-size: 0.9rem !important;
}
.stSlider > div > div > div > div { background: var(--teal) !important; }
.stSlider [data-baseweb="thumb"] {
    background: var(--gold) !important;
    border: 2px solid #0b0f1a !important;
    width: 20px !important; height: 20px !important;
    box-shadow: 0 0 12px rgba(212,169,67,0.5) !important;
}
.stSelectbox > div > div {
    background: #1c2539 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stRadio > div { flex-direction: row !important; gap: 1rem !important; flex-wrap: wrap; }
.stRadio [data-baseweb="radio"] > div:first-child {
    border-color: var(--teal) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 12px !important;
    height: 3.2em !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all .25s ease !important;
    letter-spacing: 0.3px !important;
}

/* Primary — gold */
div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button,
.primary-btn .stButton > button {
    background: linear-gradient(135deg, #d4a943, #c4922a) !important;
    color: #0b0f1a !important;
    box-shadow: 0 4px 18px rgba(212,169,67,0.35) !important;
}
/* Back — ghost */
.back-btn .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: var(--muted) !important;
}
.back-btn .stButton > button:hover {
    border-color: rgba(255,255,255,0.3) !important;
    color: var(--text) !important;
}

/* ── Metric card ── */
.res-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    position: relative; overflow: hidden;
    margin-bottom: 1rem;
}
.res-card::before {
    content: ''; position: absolute;
    top:0; left:0; right:0; height:3px;
}
.res-card.teal::before { background: var(--teal); }
.res-card.rose::before { background: var(--rose); }
.res-card.green::before { background: #4caf84; }

.res-lbl { font-size: 0.68rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; color: var(--muted); }
.res-val {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem; font-weight: 900;
    color: var(--text); line-height: 1.1; margin: 0.3rem 0;
}
.res-sub { font-size: 0.8rem; color: var(--muted); }

/* ── Tip cards ── */
.tip-card {
    background: #1a2235;
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.8rem;
}
.tip-title { font-weight: 600; color: var(--text); font-size: 0.9rem; margin-bottom: 0.25rem; }
.tip-body  { font-size: 0.8rem; color: var(--muted); line-height: 1.55; }

/* ── Welcome hero ── */
.hero-wrap {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero-eyebrow {
    font-size: 0.7rem; letter-spacing: 4px; text-transform: uppercase;
    color: var(--teal); font-weight: 600; margin-bottom: 1rem;
}
.hero-h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.6rem; font-weight: 900;
    color: var(--text); line-height: 1.1;
    margin: 0 0 1rem;
}
.hero-h1 span { color: var(--gold); font-style: italic; }
.hero-para {
    color: var(--muted); font-size: 1rem; font-weight: 300;
    max-width: 520px; margin: 0 auto 2.5rem;
    line-height: 1.7;
}
.hero-stats {
    display: flex; justify-content: center; gap: 2.5rem; margin-bottom: 2.5rem;
}
.hero-stat { text-align: center; }
.hero-stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem; font-weight: 900; color: var(--gold);
}
.hero-stat-lbl { font-size: 0.72rem; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── General ── */
p, label { color: var(--muted) !important; }

/* ── Result header ── */
.result-header {
    text-align: center; padding: 1.5rem 0 1rem;
}
.result-header h2 {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem; font-weight: 900; color: var(--text);
}
.result-header p { font-size: 0.9rem; color: var(--muted); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────
def load_model(file):
    if not os.path.exists(file):
        st.error(f"Missing: `{file}` — place all .pkl files next to this script.")
        st.stop()
    return pickle.load(open(file, "rb"))

reg    = load_model("reg_model.pkl")
clf    = load_model("clf_model.pkl")
scaler = load_model("scaler.pkl")


# ─────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────
TOTAL_SLIDES = 7   # 0=welcome, 1-6=input slides, 7=results

defaults = dict(
    slide=0,
    age=30, bmi=24.0, children=0,
    gender="Male", smoker="No", activity="Medium",
    insurance="Basic", city="Urban",
    diabetes=0, hypertension=0, heart=0, asthma=0,
    predicted=False, cost=0.0, disease=0
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

s = st.session_state


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def go(slide_n): s.slide = slide_n
def back():      s.slide = max(0, s.slide - 1)
def nxt():       s.slide = s.slide + 1

def progress_bar(current, total):
    labels = ["You", "Body", "Lifestyle", "Coverage", "History", "Predict"]
    html = '<div class="prog-wrap">'
    for i in range(1, total + 1):
        line_cls = "done" if i < current else ""
        if i > 1:
            html += f'<div class="prog-line {line_cls}"></div>'
        if i < current:
            dot_cls = "done"
        elif i == current:
            dot_cls = "active"
        else:
            dot_cls = ""
        lbl = labels[i-1] if i <= len(labels) else str(i)
        html += f'<div class="prog-step {dot_cls}">{i}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def slide_header(step_label, title, desc):
    st.markdown(f"""
    <div class="slide-label">{step_label}</div>
    <p class="slide-title">{title}</p>
    <p class="slide-desc">{desc}</p>
    """, unsafe_allow_html=True)

def nav_buttons(back_label="Back", next_label="Continue →", show_back=True):
    if show_back:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<div class="back-btn">', unsafe_allow_html=True)
            if st.button(back_label, use_container_width=True, key=f"back_{s.slide}"):
                back()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            if st.button(next_label, use_container_width=True, key=f"next_{s.slide}"):
                nxt()
                st.rerun()
    else:
        if st.button(next_label, use_container_width=True, key=f"next_{s.slide}"):
            nxt()
            st.rerun()


# ─────────────────────────────────────────
# SLIDE 0 — Welcome
# ─────────────────────────────────────────
if s.slide == 0:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">AI-Powered Health Intelligence</div>
        <h1 class="hero-h1">Medi<span>Predict</span> AI</h1>
        <p class="hero-para">
            A machine-learning platform that estimates your annual medical cost
            and evaluates disease risk — based on your personal health profile.
            Answer 6 quick steps and get a detailed analysis instantly.
        </p>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-val">12</div>
                <div class="hero-stat-lbl">Parameters</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-val">6</div>
                <div class="hero-stat-lbl">Quick Steps</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-val">2</div>
                <div class="hero-stat-lbl">Predictions</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        if st.button("Begin Assessment →", use_container_width=True, key="start"):
            go(1); st.rerun()

    st.markdown("""
    <p style="text-align:center; font-size:0.72rem; margin-top:1.5rem; color:#3a4456; letter-spacing:1px;">
    FOR EDUCATIONAL USE ONLY · NOT A SUBSTITUTE FOR MEDICAL ADVICE
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# SLIDE 1 — Age & Gender
# ─────────────────────────────────────────
elif s.slide == 1:
    progress_bar(1, 6)
    st.markdown('<div class="slide-card">', unsafe_allow_html=True)
    slide_header("Step 1 of 6", "Who <em>are</em> you?", "Let's start with the basics — your age and gender.")

    s.age = st.slider("Age (years)", 1, 100, s.age)
    st.markdown("<br>", unsafe_allow_html=True)
    s.gender = st.radio("Gender", ["Male", "Female"], index=0 if s.gender == "Male" else 1, horizontal=True)

    st.markdown('</div>', unsafe_allow_html=True)
    nav_buttons(show_back=False, next_label="Continue →")


# ─────────────────────────────────────────
# SLIDE 2 — Body Metrics
# ─────────────────────────────────────────
elif s.slide == 2:
    progress_bar(2, 6)
    st.markdown('<div class="slide-card">', unsafe_allow_html=True)
    slide_header("Step 2 of 6", "Your <em>body</em> metrics.", "BMI and number of dependents affect cost significantly.")

    s.bmi      = st.slider("Body Mass Index (BMI)", 10.0, 50.0, float(s.bmi), step=0.1)
    st.markdown("<br>", unsafe_allow_html=True)
    s.children = st.slider("Number of Children / Dependents", 0, 5, s.children)

    st.markdown('</div>', unsafe_allow_html=True)
    nav_buttons()


# ─────────────────────────────────────────
# SLIDE 3 — Lifestyle
# ─────────────────────────────────────────
elif s.slide == 3:
    progress_bar(3, 6)
    st.markdown('<div class="slide-card">', unsafe_allow_html=True)
    slide_header("Step 3 of 6", "Your <em>lifestyle</em>.", "Smoking and activity level are strong cost predictors.")

    s.smoker   = st.radio("Do you smoke?",        ["No", "Yes"], index=["No","Yes"].index(s.smoker), horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)
    s.activity = st.radio("Physical activity level", ["Low", "Medium", "High"], index=["Low","Medium","High"].index(s.activity), horizontal=True)

    st.markdown('</div>', unsafe_allow_html=True)
    nav_buttons()


# ─────────────────────────────────────────
# SLIDE 4 — Coverage & Location
# ─────────────────────────────────────────
elif s.slide == 4:
    progress_bar(4, 6)
    st.markdown('<div class="slide-card">', unsafe_allow_html=True)
    slide_header("Step 4 of 6", "Coverage &amp; <em>location</em>.", "Your insurance tier and region shape your cost estimate.")

    s.insurance = st.radio("Insurance Plan", ["Basic", "Premium"], index=["Basic","Premium"].index(s.insurance), horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)
    s.city = st.radio("City Type", ["Urban", "Semi-Urban", "Rural"], index=["Urban","Semi-Urban","Rural"].index(s.city), horizontal=True)

    st.markdown('</div>', unsafe_allow_html=True)
    nav_buttons()


# ─────────────────────────────────────────
# SLIDE 5 — Medical History
# ─────────────────────────────────────────
elif s.slide == 5:
    progress_bar(5, 6)
    st.markdown('<div class="slide-card">', unsafe_allow_html=True)
    slide_header("Step 5 of 6", "Medical <em>history</em>.", "Existing conditions help calibrate the disease risk model.")

    c1, c2 = st.columns(2)
    with c1:
        s.diabetes     = st.radio("Diabetes",      ["No","Yes"], index=s.diabetes, horizontal=True, key="r_db")
        st.markdown("<br>", unsafe_allow_html=True)
        s.heart        = st.radio("Heart Disease", ["No","Yes"], index=s.heart,    horizontal=True, key="r_hd")
    with c2:
        s.hypertension = st.radio("Hypertension",  ["No","Yes"], index=s.hypertension, horizontal=True, key="r_ht")
        st.markdown("<br>", unsafe_allow_html=True)
        s.asthma       = st.radio("Asthma",        ["No","Yes"], index=s.asthma,  horizontal=True, key="r_as")

    # normalise radio to 0/1
    for attr in ["diabetes","hypertension","heart","asthma"]:
        val = getattr(s, attr)
        if isinstance(val, str):
            setattr(s, attr, 1 if val == "Yes" else 0)

    st.markdown('</div>', unsafe_allow_html=True)
    nav_buttons()


# ─────────────────────────────────────────
# SLIDE 6 — Confirm & Predict
# ─────────────────────────────────────────
elif s.slide == 6:
    progress_bar(6, 6)
    st.markdown('<div class="slide-card">', unsafe_allow_html=True)
    slide_header("Step 6 of 6", "Ready to <em>predict</em>?", "Here's a summary of your inputs. Hit Predict to run the models.")

    def yn(v): return "Yes" if v == 1 else "No"

    rows = [
        ("Age",          s.age),
        ("Gender",       s.gender),
        ("BMI",          f"{s.bmi:.1f}"),
        ("Children",     s.children),
        ("Smoker",       s.smoker),
        ("Activity",     s.activity),
        ("Insurance",    s.insurance),
        ("City",         s.city),
        ("Diabetes",     yn(s.diabetes)),
        ("Hypertension", yn(s.hypertension)),
        ("Heart Disease",yn(s.heart)),
        ("Asthma",       yn(s.asthma)),
    ]

    col_a, col_b = st.columns(2)
    for i, (lbl, val) in enumerate(rows):
        col = col_a if i % 2 == 0 else col_b
        col.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:0.55rem 0;border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="color:#8892a4;font-size:0.82rem;">{lbl}</span>
            <span style="color:#e8edf5;font-weight:600;font-size:0.88rem;">{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns([1, 2, 1])
    with b1:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("Back", use_container_width=True, key="back_6"):
            back(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with b2:
        if st.button("Run Prediction Now", use_container_width=True, key="predict_btn"):
            # encode
            g_enc  = 1 if s.gender == "Male" else 0
            sm_enc = 1 if s.smoker == "Yes" else 0
            ac_enc = {"Low":0,"Medium":1,"High":2}[s.activity]
            in_enc = {"Basic":0,"Premium":1}[s.insurance]
            ci_enc = {"Urban":0,"Semi-Urban":1,"Rural":2}[s.city]

            inp = np.array([[s.age, g_enc, s.bmi, s.children, sm_enc,
                             ac_enc, in_enc, ci_enc,
                             s.diabetes, s.hypertension, s.heart, s.asthma]])
            scaled      = scaler.transform(inp)
            s.cost      = float(reg.predict(scaled)[0])
            s.disease   = int(clf.predict(scaled)[0])
            s.predicted = True
            go(7); st.rerun()
    with b3:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("Restart", use_container_width=True, key="restart_6"):
            for k, v in defaults.items(): s[k] = v
            go(0); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# SLIDE 7 — Results
# ─────────────────────────────────────────
elif s.slide == 7 and s.predicted:

    cost    = s.cost
    disease = s.disease
    smoker_enc    = 1 if s.smoker   == "Yes"    else 0
    activity_enc  = {"Low":0,"Medium":1,"High":2}[s.activity]
    insurance_enc = {"Basic":0,"Premium":1}[s.insurance]

    st.markdown("""
    <div class="result-header">
        <h2>Your Health Assessment</h2>
        <p>Scroll down for the full breakdown, charts, and personalised recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Cards ──
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"""
        <div class="res-card teal">
            <div class="res-lbl">Estimated Annual Cost</div>
            <div class="res-val">&#8377; {cost:,.0f}</div>
            <div class="res-sub">Based on your full health profile</div>
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        rc   = "rose" if disease else "green"
        rlbl = "High Risk" if disease else "Low Risk"
        rsub = "Consult a specialist soon." if disease else "Your profile looks healthy."
        st.markdown(f"""
        <div class="res-card {rc}">
            <div class="res-lbl">Disease Risk</div>
            <div class="res-val">{rlbl}</div>
            <div class="res-sub">{rsub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart colours ──
    BG   = "#161d2e"
    AX   = "#1a2235"
    TEXT = "#8892a4"
    GOLD = "#d4a943"
    TEAL = "#2ec4b6"
    ROSE = "#e05c7a"

    ch1, ch2 = st.columns(2)

    # Chart 1 — Profile bar
    with ch1:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(AX)

        feats = ["Age", "BMI", "Children", "Activity"]
        vals  = [s.age, s.bmi, s.children, activity_enc]
        clrs  = [TEAL, GOLD, "#a78bfa", "#f472b6"]

        bars = ax.bar(feats, vals, color=clrs, width=0.5, edgecolor='none', zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    str(round(v,1)), ha='center', va='bottom',
                    color='#e8edf5', fontsize=9, fontweight='600')

        ax.set_title("Profile Overview", color=TEXT, fontsize=10, pad=10, fontweight='600')
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[:].set_visible(False)
        ax.set_ylim(0, max(vals)*1.3+2)
        ax.yaxis.set_visible(False)
        ax.grid(axis='y', color='#232e45', linewidth=0.8, zorder=0)
        fig.tight_layout()
        st.pyplot(fig)

    # Chart 2 — Condition flags
    with ch2:
        fig2, ax2 = plt.subplots(figsize=(5, 3.6))
        fig2.patch.set_facecolor(BG)
        ax2.set_facecolor(AX)

        conds  = ["Diabetes", "Hypertension", "Heart Disease", "Asthma", "Smoker"]
        cvals  = [s.diabetes, s.hypertension, s.heart, s.asthma, smoker_enc]
        bclrs  = [ROSE if v else TEAL for v in cvals]

        hbars = ax2.barh(conds, cvals, color=bclrs, height=0.38, edgecolor='none', zorder=3)
        for bar, v in zip(hbars, cvals):
            lbl = "Present" if v else "Absent"
            ax2.text(v+0.04, bar.get_y()+bar.get_height()/2,
                     lbl, va='center', color='#e8edf5', fontsize=8.5, fontweight='500')

        ax2.set_title("Medical Conditions", color=TEXT, fontsize=10, pad=10, fontweight='600')
        ax2.set_xlim(0, 1.5)
        ax2.tick_params(colors=TEXT, labelsize=8)
        ax2.spines[:].set_visible(False)
        ax2.xaxis.set_visible(False)
        leg = [mpatches.Patch(color=ROSE, label='Present'), mpatches.Patch(color=TEAL, label='Absent')]
        ax2.legend(handles=leg, facecolor='#232e45', edgecolor='none', labelcolor=TEXT, fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown("<br>", unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)

    # Chart 3 — Cost breakdown donut
    with dc1:
        fig3, ax3 = plt.subplots(figsize=(5, 3.6))
        fig3.patch.set_facecolor(BG)
        ax3.set_facecolor(BG)

        base_c  = s.age * 100
        bmi_c   = s.bmi * 80
        smk_c   = 4000 if smoker_enc else 0
        cond_c  = (s.diabetes+s.hypertension+s.heart+s.asthma) * 1200
        rem_c   = max(cost - base_c - bmi_c - smk_c - cond_c, 500)

        sizes  = [base_c, bmi_c, smk_c, cond_c, rem_c]
        labels = ["Age", "BMI", "Smoking", "Conditions", "Base"]
        clrs3  = [TEAL, GOLD, ROSE, "#f472b6", "#a78bfa"]

        wedges, _, auts = ax3.pie(
            sizes, labels=None, autopct='%1.0f%%', startangle=140,
            colors=clrs3, explode=[0.03]*5,
            wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
            pctdistance=0.78,
            textprops=dict(color='#e8edf5', fontsize=8, fontweight='600')
        )
        ax3.set_title("Cost Factors", color=TEXT, fontsize=10, pad=10, fontweight='600')
        ax3.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15),
                   ncol=3, facecolor='#1a2235', edgecolor='none', labelcolor=TEXT, fontsize=7.5)
        fig3.tight_layout()
        st.pyplot(fig3)

    # Chart 4 — Coverage
    with dc2:
        fig4, ax4 = plt.subplots(figsize=(5, 3.6))
        fig4.patch.set_facecolor(BG)
        ax4.set_facecolor(AX)

        cov_pct = 60 if insurance_enc == 0 else 85
        oop     = cost * (1 - cov_pct/100)

        cats4  = ["Total Cost", "Covered", "Out of Pocket"]
        vals4  = [cost, cost*cov_pct/100, oop]
        clrs4  = [TEAL, "#4caf84", ROSE]

        bars4 = ax4.bar(cats4, vals4, color=clrs4, width=0.45, edgecolor='none', zorder=3)
        for bar, v in zip(bars4, vals4):
            ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+cost*0.01,
                     f"₹{v:,.0f}", ha='center', va='bottom',
                     color='#e8edf5', fontsize=7.5, fontweight='600')

        ax4.set_title(f"Coverage  ({cov_pct}% insured)", color=TEXT, fontsize=10, pad=10, fontweight='600')
        ax4.tick_params(colors=TEXT, labelsize=8)
        ax4.spines[:].set_visible(False)
        ax4.yaxis.set_visible(False)
        ax4.set_ylim(0, max(vals4)*1.28)
        ax4.grid(axis='y', color='#232e45', linewidth=0.8, zorder=0)
        fig4.tight_layout()
        st.pyplot(fig4)

    # ── Recommendations ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.68rem;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#2ec4b6;margin-bottom:1rem;">
    Personalised Recommendations
    </p>
    """, unsafe_allow_html=True)

    tips = []
    if smoker_enc:      tips.append(("Smoking Cessation",     "Smoking is the single largest avoidable cost driver. Cessation programmes can reduce risk by up to 40%."))
    if s.bmi > 30:      tips.append(("Weight Management",     f"A BMI of {s.bmi:.1f} is above the healthy range. Structured diet and exercise can meaningfully lower costs."))
    if s.age > 50:      tips.append(("Routine Screenings",    "Annual screenings are strongly recommended for individuals above 50 to catch conditions early."))
    if s.diabetes:      tips.append(("Diabetes Management",   "Consistent blood glucose monitoring and adherence to your medication plan are essential."))
    if s.hypertension:  tips.append(("Blood Pressure Control","Reduce sodium, manage stress, and follow your physician's treatment plan closely."))
    if s.heart:         tips.append(("Cardiac Care",          "A low-sodium, heart-healthy diet and moderate physical activity as advised by your doctor."))
    if s.asthma:        tips.append(("Respiratory Care",      "Keep rescue medication accessible at all times and identify and avoid environmental triggers."))
    if not tips:        tips.append(("Excellent Profile",     "Your health indicators are in a favourable range. Maintain a balanced diet, exercise regularly, and schedule annual check-ups."))

    for title, body in tips:
        st.markdown(f"""
        <div class="tip-card">
            <div class="tip-title">{title}</div>
            <div class="tip-body">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_x, col_y, col_z = st.columns([1,2,1])
    with col_y:
        if st.button("Start New Assessment", use_container_width=True, key="restart_res"):
            for k, v in defaults.items(): s[k] = v
            go(0); st.rerun()

    st.balloons()

    st.markdown("""
    <p style="text-align:center;font-size:0.7rem;color:#232e45;letter-spacing:1px;margin-top:2rem;">
    MEDIPREDICT AI · EDUCATIONAL USE ONLY · NOT A SUBSTITUTE FOR MEDICAL ADVICE
    </p>
    """, unsafe_allow_html=True)

else:
    go(0); st.rerun()
