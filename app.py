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
# CSS — Moderate, Readable, Formal
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,600&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #1e2a3a;
    --surface: #253447;
    --card:    #2c3e55;
    --border:  rgba(255,255,255,0.10);
    --accent:  #5ba3c9;
    --gold:    #e8c16a;
    --green:   #5ab49a;
    --rose:    #d9717a;
    --text:    #eef2f7;
    --sub:     #9fb3c8;
    --muted:   #6b8299;
}

* { font-family: 'DM Sans', sans-serif; box-sizing: border-box; }

.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(ellipse 70% 50% at 0% 0%,   rgba(91,163,201,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 100% 100%, rgba(90,180,154,0.06) 0%, transparent 55%);
}

.block-container { padding: 2.5rem 2.5rem 4rem !important; max-width: 720px !important; }

/* ── Progress ── */
.prog-wrap { display:flex; align-items:center; gap:8px; margin-bottom:2.2rem; }
.prog-step {
    width:30px; height:30px; border-radius:50%;
    border: 2px solid var(--border);
    display:flex; align-items:center; justify-content:center;
    font-size:0.72rem; font-weight:700; color:var(--muted);
    background:var(--card); flex-shrink:0;
}
.prog-step.done   { background:var(--green);  border-color:var(--green);  color:#1a2a24; }
.prog-step.active { background:var(--accent); border-color:var(--accent); color:#0f1e2b; }
.prog-line      { flex:1; height:1px; background:rgba(255,255,255,0.07); }
.prog-line.done { background:var(--green); }

/* ── Slide Card ── */
.slide-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.6rem 2.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    position:relative; overflow:hidden;
}
.slide-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, var(--accent), var(--green));
}

.slide-eyebrow {
    font-size:0.68rem; font-weight:600; letter-spacing:3px;
    text-transform:uppercase; color:var(--accent); margin-bottom:0.5rem;
}
.slide-title {
    font-family:'Cormorant Garamond', serif;
    font-size:2rem; font-weight:700; color:var(--text);
    line-height:1.2; margin:0 0 0.35rem;
}
.slide-title em { font-style:italic; color:var(--gold); }
.slide-desc { color:var(--sub); font-size:0.87rem; font-weight:300; margin-bottom:1.8rem; line-height:1.6; }

/* ── Inputs ── */
.stSlider label, .stSelectbox label, .stRadio label, .stRadio > label {
    color: var(--text) !important; font-weight:500 !important; font-size:0.9rem !important;
}
.stSlider > div > div > div > div { background: var(--accent) !important; }
.stSlider [data-baseweb="thumb"] {
    background: var(--gold) !important;
    border: 2px solid var(--bg) !important;
    width:18px !important; height:18px !important;
    box-shadow: 0 0 8px rgba(232,193,106,0.4) !important;
}

/* Radio buttons */
div[data-testid="stRadio"] > div { gap:0.6rem !important; flex-wrap:wrap; }
div[data-testid="stRadio"] > div > label {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.2rem !important;
    color: var(--sub) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    cursor: pointer;
    transition: all 0.2s;
}
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: rgba(91,163,201,0.18) !important;
    border-color: var(--accent) !important;
    color: var(--text) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    height: 3em !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all .2s ease !important;
    letter-spacing: 0.3px !important;
    width: 100% !important;
}
/* Primary */
.primary-btn .stButton > button {
    background: var(--accent) !important;
    color: #ffffff !important;
    box-shadow: 0 3px 14px rgba(91,163,201,0.35) !important;
}
.primary-btn .stButton > button:hover { background: #4a8fb5 !important; }

/* Ghost */
.ghost-btn .stButton > button {
    background: transparent !important;
    border: 1.5px solid rgba(255,255,255,0.15) !important;
    color: var(--sub) !important;
}
.ghost-btn .stButton > button:hover {
    border-color: rgba(255,255,255,0.3) !important;
    color: var(--text) !important;
}

/* Danger ghost */
.danger-btn .stButton > button {
    background: transparent !important;
    border: 1.5px solid rgba(217,113,122,0.3) !important;
    color: var(--rose) !important;
}

/* ── Result Metric Cards ── */
.res-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    position:relative; overflow:hidden;
    margin-bottom: 1rem;
}
.res-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
}
.res-card.blue::before  { background: var(--accent); }
.res-card.rose::before  { background: var(--rose);   }
.res-card.green::before { background: var(--green);  }

.res-lbl { font-size:0.65rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:var(--muted); margin-bottom:0.4rem; }
.res-val {
    font-family:'Cormorant Garamond', serif;
    font-size:2.2rem; font-weight:700; color:var(--text); line-height:1.15;
}
.res-sub { font-size:0.78rem; color:var(--sub); margin-top:0.25rem; }

/* ── Tip Cards ── */
.tip-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.75rem;
}
.tip-title { font-weight:600; color:var(--text); font-size:0.88rem; margin-bottom:0.2rem; }
.tip-body  { font-size:0.8rem; color:var(--sub); line-height:1.6; }

/* ── Hero ── */
.hero-wrap { text-align:center; padding:2.5rem 1rem 2rem; }
.hero-eyebrow { font-size:0.68rem; letter-spacing:4px; text-transform:uppercase; color:var(--accent); font-weight:600; margin-bottom:0.9rem; }
.hero-h1 {
    font-family:'Cormorant Garamond', serif;
    font-size:3.4rem; font-weight:700; color:var(--text);
    line-height:1.1; margin:0 0 1rem;
}
.hero-h1 span { color:var(--gold); font-style:italic; }
.hero-para { color:var(--sub); font-size:0.95rem; font-weight:300; max-width:500px; margin:0 auto 2.2rem; line-height:1.75; }
.hero-stats { display:flex; justify-content:center; gap:3rem; margin-bottom:2.2rem; }
.hero-stat-val { font-family:'Cormorant Garamond', serif; font-size:1.9rem; font-weight:700; color:var(--gold); }
.hero-stat-lbl { font-size:0.7rem; color:var(--muted); letter-spacing:1px; text-transform:uppercase; margin-top:2px; }

/* ── Summary rows ── */
.sum-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.05);
}
.sum-lbl { color:var(--muted); font-size:0.82rem; }
.sum-val { color:var(--text);  font-weight:600;   font-size:0.88rem; }

/* ── Result header ── */
.result-hdr { text-align:center; padding:1.2rem 0 1rem; }
.result-hdr h2 {
    font-family:'Cormorant Garamond', serif;
    font-size:2.2rem; font-weight:700; color:var(--text);
}
.result-hdr p { font-size:0.87rem; color:var(--sub); margin:0; }

/* ── Section label ── */
.sec-label {
    font-size:0.65rem; font-weight:600; letter-spacing:3px;
    text-transform:uppercase; color:var(--accent); margin:1.5rem 0 0.9rem;
}

hr { border-color: rgba(255,255,255,0.07) !important; margin:1.4rem 0 !important; }
p  { color: var(--sub) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────
def load_model(file):
    if not os.path.exists(file):
        st.error(f"Missing: `{file}` — place all .pkl files in the same folder as this script.")
        st.stop()
    return pickle.load(open(file, "rb"))

reg    = load_model("reg_model.pkl")
clf    = load_model("clf_model.pkl")
scaler = load_model("scaler.pkl")


# ─────────────────────────────────────────
# Session State
# ─────────────────────────────────────────
defaults = dict(
    slide=0,
    age=30, bmi=24.0, children=0,
    gender="Male", smoker="No", activity="Medium",
    insurance="Basic", city="Urban",
    diabetes="No", hypertension="No", heart="No", asthma="No",
    predicted=False, cost=0.0, disease=0
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

s = st.session_state


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def go(n):   s.slide = n
def nxt():   s.slide += 1
def back():  s.slide = max(0, s.slide - 1)
def yn(v):   return "Yes" if v == 1 else "No"

def progress_bar(current):
    labels = ["You", "Body", "Lifestyle", "Coverage", "History", "Predict"]
    html = '<div class="prog-wrap">'
    for i in range(1, 7):
        if i > 1:
            lc = "done" if i <= current else ""
            html += f'<div class="prog-line {lc}"></div>'
        if i < current:   dc = "done"
        elif i == current: dc = "active"
        else:              dc = ""
        html += f'<div class="prog-step {dc}">{i}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def slide_card_open(eyebrow, title, desc):
    st.markdown(f"""
    <div class="slide-card">
        <div class="slide-eyebrow">{eyebrow}</div>
        <p class="slide-title">{title}</p>
        <p class="slide-desc">{desc}</p>
    """, unsafe_allow_html=True)

def slide_card_close():
    st.markdown('</div>', unsafe_allow_html=True)

def btn_row(back_label="Back", next_label="Continue →", show_back=True, next_key="", back_key=""):
    st.markdown("<br>", unsafe_allow_html=True)
    if show_back:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
            if st.button(back_label, use_container_width=True, key=back_key or f"bk_{s.slide}"):
                back(); st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button(next_label, use_container_width=True, key=next_key or f"nx_{s.slide}"):
                nxt(); st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        _, c = st.columns([1, 2])
        with c:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button(next_label, use_container_width=True, key=next_key or f"nx_{s.slide}"):
                nxt(); st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


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
            Answer 6 short steps and receive a detailed prediction instantly.
        </p>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-val">12</div>
                <div class="hero-stat-lbl">Parameters</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-val">6</div>
                <div class="hero-stat-lbl">Steps</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-val">2</div>
                <div class="hero-stat-lbl">Predictions</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Begin Assessment →", use_container_width=True, key="start"):
            go(1); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align:center;font-size:0.7rem;margin-top:1.8rem;color:#3d5068;letter-spacing:1px;">
    FOR EDUCATIONAL USE ONLY · NOT A SUBSTITUTE FOR MEDICAL ADVICE
    </p>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SLIDE 1 — Age & Gender
# ─────────────────────────────────────────
elif s.slide == 1:
    progress_bar(1)
    slide_card_open("Step 1 of 6", "Who <em>are</em> you?",
                    "We start with the basics — your age and gender.")

    s.age = st.slider("Age (years)", 1, 100, int(s.age))
    st.markdown("<br>", unsafe_allow_html=True)
    s.gender = st.radio("Gender", ["Male", "Female"],
                        index=0 if s.gender == "Male" else 1,
                        horizontal=True, key="r_gender")
    slide_card_close()
    btn_row(show_back=False)


# ─────────────────────────────────────────
# SLIDE 2 — Body Metrics
# ─────────────────────────────────────────
elif s.slide == 2:
    progress_bar(2)
    slide_card_open("Step 2 of 6", "Your <em>body</em> metrics.",
                    "BMI and number of dependents are key cost indicators.")

    s.bmi = st.slider("Body Mass Index (BMI)", 10.0, 50.0, float(s.bmi), step=0.1)
    st.markdown("<br>", unsafe_allow_html=True)
    s.children = st.slider("Number of Children / Dependents", 0, 5, int(s.children))
    slide_card_close()
    btn_row()


# ─────────────────────────────────────────
# SLIDE 3 — Lifestyle
# ─────────────────────────────────────────
elif s.slide == 3:
    progress_bar(3)
    slide_card_open("Step 3 of 6", "Your <em>lifestyle</em>.",
                    "Smoking and activity level are strong predictors of health cost.")

    s.smoker = st.radio("Do you smoke?", ["No", "Yes"],
                        index=["No","Yes"].index(s.smoker),
                        horizontal=True, key="r_smoker")
    st.markdown("<br>", unsafe_allow_html=True)
    s.activity = st.radio("Physical Activity Level", ["Low", "Medium", "High"],
                          index=["Low","Medium","High"].index(s.activity),
                          horizontal=True, key="r_activity")
    slide_card_close()
    btn_row()


# ─────────────────────────────────────────
# SLIDE 4 — Coverage & Location
# ─────────────────────────────────────────
elif s.slide == 4:
    progress_bar(4)
    slide_card_open("Step 4 of 6", "Coverage &amp; <em>location</em>.",
                    "Your insurance tier and region shape your cost estimate.")

    s.insurance = st.radio("Insurance Plan", ["Basic", "Premium"],
                           index=["Basic","Premium"].index(s.insurance),
                           horizontal=True, key="r_insurance")
    st.markdown("<br>", unsafe_allow_html=True)
    s.city = st.radio("City Type", ["Urban", "Semi-Urban", "Rural"],
                      index=["Urban","Semi-Urban","Rural"].index(s.city),
                      horizontal=True, key="r_city")
    slide_card_close()
    btn_row()


# ─────────────────────────────────────────
# SLIDE 5 — Medical History
# ─────────────────────────────────────────
elif s.slide == 5:
    progress_bar(5)
    slide_card_open("Step 5 of 6", "Medical <em>history</em>.",
                    "Existing conditions help calibrate the disease risk model.")

    c1, c2 = st.columns(2)
    with c1:
        s.diabetes = st.radio("Diabetes", ["No", "Yes"],
                              index=["No","Yes"].index(str(s.diabetes)) if str(s.diabetes) in ["No","Yes"] else 0,
                              horizontal=True, key="r_db")
        st.markdown("<br>", unsafe_allow_html=True)
        s.heart = st.radio("Heart Disease", ["No", "Yes"],
                           index=["No","Yes"].index(str(s.heart)) if str(s.heart) in ["No","Yes"] else 0,
                           horizontal=True, key="r_hd")
    with c2:
        s.hypertension = st.radio("Hypertension", ["No", "Yes"],
                                  index=["No","Yes"].index(str(s.hypertension)) if str(s.hypertension) in ["No","Yes"] else 0,
                                  horizontal=True, key="r_ht")
        st.markdown("<br>", unsafe_allow_html=True)
        s.asthma = st.radio("Asthma", ["No", "Yes"],
                            index=["No","Yes"].index(str(s.asthma)) if str(s.asthma) in ["No","Yes"] else 0,
                            horizontal=True, key="r_as")
    slide_card_close()
    btn_row()


# ─────────────────────────────────────────
# SLIDE 6 — Review & Predict
# ─────────────────────────────────────────
elif s.slide == 6:
    progress_bar(6)

    st.markdown('<div class="slide-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="slide-eyebrow">Step 6 of 6</div>
    <p class="slide-title">Review &amp; <em>predict</em>.</p>
    <p class="slide-desc">Confirm your details below, then run the prediction.</p>
    """, unsafe_allow_html=True)

    rows = [
        ("Age",           s.age),
        ("Gender",        s.gender),
        ("BMI",           f"{float(s.bmi):.1f}"),
        ("Children",      s.children),
        ("Smoker",        s.smoker),
        ("Activity",      s.activity),
        ("Insurance",     s.insurance),
        ("City",          s.city),
        ("Diabetes",      s.diabetes),
        ("Hypertension",  s.hypertension),
        ("Heart Disease", s.heart),
        ("Asthma",        s.asthma),
    ]

    col_a, col_b = st.columns(2)
    for i, (lbl, val) in enumerate(rows):
        col = col_a if i % 2 == 0 else col_b
        col.markdown(f"""
        <div class="sum-row">
            <span class="sum-lbl">{lbl}</span>
            <span class="sum-val">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns([1, 2, 1])
    with b1:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("Back", use_container_width=True, key="bk6"):
            back(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Run Prediction", use_container_width=True, key="predict"):

            # ── Encode all inputs safely ──
            g_enc  = 1 if s.gender   == "Male"    else 0
            sm_enc = 1 if s.smoker   == "Yes"     else 0
            ac_enc = {"Low":0, "Medium":1, "High":2}.get(s.activity, 1)
            in_enc = {"Basic":0, "Premium":1}.get(s.insurance, 0)
            ci_enc = {"Urban":0, "Semi-Urban":1, "Rural":2}.get(s.city, 0)
            db_enc = 1 if str(s.diabetes)     == "Yes" else 0
            ht_enc = 1 if str(s.hypertension) == "Yes" else 0
            hd_enc = 1 if str(s.heart)        == "Yes" else 0
            as_enc = 1 if str(s.asthma)       == "Yes" else 0

            # Build array — match exact feature order used during training
            inp = np.array([[
                int(s.age), g_enc, float(s.bmi), int(s.children),
                sm_enc, ac_enc, in_enc, ci_enc,
                db_enc, ht_enc, hd_enc, as_enc
            ]], dtype=float)

            try:
                scaled    = scaler.transform(inp)
                s.cost    = float(reg.predict(scaled)[0])
                s.disease = int(clf.predict(scaled)[0])
                s.predicted = True
                go(7); st.rerun()
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    with b3:
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("Restart", use_container_width=True, key="restart6"):
            for k, v in defaults.items(): s[k] = v
            go(0); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# SLIDE 7 — Results
# ─────────────────────────────────────────
elif s.slide == 7 and s.predicted:

    cost    = s.cost
    disease = s.disease
    sm_enc  = 1 if s.smoker   == "Yes" else 0
    ac_enc  = {"Low":0,"Medium":1,"High":2}.get(s.activity, 1)
    in_enc  = {"Basic":0,"Premium":1}.get(s.insurance, 0)
    db_enc  = 1 if str(s.diabetes)     == "Yes" else 0
    ht_enc  = 1 if str(s.hypertension) == "Yes" else 0
    hd_enc  = 1 if str(s.heart)        == "Yes" else 0
    as_enc  = 1 if str(s.asthma)       == "Yes" else 0

    st.markdown("""
    <div class="result-hdr">
        <h2>Your Health Assessment</h2>
        <p>Full breakdown, charts, and personalised recommendations below.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Cards ──
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"""
        <div class="res-card blue">
            <div class="res-lbl">Estimated Annual Cost</div>
            <div class="res-val">&#8377; {cost:,.0f}</div>
            <div class="res-sub">Based on your health profile</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        rc   = "rose"  if disease else "green"
        rlbl = "High Risk" if disease else "Low Risk"
        rsub = "Consult a specialist soon." if disease else "Your profile looks healthy."
        st.markdown(f"""
        <div class="res-card {rc}">
            <div class="res-lbl">Disease Risk</div>
            <div class="res-val">{rlbl}</div>
            <div class="res-sub">{rsub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart theme ──
    BG   = "#2c3e55"; AX = "#253447"
    TEXT = "#9fb3c8"; ACCENT = "#5ba3c9"
    GOLD = "#e8c16a"; GREEN  = "#5ab49a"; ROSE = "#d9717a"

    ch1, ch2 = st.columns(2)

    with ch1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(AX)
        feats = ["Age", "BMI", "Children", "Activity"]
        vals  = [s.age, float(s.bmi), s.children, ac_enc]
        clrs  = [ACCENT, GOLD, GREEN, "#a78bfa"]
        bars  = ax.bar(feats, vals, color=clrs, width=0.48, edgecolor='none', zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    str(round(v,1)), ha='center', va='bottom',
                    color='#eef2f7', fontsize=8.5, fontweight='600')
        ax.set_title("Profile Overview", color=TEXT, fontsize=10, pad=10, fontweight='600')
        ax.tick_params(colors=TEXT, labelsize=8); ax.spines[:].set_visible(False)
        ax.set_ylim(0, max(vals)*1.3+2); ax.yaxis.set_visible(False)
        ax.grid(axis='y', color='#1e2a3a', linewidth=0.8, zorder=0)
        fig.tight_layout(); st.pyplot(fig)

    with ch2:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        fig2.patch.set_facecolor(BG); ax2.set_facecolor(AX)
        conds = ["Diabetes","Hypertension","Heart Disease","Asthma","Smoker"]
        cvals = [db_enc, ht_enc, hd_enc, as_enc, sm_enc]
        bclrs = [ROSE if v else GREEN for v in cvals]
        hbars = ax2.barh(conds, cvals, color=bclrs, height=0.36, edgecolor='none', zorder=3)
        for bar, v in zip(hbars, cvals):
            ax2.text(v+0.04, bar.get_y()+bar.get_height()/2,
                     "Present" if v else "Absent",
                     va='center', color='#eef2f7', fontsize=8.5, fontweight='500')
        ax2.set_title("Condition Status", color=TEXT, fontsize=10, pad=10, fontweight='600')
        ax2.set_xlim(0,1.5); ax2.tick_params(colors=TEXT, labelsize=8)
        ax2.spines[:].set_visible(False); ax2.xaxis.set_visible(False)
        leg = [mpatches.Patch(color=ROSE,label='Present'), mpatches.Patch(color=GREEN,label='Absent')]
        ax2.legend(handles=leg, facecolor='#1e2a3a', edgecolor='none', labelcolor=TEXT, fontsize=8)
        fig2.tight_layout(); st.pyplot(fig2)

    st.markdown("<br>", unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)

    with dc1:
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        fig3.patch.set_facecolor(BG); ax3.set_facecolor(BG)
        base_c = s.age*100; bmi_c = float(s.bmi)*80
        smk_c  = 4000 if sm_enc else 0
        cond_c = (db_enc+ht_enc+hd_enc+as_enc)*1200
        rem_c  = max(cost-base_c-bmi_c-smk_c-cond_c, 500)
        sizes  = [base_c, bmi_c, smk_c, cond_c, rem_c]
        labels = ["Age","BMI","Smoking","Conditions","Base"]
        clrs3  = [ACCENT, GOLD, ROSE, "#f472b6", GREEN]
        wedges, _, auts = ax3.pie(
            sizes, labels=None, autopct='%1.0f%%', startangle=140, colors=clrs3,
            explode=[0.03]*5,
            wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
            pctdistance=0.78,
            textprops=dict(color='#eef2f7', fontsize=8, fontweight='600')
        )
        ax3.set_title("Cost Breakdown", color=TEXT, fontsize=10, pad=10, fontweight='600')
        ax3.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5,-0.15),
                   ncol=3, facecolor='#253447', edgecolor='none', labelcolor=TEXT, fontsize=7.5)
        fig3.tight_layout(); st.pyplot(fig3)

    with dc2:
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        fig4.patch.set_facecolor(BG); ax4.set_facecolor(AX)
        cov  = 60 if in_enc == 0 else 85
        oop  = cost*(1-cov/100)
        cats = ["Total","Covered","Out of Pocket"]
        v4   = [cost, cost*cov/100, oop]
        c4   = [ACCENT, GREEN, ROSE]
        b4   = ax4.bar(cats, v4, color=c4, width=0.44, edgecolor='none', zorder=3)
        for bar, v in zip(b4, v4):
            ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+cost*0.01,
                     f"₹{v:,.0f}", ha='center', va='bottom',
                     color='#eef2f7', fontsize=7.5, fontweight='600')
        ax4.set_title(f"Coverage  ({cov}% insured)", color=TEXT, fontsize=10, pad=10, fontweight='600')
        ax4.tick_params(colors=TEXT, labelsize=8); ax4.spines[:].set_visible(False)
        ax4.yaxis.set_visible(False); ax4.set_ylim(0, max(v4)*1.28)
        ax4.grid(axis='y', color='#1e2a3a', linewidth=0.8, zorder=0)
        fig4.tight_layout(); st.pyplot(fig4)

    # ── Recommendations ──
    st.markdown('<div class="sec-label">Personalised Recommendations</div>', unsafe_allow_html=True)

    tips = []
    if sm_enc:       tips.append(("Smoking Cessation",      "Smoking is the single largest avoidable cost driver. Cessation programmes can reduce risk significantly."))
    if float(s.bmi) > 30: tips.append(("Weight Management", f"A BMI of {float(s.bmi):.1f} is above the healthy range. Structured diet and regular exercise can help."))
    if int(s.age) > 50:   tips.append(("Routine Screenings","Annual screenings are strongly recommended for individuals above 50."))
    if db_enc:       tips.append(("Diabetes Management",    "Consistent blood glucose monitoring and adherence to your medication plan are essential."))
    if ht_enc:       tips.append(("Blood Pressure Control", "Reduce sodium intake, manage stress, and follow your physician's treatment plan closely."))
    if hd_enc:       tips.append(("Cardiac Care",           "A low-sodium, heart-healthy diet and moderate physical activity as advised by your doctor."))
    if as_enc:       tips.append(("Respiratory Care",       "Keep rescue medication accessible and identify and avoid known environmental triggers."))
    if not tips:     tips.append(("Excellent Profile",      "Your health indicators are in a healthy range. Maintain a balanced diet, exercise regularly, and schedule annual check-ups."))

    for title, body in tips:
        st.markdown(f"""
        <div class="tip-card">
            <div class="tip-title">{title}</div>
            <div class="tip-body">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1,2,1])
    with mid:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Start New Assessment", use_container_width=True, key="restart_res"):
            for k, v in defaults.items(): s[k] = v
            go(0); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.balloons()
    st.markdown("""
    <p style="text-align:center;font-size:0.7rem;color:#2c3e55;margin-top:2rem;letter-spacing:1px;">
    MEDIPREDICT AI · EDUCATIONAL USE ONLY · NOT A SUBSTITUTE FOR MEDICAL ADVICE
    </p>""", unsafe_allow_html=True)

else:
    go(0); st.rerun()
