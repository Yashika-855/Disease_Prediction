import streamlit as st
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

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
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,600&family=Nunito:wght@300;400;500;600;700&display=swap');

:root {
    --bg:      #f0f4f8;
    --surface: #ffffff;
    --card:    #ffffff;
    --border:  #dde4ed;
    --accent:  #3b7dd8;
    --teal:    #2a9d8f;
    --rose:    #e76f6f;
    --gold:    #e9a84c;
    --text:    #1a2740;
    --sub:     #4a6080;
    --muted:   #8fa5bf;
    --shadow:  rgba(30,60,100,0.10);
}

* { font-family: 'Nunito', sans-serif; box-sizing: border-box; }

.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(ellipse 60% 40% at 0% 0%,   rgba(59,125,216,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 100% 100%, rgba(42,157,143,0.06) 0%, transparent 55%);
}

.block-container { padding: 2.5rem 2rem 5rem !important; max-width: 700px !important; }

/* ── Progress ── */
.prog-wrap { display:flex; align-items:center; gap:6px; margin-bottom:2rem; }
.prog-step {
    width:32px; height:32px; border-radius:50%;
    border: 2px solid var(--border);
    display:flex; align-items:center; justify-content:center;
    font-size:0.72rem; font-weight:700; color:var(--muted);
    background:var(--surface); flex-shrink:0;
}
.prog-step.done   { background:var(--teal);  border-color:var(--teal);  color:#fff; }
.prog-step.active { background:var(--accent); border-color:var(--accent); color:#fff; }
.prog-line      { flex:1; height:2px; background:var(--border); border-radius:2px; }
.prog-line.done { background:var(--teal); }

/* ── Card ── */
.slide-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.4rem 2.6rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px var(--shadow);
    position: relative; overflow: hidden;
}
.slide-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:4px;
    background: linear-gradient(90deg, var(--accent), var(--teal));
    border-radius: 20px 20px 0 0;
}

.step-tag {
    display:inline-block;
    font-size:0.65rem; font-weight:700; letter-spacing:2.5px;
    text-transform:uppercase; color:var(--accent);
    background:rgba(59,125,216,0.08); border-radius:100px;
    padding:0.28rem 0.85rem; margin-bottom:0.75rem;
}
.slide-title {
    font-family:'Cormorant Garamond', serif;
    font-size:1.85rem; font-weight:700; color:var(--text);
    line-height:1.2; margin:0 0 0.3rem;
}
.slide-title em { font-style:italic; color:var(--accent); }
.slide-desc { color:var(--sub); font-size:0.86rem; margin-bottom:1.6rem; line-height:1.65; }

/* ── Radio pill buttons ── */
div[data-testid="stRadio"] > label {
    color:var(--text) !important; font-weight:600 !important; font-size:0.88rem !important;
    margin-bottom:0.5rem !important;
}
div[data-testid="stRadio"] > div {
    display:flex !important; flex-wrap:wrap; gap:0.55rem !important;
}
div[data-testid="stRadio"] > div > label {
    background: #f0f4f8 !important;
    border: 2px solid var(--border) !important;
    border-radius: 100px !important;
    padding: 0.5rem 1.4rem !important;
    color: var(--sub) !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    cursor: pointer; transition: all 0.18s;
    box-shadow: none !important;
}
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #ffffff !important;
    box-shadow: 0 3px 12px rgba(59,125,216,0.28) !important;
}

/* ── Sliders ── */
.stSlider label { color:var(--text) !important; font-weight:600 !important; font-size:0.88rem !important; }
.stSlider > div > div > div > div { background: var(--accent) !important; }
.stSlider [data-baseweb="thumb"] {
    background: #fff !important;
    border: 3px solid var(--accent) !important;
    width:18px !important; height:18px !important;
    box-shadow: 0 2px 8px rgba(59,125,216,0.3) !important;
}

/* ── Nav Buttons ── */
.stButton > button {
    border-radius: 100px !important;
    height: 3em !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    border: none !important;
    transition: all .2s ease !important;
    width: 100% !important;
}
.primary-btn .stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    box-shadow: 0 4px 16px rgba(59,125,216,0.30) !important;
}
.primary-btn .stButton > button:hover { background:#2f6bbf !important; transform:translateY(-1px) !important; }

.ghost-btn .stButton > button {
    background: #fff !important;
    border: 2px solid var(--border) !important;
    color: var(--sub) !important;
    box-shadow: none !important;
}
.ghost-btn .stButton > button:hover { border-color:var(--accent) !important; color:var(--accent) !important; }

.danger-btn .stButton > button {
    background: #fff !important;
    border: 2px solid #f5c6c6 !important;
    color: var(--rose) !important;
    box-shadow: none !important;
}

/* ── Hero ── */
.hero-wrap {
    text-align:center;
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:24px;
    padding:3.2rem 2.5rem 2.8rem;
    box-shadow:0 4px 28px var(--shadow);
    position:relative; overflow:hidden;
    margin-bottom:1.5rem;
}
.hero-wrap::before {
    content:''; position:absolute; top:0; left:0; right:0; height:5px;
    background:linear-gradient(90deg, var(--accent), var(--teal), var(--gold));
}
.hero-eyebrow {
    font-size:0.65rem; letter-spacing:4px; text-transform:uppercase;
    color:var(--accent); font-weight:700; margin-bottom:0.8rem;
}
.hero-h1 {
    font-family:'Cormorant Garamond', serif;
    font-size:3.2rem; font-weight:700; color:var(--text);
    line-height:1.1; margin:0 0 0.9rem;
}
.hero-h1 span { color:var(--accent); font-style:italic; }
.hero-para {
    color:var(--sub); font-size:0.95rem; font-weight:400;
    max-width:480px; margin:0 auto 2rem; line-height:1.75;
}
.hero-stats {
    display:flex; justify-content:center; gap:0;
    border:1px solid var(--border); border-radius:14px;
    overflow:hidden; margin-bottom:2rem;
}
.hero-stat {
    flex:1; padding:1.1rem 0.5rem;
    border-right:1px solid var(--border);
}
.hero-stat:last-child { border-right:none; }
.hero-stat-val { font-family:'Cormorant Garamond',serif; font-size:2rem; font-weight:700; color:var(--accent); }
.hero-stat-lbl { font-size:0.68rem; color:var(--muted); letter-spacing:1px; text-transform:uppercase; margin-top:2px; }

/* ── Result cards ── */
.res-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    position:relative; overflow:hidden;
    box-shadow:0 2px 12px var(--shadow);
    margin-bottom: 0.8rem;
}
.res-card::before { content:''; position:absolute; top:0; left:0; right:0; height:4px; }
.res-card.blue::before  { background:linear-gradient(90deg,var(--accent),#5b9be8); }
.res-card.rose::before  { background:linear-gradient(90deg,var(--rose),#f0a0a0); }
.res-card.green::before { background:linear-gradient(90deg,var(--teal),#5bbcb0); }

.res-lbl  { font-size:0.65rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--muted); margin-bottom:0.4rem; }
.res-val  { font-family:'Cormorant Garamond',serif; font-size:2.2rem; font-weight:700; color:var(--text); line-height:1.15; }
.res-sub  { font-size:0.78rem; color:var(--sub); margin-top:0.2rem; }

/* ── Tip cards ── */
.tip-card {
    background: var(--surface);
    border:1px solid var(--border);
    border-left:4px solid var(--accent);
    border-radius:12px;
    padding:1rem 1.3rem;
    margin-bottom:0.7rem;
    box-shadow:0 1px 6px var(--shadow);
}
.tip-title { font-weight:700; color:var(--text); font-size:0.88rem; margin-bottom:0.2rem; }
.tip-body  { font-size:0.8rem; color:var(--sub); line-height:1.6; }

/* ── Summary table ── */
.sum-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:0.55rem 0; border-bottom:1px solid #eef1f5;
}
.sum-lbl { color:var(--sub); font-size:0.82rem; }
.sum-val { color:var(--text); font-weight:700; font-size:0.86rem; }

/* ── Misc ── */
hr { border-color:var(--border) !important; margin:1.3rem 0 !important; }
.sec-label {
    font-size:0.65rem; font-weight:700; letter-spacing:3px;
    text-transform:uppercase; color:var(--accent); margin:1.4rem 0 0.8rem;
}
p { color:var(--sub) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Load Models & Detect Feature Count
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    missing = [f for f in ["reg_model.pkl","clf_model.pkl","scaler.pkl"] if not os.path.exists(f)]
    if missing:
        st.error(f"Missing files: {', '.join(missing)}")
        st.stop()
    reg    = pickle.load(open("reg_model.pkl","rb"))
    clf    = pickle.load(open("clf_model.pkl","rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))
    n_feat = scaler.n_features_in_

    # Detect which class label the model uses for "disease present"
    # Models trained with 0=healthy,1=disease OR 1=healthy,0=disease both handled
    classes = list(clf.classes_) if hasattr(clf, 'classes_') else [0, 1]
    # HIGH risk label = whichever class is NOT the majority / not 0
    # We trust: if classes are [0,1] → 1 means disease (standard)
    # predict_proba col index for disease=1
    if 1 in classes:
        high_risk_label = 1
    else:
        high_risk_label = classes[-1]   # fallback: last class

    return reg, clf, scaler, n_feat, high_risk_label

reg, clf, scaler, N_FEATURES, HIGH_RISK_LABEL = load_models()


# ─────────────────────────────────────────
# Build Input Vector — handles 12 or 19 features
# ─────────────────────────────────────────
def build_input(age, gender, bmi, children, smoker,
                activity, insurance, city,
                diabetes, hypertension, heart, asthma):
    """
    Tries the exact feature set the scaler was trained on.
    Supports both raw-encoded (12 features) and one-hot-encoded (19 features) schemas.
    """
    g  = 1 if gender   == "Male"    else 0
    sm = 1 if smoker   == "Yes"     else 0
    ac = {"Low":0,"Medium":1,"High":2}[activity]
    ins= {"Basic":0,"Premium":1}[insurance]
    ct = {"Urban":0,"Semi-Urban":1,"Rural":2}[city]
    db = diabetes; ht = hypertension; hd = heart; ab = asthma

    # 12-feature raw schema
    raw12 = [age, g, bmi, children, sm, ac, ins, ct, db, ht, hd, ab]

    # 19-feature one-hot schema
    # gender: male, female
    # smoker: yes, no
    # activity: low, medium, high
    # insurance: basic, premium
    # city: urban, semi-urban, rural
    # numeric: age, bmi, children
    # conditions: diabetes, hypertension, heart, asthma
    g_male   = 1 if gender    == "Male"      else 0
    g_fem    = 1 if gender    == "Female"    else 0
    sm_yes   = 1 if smoker    == "Yes"       else 0
    sm_no    = 1 if smoker    == "No"        else 0
    ac_low   = 1 if activity  == "Low"       else 0
    ac_med   = 1 if activity  == "Medium"    else 0
    ac_high  = 1 if activity  == "High"      else 0
    ins_bas  = 1 if insurance == "Basic"     else 0
    ins_prem = 1 if insurance == "Premium"   else 0
    ct_urb   = 1 if city      == "Urban"     else 0
    ct_semi  = 1 if city      == "Semi-Urban" else 0
    ct_rur   = 1 if city      == "Rural"     else 0

    ohe19 = [age, bmi, children,
             g_male, g_fem,
             sm_yes, sm_no,
             ac_low, ac_med, ac_high,
             ins_bas, ins_prem,
             ct_urb, ct_semi, ct_rur,
             db, ht, hd, ab]

    # Alt 19: age,gender,bmi,children,smoker,ac,ins,ct + 4 conds + 7 ohe dummies
    # Pick whichever matches scaler expectation
    candidates = {
        12: raw12,
        19: ohe19,
    }

    if N_FEATURES in candidates:
        return np.array([candidates[N_FEATURES]], dtype=float)

    # Fallback: pad raw12 with zeros up to N_FEATURES, or trim
    arr = raw12[:N_FEATURES] + [0]*(max(0, N_FEATURES - len(raw12)))
    return np.array([arr[:N_FEATURES]], dtype=float)


# ─────────────────────────────────────────
# Session State
# ─────────────────────────────────────────
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
# UI Helpers
# ─────────────────────────────────────────
def go(n):  s.slide = n
def nxt():  s.slide += 1
def back(): s.slide = max(0, s.slide-1)

def progress_bar(current, total=3):
    labels = ["Profile","Lifestyle","History"]
    html = '<div class="prog-wrap">'
    for i in range(1, total+1):
        if i > 1:
            lc = "done" if i <= current else ""
            html += f'<div class="prog-line {lc}"></div>'
        dc = "done" if i < current else ("active" if i == current else "")
        html += f'<div class="prog-step {dc}">{i}</div>'
    html += f'<span style="margin-left:6px;font-size:0.75rem;color:var(--muted);font-weight:600;">{labels[current-1]}</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def card_open(step, title, desc):
    st.markdown(f"""
    <div class="slide-card">
      <span class="step-tag">{step}</span>
      <p class="slide-title">{title}</p>
      <p class="slide-desc">{desc}</p>
    """, unsafe_allow_html=True)

def card_close():
    st.markdown('</div>', unsafe_allow_html=True)

def nav(back_label="← Back", next_label="Continue →", show_back=True, next_key="", back_key=""):
    st.markdown("<br>", unsafe_allow_html=True)
    if show_back:
        c1, c2 = st.columns([1,2])
        with c1:
            st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
            if st.button(back_label, use_container_width=True, key=back_key or f"bk{s.slide}"):
                back(); st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button(next_label, use_container_width=True, key=next_key or f"nx{s.slide}"):
                nxt(); st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        _, c = st.columns([1,2])
        with c:
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button(next_label, use_container_width=True, key=next_key or f"nx{s.slide}"):
                nxt(); st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# SLIDE 0 — Welcome
# ═══════════════════════════════════════════
if s.slide == 0:
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-eyebrow">AI · Health Intelligence</div>
      <h1 class="hero-h1">Medi<span>Predict</span></h1>
      <p class="hero-para">
        Enter your health profile and receive an instant estimate of your
        annual medical cost alongside a personalised disease risk score.
      </p>
      <div class="hero-stats">
        <div class="hero-stat">
          <div class="hero-stat-val">3</div>
          <div class="hero-stat-lbl">Steps</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-val">ML</div>
          <div class="hero-stat-lbl">Powered</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-val">2</div>
          <div class="hero-stat-lbl">Outputs</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1,2,1])
    with mid:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Get Started →", use_container_width=True, key="start"):
            go(1); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align:center;font-size:0.68rem;margin-top:1.4rem;color:var(--muted);letter-spacing:0.8px;">
    For educational purposes only · Not medical advice
    </p>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# SLIDE 1 — Personal Profile
# ═══════════════════════════════════════════
elif s.slide == 1:
    progress_bar(1)
    card_open("Step 1 of 3", "Personal <em>Profile</em>",
              "Tell us about yourself — age, body metrics, and coverage details.")

    col1, col2 = st.columns(2)
    with col1:
        s.age = st.slider("Age", 1, 100, int(s.age))
    with col2:
        s.bmi = st.slider("BMI", 10.0, 50.0, float(s.bmi), step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        s.children = st.slider("Dependents", 0, 5, int(s.children))
    with col4:
        s.gender = st.radio("Gender", ["Male","Female"],
                            index=0 if s.gender=="Male" else 1,
                            horizontal=True, key="r_gen")

    st.markdown("<br>", unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        s.insurance = st.radio("Insurance", ["Basic","Premium"],
                               index=["Basic","Premium"].index(s.insurance),
                               horizontal=True, key="r_ins")
    with col6:
        s.city = st.radio("City", ["Urban","Semi-Urban","Rural"],
                          index=["Urban","Semi-Urban","Rural"].index(s.city),
                          horizontal=True, key="r_city")

    card_close()
    nav(show_back=False)


# ═══════════════════════════════════════════
# SLIDE 2 — Lifestyle
# ═══════════════════════════════════════════
elif s.slide == 2:
    progress_bar(2)
    card_open("Step 2 of 3", "Lifestyle <em>Habits</em>",
              "Smoking and activity level are among the strongest cost predictors.")

    st.markdown("<br>", unsafe_allow_html=True)
    s.smoker = st.radio("Do you currently smoke?", ["No","Yes"],
                        index=["No","Yes"].index(s.smoker),
                        horizontal=True, key="r_smoke")

    st.markdown("<br>", unsafe_allow_html=True)
    s.activity = st.radio("Physical Activity Level", ["Low","Medium","High"],
                          index=["Low","Medium","High"].index(s.activity),
                          horizontal=True, key="r_act")
    st.markdown("<br>", unsafe_allow_html=True)

    card_close()
    nav()


# ═══════════════════════════════════════════
# SLIDE 3 — Medical History + Predict
# ═══════════════════════════════════════════
elif s.slide == 3:
    progress_bar(3)
    card_open("Step 3 of 3", "Medical <em>History</em>",
              "Existing conditions help calibrate the risk model. Select all that apply.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Display as 2x2 toggle radios
    c1, c2 = st.columns(2)
    with c1:
        db = st.radio("Diabetes", ["No","Yes"],
                      index=int(s.diabetes), horizontal=True, key="r_db")
        st.markdown("<br>", unsafe_allow_html=True)
        hd = st.radio("Heart Disease", ["No","Yes"],
                      index=int(s.heart), horizontal=True, key="r_hd")
    with c2:
        ht = st.radio("Hypertension", ["No","Yes"],
                      index=int(s.hypertension), horizontal=True, key="r_ht")
        st.markdown("<br>", unsafe_allow_html=True)
        ab = st.radio("Asthma", ["No","Yes"],
                      index=int(s.asthma), horizontal=True, key="r_ab")

    s.diabetes     = 1 if db == "Yes" else 0
    s.hypertension = 1 if ht == "Yes" else 0
    s.heart        = 1 if hd == "Yes" else 0
    s.asthma       = 1 if ab == "Yes" else 0

    card_close()
    st.markdown("<br>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns([1,2,1])
    with b1:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True, key="bk3"):
            back(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with b2:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Predict Now →", use_container_width=True, key="predict"):
            try:
                inp    = build_input(
                    int(s.age), s.gender, float(s.bmi), int(s.children),
                    s.smoker, s.activity, s.insurance, s.city,
                    s.diabetes, s.hypertension, s.heart, s.asthma
                )
                scaled = scaler.transform(inp)
                s.cost = float(reg.predict(scaled)[0])

                # Robust disease label: use predict_proba to avoid label inversion bugs
                if hasattr(clf, 'predict_proba'):
                    proba   = clf.predict_proba(scaled)[0]
                    classes = list(clf.classes_)
                    idx     = classes.index(HIGH_RISK_LABEL) if HIGH_RISK_LABEL in classes else -1
                    risk_prob = proba[idx]
                    s.disease = 1 if risk_prob >= 0.5 else 0
                else:
                    raw = int(clf.predict(scaled)[0])
                    s.disease = 1 if raw == HIGH_RISK_LABEL else 0

                s.predicted = True
                go(4); st.rerun()
            except Exception as e:
                st.error(f"Could not generate prediction. Please check your model files are compatible. ({e})")
        st.markdown('</div>', unsafe_allow_html=True)
    with b3:
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("Restart", use_container_width=True, key="restart3"):
            for k,v in defaults.items(): s[k]=v
            go(0); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# SLIDE 4 — Results
# ═══════════════════════════════════════════
elif s.slide == 4 and s.predicted:

    cost    = s.cost
    disease = s.disease
    sm_enc  = 1 if s.smoker=="Yes" else 0
    ac_enc  = {"Low":0,"Medium":1,"High":2}[s.activity]
    in_enc  = {"Basic":0,"Premium":1}[s.insurance]

    # ── Result header ──
    st.markdown(f"""
    <div style="text-align:center;padding:1.5rem 0 1rem;">
      <div style="font-size:0.65rem;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:var(--accent);margin-bottom:0.5rem;">Assessment Complete</div>
      <p style="font-family:'Cormorant Garamond',serif;font-size:2rem;font-weight:700;color:var(--text);margin:0 0 0.3rem;">Your Health Report</p>
      <p style="font-size:0.85rem;color:var(--sub);">Based on the profile you provided</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Two metric cards ──
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class="res-card blue">
          <div class="res-lbl">Annual Medical Cost</div>
          <div class="res-val">&#8377;{cost:,.0f}</div>
          <div class="res-sub">Estimated for your profile</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        rc   = "rose"  if disease else "green"
        rlbl = "High Risk" if disease else "Low Risk"
        rsub = "Specialist consultation advised" if disease else "Profile within healthy range"
        st.markdown(f"""
        <div class="res-card {rc}">
          <div class="res-lbl">Disease Risk</div>
          <div class="res-val">{rlbl}</div>
          <div class="res-sub">{rsub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    BG    = "#ffffff"; AX = "#f7f9fc"
    TEXT  = "#6b8299"; BLUE = "#3b7dd8"
    TEAL  = "#2a9d8f"; ROSE = "#e76f6f"; GOLD = "#e9a84c"

    ch1, ch2 = st.columns(2)
    with ch1:
        fig, ax = plt.subplots(figsize=(4.8,3.4))
        fig.patch.set_facecolor(BG); ax.set_facecolor(AX)
        feats = ["Age","BMI","Children","Activity"]
        vals  = [s.age, float(s.bmi), s.children, ac_enc]
        clrs  = [BLUE, TEAL, GOLD, "#a78bfa"]
        bars  = ax.bar(feats, vals, color=clrs, width=0.46, edgecolor='none', zorder=3)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    str(round(v,1)), ha='center', va='bottom',
                    color='#1a2740', fontsize=8.5, fontweight='700')
        ax.set_title("Profile Overview", color=TEXT, fontsize=9.5, pad=8, fontweight='700')
        ax.tick_params(colors=TEXT, labelsize=8); ax.spines[:].set_visible(False)
        ax.set_ylim(0, max(vals)*1.32+1); ax.yaxis.set_visible(False)
        ax.grid(axis='y', color='#e8edf3', linewidth=0.8, zorder=0)
        fig.tight_layout(); st.pyplot(fig)

    with ch2:
        fig2, ax2 = plt.subplots(figsize=(4.8,3.4))
        fig2.patch.set_facecolor(BG); ax2.set_facecolor(AX)
        conds = ["Diabetes","Hypertension","Heart Dis.","Asthma","Smoker"]
        cvals = [s.diabetes, s.hypertension, s.heart, s.asthma, sm_enc]
        bclrs = [ROSE if v else TEAL for v in cvals]
        hbars = ax2.barh(conds, cvals, color=bclrs, height=0.34, edgecolor='none', zorder=3)
        for bar,v in zip(hbars,cvals):
            ax2.text(v+0.04, bar.get_y()+bar.get_height()/2,
                     "Yes" if v else "No",
                     va='center', color='#1a2740', fontsize=8.5, fontweight='700')
        ax2.set_title("Condition Flags", color=TEXT, fontsize=9.5, pad=8, fontweight='700')
        ax2.set_xlim(0,1.5); ax2.tick_params(colors=TEXT, labelsize=8)
        ax2.spines[:].set_visible(False); ax2.xaxis.set_visible(False)
        leg=[mpatches.Patch(color=ROSE,label='Present'),mpatches.Patch(color=TEAL,label='Absent')]
        ax2.legend(handles=leg, facecolor='#f7f9fc', edgecolor='#dde4ed', labelcolor=TEXT, fontsize=8)
        fig2.tight_layout(); st.pyplot(fig2)

    st.markdown("<br>", unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        fig3, ax3 = plt.subplots(figsize=(4.8,3.4))
        fig3.patch.set_facecolor(BG); ax3.set_facecolor(BG)
        bc = s.age*100; bmic = float(s.bmi)*80
        sc = 4000 if sm_enc else 0
        cc = (s.diabetes+s.hypertension+s.heart+s.asthma)*1200
        rc2= max(cost-bc-bmic-sc-cc,500)
        szs=[bc,bmic,sc,cc,rc2]; lbls=["Age","BMI","Smoking","Conditions","Base"]
        cl3=[BLUE,TEAL,ROSE,GOLD,"#a78bfa"]
        wedges,_,auts=ax3.pie(szs,labels=None,autopct='%1.0f%%',startangle=140,
            colors=cl3,explode=[0.03]*5,
            wedgeprops=dict(width=0.55,edgecolor='white',linewidth=1.5),
            pctdistance=0.78,textprops=dict(color='#1a2740',fontsize=8,fontweight='700'))
        ax3.set_title("Cost Factors", color=TEXT, fontsize=9.5, pad=8, fontweight='700')
        ax3.legend(wedges,lbls,loc="lower center",bbox_to_anchor=(0.5,-0.15),
                   ncol=3,facecolor='#f7f9fc',edgecolor='#dde4ed',labelcolor=TEXT,fontsize=7.5)
        fig3.tight_layout(); st.pyplot(fig3)

    with dc2:
        fig4, ax4 = plt.subplots(figsize=(4.8,3.4))
        fig4.patch.set_facecolor(BG); ax4.set_facecolor(AX)
        cov=60 if in_enc==0 else 85; oop=cost*(1-cov/100)
        cats=["Total","Covered","Out of Pocket"]; v4=[cost,cost*cov/100,oop]; c4=[BLUE,TEAL,ROSE]
        b4=ax4.bar(cats,v4,color=c4,width=0.42,edgecolor='none',zorder=3)
        for bar,v in zip(b4,v4):
            ax4.text(bar.get_x()+bar.get_width()/2,bar.get_height()+cost*0.012,
                     f"₹{v:,.0f}",ha='center',va='bottom',
                     color='#1a2740',fontsize=7.5,fontweight='700')
        ax4.set_title(f"Coverage ({cov}%)",color=TEXT,fontsize=9.5,pad=8,fontweight='700')
        ax4.tick_params(colors=TEXT,labelsize=8); ax4.spines[:].set_visible(False)
        ax4.yaxis.set_visible(False); ax4.set_ylim(0,max(v4)*1.28)
        ax4.grid(axis='y',color='#e8edf3',linewidth=0.8,zorder=0)
        fig4.tight_layout(); st.pyplot(fig4)

    # ── Recommendations ──
    st.markdown('<div class="sec-label">Recommendations</div>', unsafe_allow_html=True)
    tips=[]
    if sm_enc:           tips.append(("Smoking Cessation",  "Quitting smoking is the single most impactful change to lower both cost and disease risk."))
    if float(s.bmi)>30: tips.append(("Weight Management",  f"BMI of {float(s.bmi):.1f} is above the healthy range. A structured plan can reduce risk significantly."))
    if int(s.age)>50:   tips.append(("Routine Screenings", "Annual health screenings are strongly advised for individuals above 50."))
    if s.diabetes:      tips.append(("Diabetes Care",       "Regular blood glucose monitoring and strict medication adherence are essential."))
    if s.hypertension:  tips.append(("Blood Pressure",      "Low sodium diet, stress management, and physician-guided treatment are key."))
    if s.heart:         tips.append(("Cardiac Health",      "Heart-healthy diet and moderate, doctor-approved exercise are recommended."))
    if s.asthma:        tips.append(("Respiratory Care",    "Keep rescue inhalers accessible and avoid known environmental triggers."))
    # If model says low risk AND no conditions flagged — show positive message
    if not tips:
        if disease == 0:
            tips.append(("Healthy Profile", "Your indicators are in a healthy range. Maintain regular check-ups and a balanced lifestyle."))
        else:
            tips.append(("General Wellness", "Even without specific conditions flagged, maintaining a healthy lifestyle helps reduce overall risk."))

    for title,body in tips:
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
            for k,v in defaults.items(): s[k]=v
            go(0); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align:center;font-size:0.68rem;color:var(--muted);margin-top:2rem;letter-spacing:0.8px;">
    MediPredict AI · For educational purposes only · Not a substitute for medical advice
    </p>""", unsafe_allow_html=True)

else:
    go(0); st.rerun()
