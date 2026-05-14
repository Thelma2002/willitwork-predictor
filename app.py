import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WillItWork? — Side Hustle Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables - Calmer colors ── */
:root {
    --navy:        #0A1628;
    --navy-card:   #0F1D2F;
    --navy-light:  #1A2B3F;
    --blue-main:   #4A90E2;
    --blue-mid:    #6BA3E8;
    --blue-light:  #8BB8F0;
    --blue-pale:   #AACDF5;
    --blue-ghost:  #1E3A5F;
    --text-bright: #E8F0F8;
    --text-muted:  #8BA5C4;
    --success:     #2E7D64;
    --warning:     #B8860B;
    --danger:      #8B3A3A;
}

/* ── Global background ── */
.stApp {
    background-color: var(--navy) !important;
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* ── All text color ── */
.stApp, .stMarkdown, p, label, .stSelectbox label,
.stSlider label, .stRadio label {
    color: var(--text-bright) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stSlider > div {
    background-color: var(--navy-card) !important;
    border: 1px solid var(--blue-ghost) !important;
    border-radius: 8px !important;
    color: var(--text-bright) !important;
}

div[data-baseweb="select"] > div {
    background-color: var(--navy-card) !important;
    border: 1px solid var(--blue-mid) !important;
    color: var(--text-bright) !important;
}

/* ── Slider track ── */
.stSlider > div > div > div > div {
    background: var(--blue-mid) !important;
}

/* ── Radio ── */
.stRadio > div {
    flex-direction: row !important;
    gap: 12px;
    flex-wrap: wrap;
}
.stRadio > div > label {
    background: var(--navy-card) !important;
    border: 1px solid var(--blue-ghost) !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    min-height: 44px;
    display: flex;
    align-items: center;
}
.stRadio > div > label:hover {
    border-color: var(--blue-mid) !important;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #3A7BD5, #4A90E2, #5BA0F0) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    padding: 0.9rem 2rem !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 16px rgba(74, 144, 226, 0.3) !important;
    width: 100% !important;
}
.stButton > button:hover {
    box-shadow: 0 0 24px rgba(74, 144, 226, 0.5) !important;
    transform: translateY(-2px) !important;
}

/* ── Divider ── */
hr {
    border-color: var(--blue-ghost) !important;
    margin: 1.5rem 0 !important;
}

/* ── Mobile responsive adjustments ── */
@media only screen and (max-width: 768px) {
    .block-container {
        padding: 0.5rem !important;
    }
    
    div[data-testid="column"] {
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        padding: 0.75rem !important;
        font-size: 0.9rem !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        min-height: 44px;
    }
    
    .stSlider {
        min-height: 44px;
    }
    
    .dataframe-container {
        overflow-x: auto;
    }
    
    div[style*="background: #0F1D2F"] {
        padding: 1rem !important;
    }
}

/* Tablet adjustments */
@media only screen and (min-width: 769px) and (max-width: 1024px) {
    .block-container {
        padding: 1rem !important;
    }
}

/* Fix for any raw HTML display issues */
div[style*="display: inline-block"] {
    display: inline-block !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model with caching and loading indicator ─────────────────────────────
@st.cache_resource
def load_model():
    with st.spinner("Loading model..."):
        try:
            model = joblib.load('willitwork_model.pkl')
            encoders = joblib.load('label_encoders.pkl')
            feat_cols = joblib.load('feature_cols.pkl')
            return model, encoders, feat_cols
        except FileNotFoundError as e:
            st.error(f"Model files not found: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

model, label_encoders, feature_cols = load_model()


# ── HERO SECTION ─────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #0F1D2F 0%, #1A2B3F 50%, #0F1D2F 100%); border: 1px solid #1E3A5F; border-radius: 16px; padding: 3rem 3rem 2rem 3rem; margin-bottom: 2rem; position: relative; overflow: hidden;">
    <div style="position: absolute; top: -60px; right: -60px; width: 200px; height: 200px; border-radius: 50%; background: radial-gradient(circle, rgba(74,144,226,0.1), transparent 70%);"></div>
    <div style="position: absolute; bottom: -40px; left: 30%; width: 300px; height: 150px; border-radius: 50%; background: radial-gradient(circle, rgba(74,144,226,0.05), transparent 70%);"></div>
    <div style="display: inline-block; background: rgba(74,144,226,0.15); border: 1px solid #4A90E2; border-radius: 20px; padding: 4px 16px; margin-bottom: 1rem; font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #8BB8F0; letter-spacing: 3px; text-transform: uppercase;">Side Hustle Analyser · May 2026</div>
    <h1 style="font-family: 'Space Mono', monospace; font-size: clamp(2rem, 5vw, 3.5rem); font-weight: 700; color: #E8F0F8; margin: 0 0 0.5rem 0; line-height: 1.1;">Will It Work?</h1>
    <p style="font-size: 1.1rem; color: #8BA5C4; max-width: 600px; margin: 0 0 1.5rem 0; font-weight: 300; line-height: 1.6;">Stop guessing. This tool uses a trained Random Forest model to assess the viability of your online side hustle and tell you exactly what will make or break it.</p>
    <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
        <div><span style="font-family:'Space Mono',monospace; font-size:1.4rem; color:#8BB8F0; font-weight:700;">1,500</span><span style="color:#8BA5C4; font-size:0.85rem; display:block;">training records</span></div>
        <div><span style="font-family:'Space Mono',monospace; font-size:1.4rem; color:#8BB8F0; font-weight:700;">24</span><span style="color:#8BA5C4; font-size:0.85rem; display:block;">business models</span></div>
        <div><span style="font-family:'Space Mono',monospace; font-size:1.4rem; color:#8BB8F0; font-weight:700;">9</span><span style="color:#8BA5C4; font-size:0.85rem; display:block;">input features</span></div>
        <div><span style="font-family:'Space Mono',monospace; font-size:1.4rem; color:#8BB8F0; font-weight:700;">RF</span><span style="color:#8BA5C4; font-size:0.85rem; display:block;">random forest model</span></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── SECTION LABEL helper ──────────────────────────────────────────────────────
def section_label(text):
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem; margin-top: 0.5rem;">
        <span style="font-family: 'Space Mono', monospace; font-size: 0.75rem; letter-spacing: 3px; text-transform: uppercase; color: #8BB8F0; font-weight: 700;">{text}</span>
        <div style="flex:1; height:1px; background:linear-gradient(90deg,#1E3A5F,transparent);"></div>
    </div>
    """, unsafe_allow_html=True)


# ── INPUT SECTION ─────────────────────────────────────────────────────────────
section_label("Build Your Business Profile")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
    <div style="background:#0F1D2F; border:1px solid #1E3A5F; border-radius:12px; padding:1.5rem; margin-bottom:1rem;">
    <p style="font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:2px; color:#8BB8F0; text-transform:uppercase; margin-bottom:1rem;">Business Details</p>
    """, unsafe_allow_html=True)

    business_model = st.selectbox(
        "Business Model",
        options=sorted(label_encoders['business_model'].classes_),
        help="Pick the model that best describes your planned hustle"
    )
    content_type = st.selectbox(
        "Primary Content Type",
        options=sorted(label_encoders['content_type'].classes_)
    )
    primary_platform = st.selectbox(
        "Main Platform",
        options=sorted(label_encoders['primary_platform'].classes_)
    )
    prior_experience = st.selectbox(
        "Your Experience Level",
        options=sorted(label_encoders['prior_experience'].classes_)
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div style="background:#0F1D2F; border:1px solid #1E3A5F; border-radius:12px; padding:1.5rem; margin-bottom:1rem;">
    <p style="font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:2px; color:#8BB8F0; text-transform:uppercase; margin-bottom:1rem;">Resources and Market</p>
    """, unsafe_allow_html=True)

    initial_budget = st.select_slider(
        "Startup Budget (ZAR)",
        options=[500, 1000, 2000, 3000, 5000, 8000, 10000, 15000],
        value=2000
    )
    paid_ads_budget = st.select_slider(
        "Monthly Ads Budget (ZAR)",
        options=[0, 500, 1000, 2000, 3000, 5000],
        value=500
    )
    hours_per_week = st.slider(
        "Hours Per Week", min_value=2, max_value=60, value=15, step=1
    )
    niche_saturation = st.slider(
        "Niche Saturation Score",
        min_value=1.0, max_value=10.0, value=5.5, step=0.1,
        help="From your Project 1 Scorecard: 1 = extremely saturated, 10 = untapped niche"
    )
    has_email_list = st.radio(
        "Do you have an email list?",
        options=["No", "Yes"],
        horizontal=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ── NICHE SCORE REFERENCE ─────────────────────────────────────────────────────
with st.expander("Not sure about your Niche Saturation Score? Reference your Project 1 results here"):
    st.markdown("""
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:8px; margin-top:0.5rem;">
        <div style="background:#0F1D2F; border:1px solid #1E3A5F; border-radius:8px; padding:10px; text-align:center;">
            <div style="color:#8BB8F0; font-size:1.2rem; font-weight:700; font-family:'Space Mono',monospace;">7.4</div>
            <div style="color:#8BA5C4; font-size:0.75rem;">No-code Apps</div>
        </div>
        <div style="background:#0F1D2F; border:1px solid #1E3A5F; border-radius:8px; padding:10px; text-align:center;">
            <div style="color:#8BB8F0; font-size:1.2rem; font-weight:700; font-family:'Space Mono',monospace;">7.4</div>
            <div style="color:#8BA5C4; font-size:0.75rem;">AI Video Editing</div>
        </div>
        <div style="background:#0F1D2F; border:1px solid #1E3A5F; border-radius:8px; padding:10px; text-align:center;">
            <div style="color:#8BB8F0; font-size:1.2rem; font-weight:700; font-family:'Space Mono',monospace;">6.8</div>
            <div style="color:#8BA5C4; font-size:0.75rem;">Faceless YouTube</div>
        </div>
        <div style="background:#0F1D2F; border:1px solid #1E3A5F; border-radius:8px; padding:10px; text-align:center;">
            <div style="color:#6BA3E8; font-size:1.2rem; font-weight:700; font-family:'Space Mono',monospace;">5.9</div>
            <div style="color:#8BA5C4; font-size:0.75rem;">Freelancing</div>
        </div>
        <div style="background:#0F1D2F; border:1px solid #1E3A5F; border-radius:8px; padding:10px; text-align:center;">
            <div style="color:#6BA3E8; font-size:1.2rem; font-weight:700; font-family:'Space Mono',monospace;">5.6</div>
            <div style="color:#8BA5C4; font-size:0.75rem;">Print-on-Demand</div>
        </div>
        <div style="background:#0F1D2F; border:1px solid #1E3A5F; border-radius:8px; padding:10px; text-align:center;">
            <div style="color:#8B3A3A; font-size:1.2rem; font-weight:700; font-family:'Space Mono',monospace;">3.9</div>
            <div style="color:#8BA5C4; font-size:0.75rem;">Streaming</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── PREDICT BUTTON ────────────────────────────────────────────────────────────
predict_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)


# ── RESULTS ───────────────────────────────────────────────────────────────────
if predict_btn:
    try:
        encoded = {
            'business_model':          label_encoders['business_model'].transform([business_model])[0],
            'niche_saturation_score':  niche_saturation,
            'initial_budget_zar':      initial_budget,
            'hours_per_week':          hours_per_week,
            'primary_platform':        label_encoders['primary_platform'].transform([primary_platform])[0],
            'has_email_list':          1 if has_email_list == "Yes" else 0,
            'content_type':            label_encoders['content_type'].transform([content_type])[0],
            'paid_ads_budget_zar':     paid_ads_budget,
            'prior_experience':        label_encoders['prior_experience'].transform([prior_experience])[0],
        }
        input_df = pd.DataFrame([encoded])[feature_cols]
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        pct = round(probability * 100, 1)

        if pct >= 65:
            verdict = "LIKELY TO SUCCEED"
            verdict_color = "#2E7D64"
            verdict_bg = "rgba(46,125,100,0.1)"
            verdict_border = "#2E7D64"
            verdict_sub = "Your setup shows strong viability signals."
        elif pct >= 45:
            verdict = "MODERATE ODDS"
            verdict_color = "#B8860B"
            verdict_bg = "rgba(184,134,11,0.1)"
            verdict_border = "#B8860B"
            verdict_sub = "Solid foundation but key gaps need addressing."
        else:
            verdict = "HIGH RISK"
            verdict_color = "#8B3A3A"
            verdict_bg = "rgba(139,58,58,0.1)"
            verdict_border = "#8B3A3A"
            verdict_sub = "Significant headwinds — restructure before launching."

        st.markdown("<br>", unsafe_allow_html=True)
        section_label("Analysis Report")

        # Verdict and Score Bar
        v_col, s_col = st.columns([1.2, 1], gap="large")

        with v_col:
            st.markdown(f"""
            <div style="background: {verdict_bg}; border: 1px solid {verdict_border}; border-left: 5px solid {verdict_border}; border-radius: 12px; padding: 2rem;">
                <div style="font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 3px; color: {verdict_color}; text-transform: uppercase; margin-bottom: 0.5rem;">Verdict</div>
                <div style="font-family: 'Space Mono', monospace; font-size: clamp(1.3rem, 3vw, 2rem); font-weight: 700; color: {verdict_color}; margin-bottom: 0.5rem;">{verdict}</div>
                <div style="color: #8BA5C4; font-size: 0.9rem; margin-bottom: 1.2rem;">{verdict_sub}</div>
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                    <div style="font-family: 'Space Mono', monospace; font-size: 3rem; font-weight: 700; color: {verdict_color}; line-height: 1;">{pct}%</div>
                    <div style="color: #8BA5C4; font-size: 0.8rem;">estimated<br>success<br>probability</div>
                </div>
                <div style="background: rgba(255,255,255,0.08); border-radius: 99px; height: 8px; overflow: hidden; margin-top: 1rem;">
                    <div style="background: {verdict_color}; width: {pct}%; height: 100%; border-radius: 99px; box-shadow: 0 0 8px {verdict_color};"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 0.7rem; color: #8BA5C4; font-family: 'Space Mono', monospace;">
                    <span>0%</span><span>50%</span><span>100%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with s_col:
            st.markdown(f"""
            <div style="background: #0F1D2F; border: 1px solid #1E3A5F; border-radius: 12px; padding: 1.5rem; height: 100%;">
                <div style="font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 3px; color: #8BB8F0; text-transform: uppercase; margin-bottom: 1rem;">Profile Summary</div>
                <div class="dataframe-container">
                <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
                    <tr style="border-bottom:1px solid #1E3A5F;"><td style="color:#8BA5C4; padding:6px 0;">Model</td><td style="color:#E8F0F8; text-align:right; font-weight:500;">{business_model}</td></tr>
                    <tr style="border-bottom:1px solid #1E3A5F;"><td style="color:#8BA5C4; padding:6px 0;">Platform</td><td style="color:#E8F0F8; text-align:right; font-weight:500;">{primary_platform}</td></tr>
                    <tr style="border-bottom:1px solid #1E3A5F;"><td style="color:#8BA5C4; padding:6px 0;">Budget</td><td style="color:#E8F0F8; text-align:right; font-weight:500;">R{initial_budget:,}</td></tr>
                    <tr style="border-bottom:1px solid #1E3A5F;"><td style="color:#8BA5C4; padding:6px 0;">Hours/week</td><td style="color:#E8F0F8; text-align:right; font-weight:500;">{hours_per_week}h</td></tr>
                    <tr style="border-bottom:1px solid #1E3A5F;"><td style="color:#8BA5C4; padding:6px 0;">Experience</td><td style="color:#E8F0F8; text-align:right; font-weight:500;">{prior_experience}</td></tr>
                    <tr style="border-bottom:1px solid #1E3A5F;"><td style="color:#8BA5C4; padding:6px 0;">Email list</td><td style="color:#E8F0F8; text-align:right; font-weight:500;">{has_email_list}</td></tr>
                    <tr><td style="color:#8BA5C4; padding:6px 0;">Saturation score</td><td style="color:#E8F0F8; text-align:right; font-weight:500;">{niche_saturation}/10</td></tr>
                </table>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Top 3 Factors
        section_label("What's Driving This Prediction")

        readable = {
            'business_model':          'Business Model Type',
            'niche_saturation_score':  'Niche Saturation Score',
            'initial_budget_zar':      'Initial Budget',
            'hours_per_week':          'Hours Per Week',
            'primary_platform':        'Primary Platform',
            'has_email_list':          'Having an Email List',
            'content_type':            'Content Type',
            'paid_ads_budget_zar':     'Paid Ads Budget',
            'prior_experience':        'Experience Level'
        }

        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        top3 = imp_df.head(3)
        f_cols = st.columns(3, gap="medium")
        rank_labels = ["Primary Factor", "Secondary Factor", "Contributing Factor"]
        rank_colours = ["#8BB8F0", "#6BA3E8", "#4A90E2"]

        for i, (_, row) in enumerate(top3.iterrows()):
            with f_cols[i]:
                bar_width = int(row['importance'] / imp_df['importance'].max() * 100)
                st.markdown(f"""
                <div style="background: #0F1D2F; border: 1px solid #1E3A5F; border-top: 3px solid {rank_colours[i]}; border-radius: 12px; padding: 1.3rem;">
                    <div style="font-size: 0.7rem; color: {rank_colours[i]}; font-family: 'Space Mono', monospace; letter-spacing: 2px; margin-bottom: 0.5rem; text-transform: uppercase;">{rank_labels[i]}</div>
                    <div style="font-size: 1rem; font-weight: 600; color: #E8F0F8; margin-bottom: 0.8rem;">{readable.get(row['feature'], row['feature'])}</div>
                    <div style="background: rgba(255,255,255,0.06); border-radius: 99px; height: 6px; margin-bottom: 0.4rem;">
                        <div style="background: {rank_colours[i]}; width: {bar_width}%; height: 100%; border-radius: 99px;"></div>
                    </div>
                    <div style="font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #8BA5C4;">importance: {row['importance']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Recommendations
        section_label("Personalised Action Plan")

        advice = []
        if hours_per_week < 10:
            advice.append(("Time", "Increase Your Weekly Hours", "Under 10 hours/week rarely builds momentum. Aim for at least 15 hours to see meaningful progress within 6 months.", "#4A90E2"))
        if has_email_list == "No":
            advice.append(("List", "Build an Email List Now", "It is the single highest-ROI asset in any online business. Start free with Mailchimp — even 200 subscribers changes your odds.", "#3A7BD5"))
        if initial_budget < 1000:
            advice.append(("Budget", "Stretch Your Budget", "Under R1,000 forces a purely organic strategy. This works but is slow. If possible, increase to R2,000 for basic tools and content creation.", "#2E6BB5"))
        if prior_experience == "Beginner" and initial_budget >= 5000:
            advice.append(("Risk", "Budget vs Experience Mismatch", "High budget with beginner experience is a risky combo. Start with R1,000-2,000 to learn what works before scaling spend.", "#1A5BA5"))
        if paid_ads_budget > 0 and has_email_list == "No":
            advice.append(("Ads", "Fix Ads Strategy", "Running paid ads without a list means you are paying for attention you cannot retain. Build the list first, then run ads to warm audiences.", "#3A7BD5"))
        if niche_saturation < 4.5:
            advice.append(("Niche", "High Saturation Risk", "Your niche saturation score is low — meaning high competition. Differentiate through a hyperlocal angle, a language niche, or a very specific sub-niche.", "#2E6BB5"))

        if not advice:
            advice.append(("Setup", "Setup Looks Strong", "Your configuration shows solid fundamentals. Focus on consistency, track your metrics weekly, and reinvest early revenue into content quality.", "#2E7D64"))

        adv_cols = st.columns(min(len(advice), 3), gap="medium")
        for i, (_, title, text, _) in enumerate(advice):
            with adv_cols[i % 3]:
                st.markdown(f"""
                <div style="background: #0F1D2F; border: 1px solid #1E3A5F; border-radius: 12px; padding: 1.3rem; margin-bottom: 0.8rem;">
                    <div style="font-weight: 600; color: #8BB8F0; font-size: 0.9rem; margin-bottom: 0.5rem;">{title}</div>
                    <div style="color: #8BA5C4; font-size: 0.82rem; line-height: 1.6;">{text}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature importance chart
        section_label("Full Feature Importance Breakdown")

        img_path = Path(__file__).parent / 'images' / 'feature_importance.png'
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.info("Feature importance visualization will appear here once available.")

        st.markdown("<br>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: #0F1D2F; border: 1px solid #1E3A5F; border-radius: 12px; padding: 1.5rem 2rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; margin-top: 2rem;">
    <div>
        <div style="font-family: 'Space Mono', monospace; font-size: 1rem; font-weight: 700; color: #8BB8F0; margin-bottom: 0.2rem;">WillItWork? Predictor</div>
        <div style="color:#8BA5C4; font-size:0.8rem;">Built by Lungile Zulu | Data Science Portfolio · Project 3 of 3</div>
    </div>
    <div style="text-align:right;">
        <div style="color:#8BA5C4; font-size:0.78rem; line-height:1.8;">Model: Random Forest Classifier | Training data: 1,500 synthetic records<br>Informed by Project 1: Side Hustle Viability Scorecard | ZA Market Context</div>
    </div>
</div>
""", unsafe_allow_html=True)