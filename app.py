import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WillItWork?",
    layout="wide"
)

# ── Load model files ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('willitwork_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    return model, encoders, feature_cols

model, label_encoders, feature_cols = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("WillItWork?")
st.subheader("Online Side Hustle Success Predictor")
st.write(
    "Fill in the details below about your planned online side hustle. "
    "The model will predict your chances of being profitable within 6 months "
    "— and explain the top factors driving that prediction."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Your Business")
    business_model = st.selectbox(
        "Business Model",
        options=sorted(label_encoders['business_model'].classes_)
    )
    content_type = st.selectbox(
        "Primary Content Type",
        options=sorted(label_encoders['content_type'].classes_)
    )
    primary_platform = st.selectbox(
        "Main Platform",
        options=sorted(label_encoders['primary_platform'].classes_)
    )

with col2:
    st.subheader("Your Resources")
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
        "Hours Per Week Available",
        min_value=2, max_value=60, value=15, step=1
    )

with col3:
    st.subheader("Your Situation")
    prior_experience = st.selectbox(
        "Prior Experience Level",
        options=sorted(label_encoders['prior_experience'].classes_)
    )
    has_email_list = st.radio(
        "Do you have an email list?",
        options=["No", "Yes"]
    )
    niche_saturation = st.slider(
        "Niche Saturation Score (from your research)",
        min_value=1.0, max_value=10.0, value=5.5, step=0.1,
        help="1 = extremely saturated market, 10 = untapped niche. "
             "Refer to Project 1 scorecard for your business model's score."
    )

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
predict_btn = st.button("Predict My Chances", type="primary", use_container_width=True)

if predict_btn:
    # Encode inputs exactly as done during training
    encoded = {
        'business_model':         label_encoders['business_model'].transform([business_model])[0],
        'niche_saturation_score': niche_saturation,
        'initial_budget_zar':     initial_budget,
        'hours_per_week':         hours_per_week,
        'primary_platform':       label_encoders['primary_platform'].transform([primary_platform])[0],
        'has_email_list':         1 if has_email_list == "Yes" else 0,
        'content_type':           label_encoders['content_type'].transform([content_type])[0],
        'paid_ads_budget_zar':    paid_ads_budget,
        'prior_experience':       label_encoders['prior_experience'].transform([prior_experience])[0]
    }

    input_df = pd.DataFrame([encoded])[feature_cols]

    # Get prediction and probability
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # probability of success

    # ── Result display ────────────────────────────────────────────────────────
    st.subheader("Your Prediction")
    res_col1, res_col2, res_col3 = st.columns([1, 1, 1])

    with res_col1:
        if probability >= 0.65:
            st.success("LIKELY TO SUCCEED")
            verdict = "Strong Chances"
            colour = "green"
        elif probability >= 0.45:
            st.warning("MODERATE ODDS")
            verdict = "50/50 — Needs Work"
            colour = "orange"
        else:
            st.error("HIGH RISK")
            verdict = "Challenging Path"
            colour = "red"

        st.metric(label="Success Probability", value=f"{probability*100:.1f}%")

    with res_col2:
        st.metric(label="Verdict", value=verdict)
        st.metric(label="Business Model", value=business_model)

    with res_col3:
        # Simple probability bar
        fig, ax = plt.subplots(figsize=(4, 1.5))
        ax.barh([''], [probability], color=colour, height=0.5)
        ax.barh([''], [1 - probability], left=[probability],
                color='#EEEEEE', height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability of Success')
        ax.set_title(f'{probability*100:.1f}% Chance of Success')
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        st.pyplot(fig)
        plt.close()

    # ── Top 3 factors ─────────────────────────────────────────────────────────
    st.subheader("Top 3 Factors Affecting Your Prediction")
    st.write("Based on the model's feature importance scores:")

    importances = model.feature_importances_
    readable = {
        'business_model':         'Business Model Type',
        'niche_saturation_score': 'Niche Saturation Score',
        'initial_budget_zar':     'Initial Budget',
        'hours_per_week':         'Hours Per Week',
        'primary_platform':       'Primary Platform',
        'has_email_list':         'Having an Email List',
        'content_type':           'Content Type',
        'paid_ads_budget_zar':    'Paid Ads Budget',
        'prior_experience':       'Experience Level'
    }

    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False).head(3)

    for rank, (_, row) in enumerate(imp_df.iterrows(), 1):
        feature_name = readable.get(row['feature'], row['feature'])
        st.write(f"**{rank}. {feature_name}** — importance score: {row['importance']:.3f}")

    # ── Personalised advice ───────────────────────────────────────────────────
    st.subheader(" Personalised Recommendations")

    advice = []
    if hours_per_week < 10:
        advice.append(" **Increase your weekly hours.** Under 10 hours/week rarely builds momentum. "
                      "Aim for at least 15 hours to see meaningful progress.")
    if has_email_list == "No":
        advice.append(" **Start building an email list from Day 1.** "
                      "It's the single highest-ROI asset in any online business. "
                      "Use a free tier of Mailchimp or ConvertKit.")
    if initial_budget < 1000:
        advice.append(" **Your budget is very tight.** Focus only on free/organic strategies "
                      "until you generate first revenue. Avoid paid ads until you know what converts.")
    if prior_experience == "Beginner" and initial_budget >= 5000:
        advice.append(" **High budget + beginner experience is a risky combination.** "
                      "Consider starting with R1,000-2,000 to learn before scaling.")
    if paid_ads_budget > 0 and not has_email_list:
        advice.append(" **Don't run paid ads to a cold audience with no email list.** "
                      "Build a list first — ads without retargeting burn budget fast.")

    if not advice:
        advice.append("Your setup looks solid. Focus on consistency and tracking your metrics weekly.")

    for tip in advice:
        st.info(tip)

    # ── Feature importance chart ───────────────────────────────────────────────
    st.subheader("What the Model Considers Most Important")
    try:
        st.image('images/feature_importance.png', width=True)
    except:
        st.write("Feature importance chart not found. Make sure feature_importance.png "
                 "is in the images/ folder.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built by Lungile | Data Science Portfolio | May 2026 | "
    "Model: Random Forest Classifier | Training data: 1,500 synthetic records "
    "informed by Project 1 Side Hustle Viability Scorecard"
)