# WillItWork? - Side Hustle Success Predictor

## Live App
Try it here: [https://willitwork-predictor-ltzulu.streamlit.app/](https://willitwork-predictor-ltzulu.streamlit.app/)

## What This App Does

This app helps you figure out if your online side hustle idea will actually make money within 6 months. It uses a machine learning model (Random Forest) trained on 1,500 synthetic business records to predict your chances of success.

Instead of guessing, you get:
- A clear success probability percentage
- The top 3 factors that affect your prediction
- Personalised advice to improve your chances

## Who Is This For?

- Anyone thinking about starting an online side hustle
- People who want data-backed answers, not just opinions
- Beginners who need guidance on where to focus their efforts

## How to Use It

### Step 1: Fill in Your Business Details
- Choose your **business model** (e.g., freelancing, print-on-demand, content creation)
- Pick your **content type** and **main platform**
- Select your **experience level** (Beginner, Intermediate, Advanced)

### Step 2: Tell Us About Your Resources
- Your **startup budget** (how much money you can invest upfront)
- Your **monthly ads budget** (if you plan to run paid advertising)
- **Hours per week** you can dedicate to the hustle

### Step 3: Share Your Market Situation
- **Email list status** (do you have one or not?)
- **Niche saturation score** (how crowded is your market?)
  - 1 = extremely saturated (lots of competition)
  - 10 = completely untapped (no competition)

### Step 4: Get Your Results
Click the **"RUN ANALYSIS"** button and you'll see:
- Success probability (0% to 100%)
- Verdict (Likely to Succeed, Moderate Odds, or High Risk)
- Top factors influencing your prediction
- Personalised recommendations to improve your chances

## What The Results Mean

| Probability | Verdict | What It Means |
|-------------|---------|----------------|
| 65% or higher | Likely to Succeed | Your setup looks strong. Keep going! |
| 45% - 64% | Moderate Odds | Good foundation, but fix some gaps |
| Below 45% | High Risk | Restructure before launching |

## How The Model Works

The app uses a **Random Forest Classifier** trained on:
- 1,500 synthetic business records
- 9 different input features (budget, hours, experience, etc.)
- 24 different business models

The model was built using insights from real market research and the Side Hustle Viability Scorecard (Project 1 of this portfolio).

## How I Used AI
Used AI to help structure the Streamlit app layout and debug the
label encoder integration. Model selection, feature importance
interpretation, and all personalised recommendation logic are original.

# Project Directory Structure

Below is the organized file structure for the **willitwork-predictor** project:

```text
willitwork-predictor/
├── app.py                   # Main application code
├── requirements.txt         # Python package dependencies
├── willitwork_model.pkl     # Trained Random Forest model
├── label_encoders.pkl       # Label encoders for categories
├── feature_cols.pkl         # Feature column names
├── images/
│   └── feature_importance.png
└── README.md                # Project documentation

## Data Privacy
This app does **not** collect or store any of your data.  
All inputs are processed locally in your browser session and immediately discarded after prediction.


## About The Creator
Built by **Lungile Zulu** as part of a Data Science Portfolio.  
This project demonstrates:

- Machine Learning model deployment  
- Interactive web app development with **Streamlit**  
- User-friendly interface design for non-technical users  


## Portfolio Context
This app is **Project 3 of 3** in the Data Science portfolio:

1. **Side Hustle Viability Scorecard**  
2. **Digital Marketing Autopsy**  
3. **WillItWork? Success Predictor** 


## Known Limitations
- Model trained on **synthetic data**, not real business outcomes  
- Results are **predictions, not guarantees**  
- Works best for **online, digital-first side hustles**  
- Accuracy may vary outside the **South African market context**  


## Future Improvements
- Add more business models and categories  
- Improve model with **real business outcome data**  
- Add **export functionality** for results  
- Add **comparison feature** between different business models  


## License
This project is for **portfolio and educational purposes** only.


## Tech Stack
- **Streamlit**  
- **Python**  
- **Random Forest**


## Technical Requirements

To run this app locally, you need:

```bash
Python 3.9 or higher

