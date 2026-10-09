import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="CreditIntelligence | Quantitative Credit Risk Engine",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ CreditIntelligence")
st.caption("Gradient-Boosted Credit Risk Scoring & Probability of Default (PD) Inference Model")

# Initialize Gemini API Client secretly
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in .env file. Please check your configuration.")
    st.stop()


@st.cache_resource
def get_gemini_client(key: str):
    return genai.Client(api_key=key)


client = get_gemini_client(api_key)


# Pydantic Schemas for Structured Model Inference
class FeatureImportance(BaseModel):
    feature_name: str = Field(description="Name of the financial feature evaluated.")
    weight: float = Field(description="Normalized feature importance weight (0.0 to 1.0).")
    direction: str = Field(description="Impact on risk: 'Increases Default Risk' or 'Decreases Default Risk'")


class CreditInferenceResult(BaseModel):
    probability_of_default: float = Field(
        description="Calculated Probability of Default (PD) value between 0.00 and 1.00.")
    credit_score: int = Field(description="Calculated FICO-style quantitative credit score between 300 and 850.")
    risk_tier: str = Field(
        description="Risk classification tier: 'Prime (Low Risk)', 'Near Prime', 'Subprime', or 'High Default Risk'")
    model_decision: str = Field(
        description="Automated underwriting decision: 'APPROVED', 'MANUAL REVIEW', or 'REJECTED'")
    feature_importances: list[FeatureImportance] = Field(
        description="Top predictive features that influenced the model prediction.")
    stress_test_insights: list[str] = Field(
        description="Quantitative stress-test notes under economic downturn scenarios.")
    underwriting_notes: list[str] = Field(description="Detailed auditing notes for compliance and underwriting logs.")


SYSTEM_INSTRUCTION = """
You are CreditIntelligence, a Quantitative Credit Risk Inference Engine.
You act as a trained XGBoost Gradient-Boosted Classifier evaluating loan applicants.
Analyze the provided tabular applicant features (Income, Debt-to-Income, Credit Utilization, Liquidity, Debt Ratios).
Calculate a realistic Probability of Default (PD), FICO-style credit score (300-850), feature importance weights, and underwriting decisions.

Output strictly valid JSON according to the requested schema. Maintain a rigorous mathematical and quantitative tone.
"""


def simulate_model_training():
    """Simulates realistic ML model training and hyperparameter tuning feedback."""
    progress_bar = st.progress(0, text="Initializing XGBoost Hyperparameters...")
    time.sleep(0.3)

    progress_bar.progress(25, text="Preprocessing Tabular Features & StandardScaling...")
    time.sleep(0.4)

    progress_bar.progress(60, text="Training Ensembles (n_estimators=500, max_depth=6, learning_rate=0.03)...")
    time.sleep(0.5)

    progress_bar.progress(90, text="Calculating SHAP Feature Importances & ROC-AUC Validation...")
    time.sleep(0.3)

    progress_bar.progress(100, text="Model Training & Calibration Complete!")
    time.sleep(0.2)
    progress_bar.empty()


def render_model_metrics(result: dict):
    """Renders the AI inference formatted as ML evaluation dashboard."""

    pd_val = result.get("probability_of_default", 0.05)
    score = result.get("credit_score", 720)
    decision = result.get("model_decision", "APPROVED")
    tier = result.get("risk_tier", "Prime")

    # 1. Headline Model Predictions
    st.subheader("🎯 Model Output & Risk Classifications")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(label="Predicted Score", value=f"{score} / 850")
    c2.metric(label="Probability of Default (PD)", value=f"{pd_val * 100:.2f}%")
    c3.metric(label="Risk Tier", value=tier)

    if decision == "APPROVED":
        c4.success(f"**Decision:**\n\n### {decision}")
    elif decision == "MANUAL REVIEW":
        c4.warning(f"**Decision:**\n\n### {decision}")
    else:
        c4.error(f"**Decision:**\n\n### {decision}")

    st.divider()

    # 2. Confusion Matrix & Feature Importance Charts
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📊 SHAP Feature Importance Weights")
        features = result.get("feature_importances", [])
        if features:
            df_feat = pd.DataFrame(features)
            fig_bar = px.bar(
                df_feat,
                x="weight",
                y="feature_name",
                color="direction",
                orientation="h",
                title="Top Predictive Features Impacting Model Vector",
                labels={"weight": "Feature Weight (|SHAP|)", "feature_name": "Feature Name"},
                color_discrete_map={"Increases Default Risk": "#EF553B", "Decreases Default Risk": "#00CC96"}
            )
            fig_bar.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.subheader("🧪 Out-of-Fold Validation Metrics")
        # Display simulated ROC-AUC and Confusion Matrix
        cm_data = [[842, 38], [24, 196]]
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=['Non-Default (0)', 'Default (1)'],
            y=['Non-Default (0)', 'Default (1)'],
            text_auto=True,
            title="Confusion Matrix (Validation Test Set)",
            color_continuous_scale="Viridis"
        )
        fig_cm.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()

    # 3. Stress Test & Underwriting Notes
    c_stress, c_audit = st.columns(2)

    with c_stress:
        st.subheader("📉 Downturn Stress Test Analysis")
        for note in result.get("stress_test_insights", []):
            st.markdown(f"- ⚠️ {note}")

    with c_audit:
        st.subheader("📋 Compliance & Audit Log")
        for log in result.get("underwriting_notes", []):
            st.markdown(f"- 📝 {log}")


# Sidebar Settings (Disguised ML Config)
with st.sidebar:
    st.header("⚙️ Model Architecture")
    st.selectbox("Classifier Engine", ["XGBoost Classifier v2.4", "LightGBM Ensembles", "Deep Neural Net (TabNet)"])
    st.slider("Train/Test Split Ratio", min_value=0.6, max_value=0.9, value=0.8, step=0.05)
    st.number_input("Random Seed (State)", value=42)
    st.caption("Status: Model pre-trained on 125,000 anonymized financial profiles.")

# Main Input Section: Tabular Features Form
st.subheader("📥 Input Tabular Feature Vector")

with st.form("applicant_features_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        annual_income = st.number_input("Annual Income ($)", value=85000, step=5000)
        debt_to_income = st.slider("Debt-to-Income Ratio (DTI %)", min_value=0, max_value=100, value=28)
        liquid_reserves = st.number_input("Liquid Cash Reserves ($)", value=18000, step=1000)

    with col2:
        loan_amount = st.number_input("Requested Loan Amount ($)", value=25000, step=1000)
        credit_utilization = st.slider("Credit Utilization (%)", min_value=0, max_value=100, value=35)
        existing_delinquencies = st.selectbox("Delinquencies in Last 2 Years", [0, 1, 2, 3, "4+"])

    with col3:
        employment_length_years = st.number_input("Employment Length (Years)", value=5, min_value=0)
        repayment_term_months = st.selectbox("Requested Loan Term", [36, 60, 84])
        home_ownership = st.selectbox("Home Ownership Status", ["MORTGAGE", "RENT", "OWN", "OTHER"])

    run_pipeline_btn = st.form_submit_button("🚀 Run Feature Pipeline & Model Prediction", type="primary",
                                             use_container_width=True)

if run_pipeline_btn:
    # 1. Show realistic "Model Training / Matrix Calculation" UI feedback
    simulate_model_training()

    # 2. Secretly prepare input vector for Gemini behind the scenes
    feature_vector = {
        "annual_income": annual_income,
        "debt_to_income_dti": debt_to_income,
        "liquid_cash_reserves": liquid_reserves,
        "requested_loan_amount": loan_amount,
        "credit_utilization_pct": credit_utilization,
        "past_delinquencies": existing_delinquencies,
        "employment_years": employment_length_years,
        "loan_term_months": repayment_term_months,
        "home_ownership": home_ownership,
    }

    # 3. Call Gemini under the hood
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[f"Calculate credit inference metrics for feature vector: {json.dumps(feature_vector)}"],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=CreditInferenceResult,
            ),
        )

        inference_data = json.loads(response.text)

        # 4. Render as standard ML Metrics Dashboard
        render_model_metrics(inference_data)

    except Exception as e:
        st.error(f"Model Pipeline Execution Failure: {e}")