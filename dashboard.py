
"""
dashboard.py — Streamlit Operations Dashboard for the Predictive
Maintenance Platform. Reads the CSVs already produced by the pipeline
(health scores, failure modes, recommendations) and visualizes them.
"""
 
import streamlit as st
import pandas as pd
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
 
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
 
st.title("🛠️ Predictive Maintenance & Equipment Health Dashboard")
st.caption("NASA C-MAPSS FD001 — Turbofan Engine Fleet")
 
# --- Load data ---
@st.cache_data
def load_data():
    recs = pd.read_csv(PROCESSED_DIR / "FD001_maintenance_recommendations.csv")
    return recs
 
df = load_data()
 
# --- Top-level metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Engines", len(df))
col2.metric("Critical", int((df["risk_level"] == "Critical").sum()))
col3.metric("High Risk", int((df["risk_level"] == "High Risk").sum()))
col4.metric("Healthy", int((df["risk_level"] == "Healthy").sum()))
 
st.divider()
 
# --- Risk level distribution ---
left, right = st.columns(2)
 
with left:
    st.subheader("Risk Level Distribution")
    risk_counts = df["risk_level"].value_counts()
    st.bar_chart(risk_counts)
 
with right:
    st.subheader("Recommended Action Distribution")
    action_counts = df["recommended_action"].value_counts()
    st.bar_chart(action_counts)
 
st.divider()
 
# --- Health Score distribution ---
st.subheader("Health Score Distribution Across Fleet")
st.bar_chart(df.set_index("unit_number")["health_score"])
 
st.divider()
 
# --- Full table, filterable ---
st.subheader("Engine-Level Details")
risk_filter = st.multiselect(
    "Filter by risk level",
    options=df["risk_level"].unique(),
    default=df["risk_level"].unique(),
)
filtered = df[df["risk_level"].isin(risk_filter)].sort_values("health_score")
st.dataframe(filtered, width="stretch")
 
# ============================================================
# LIVE PREDICTION: Pick an Engine ID OR upload new sensor data
# ============================================================
st.divider()
st.header("Get a Live Prediction")
 
import joblib
 
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_rul_model_latest.joblib"
TEST_FEATURES_PATH = PROCESSED_DIR / "FD001_test_features.parquet"
RUL_CAP = 125
 
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_cols"]
 
@st.cache_data
def load_test_features():
    """Load the pre-built test feature set so users can pick an existing
    engine by its unit_number instead of uploading a CSV."""
    return pd.read_parquet(TEST_FEATURES_PATH)
 
def rul_to_health_score(rul):
    score = 100 * (rul / RUL_CAP)
    return max(0, min(100, round(score, 1)))
 
def risk_level(score):
    if score >= 70:
        return "Healthy"
    elif score >= 40:
        return "Moderate Risk"
    elif score >= 15:
        return "High Risk"
    return "Critical"
 
def run_prediction(rows_df, model, feature_cols):
    """Run the model on rows_df and return a display-ready dataframe."""
    missing = set(feature_cols) - set(rows_df.columns)
    if missing:
        st.error(f"Data is missing required columns: {sorted(missing)}")
        return None
 
    preds = model.predict(rows_df[feature_cols])
    result = rows_df.copy()
    result["predicted_RUL"] = preds
    result["health_score"] = result["predicted_RUL"].apply(rul_to_health_score)
    result["risk_level"] = result["health_score"].apply(risk_level)
 
    display_cols = ["predicted_RUL", "health_score", "risk_level"]
    if "unit_number" in result.columns:
        display_cols = ["unit_number"] + display_cols
    return result[display_cols].sort_values("health_score")
 
 
input_mode = st.radio(
    "How would you like to get a prediction?",
    ["Select Engine ID", "Upload CSV"],
    horizontal=True,
)
 
# ---------- Option 1: pick an existing engine by ID ----------
if input_mode == "Select Engine ID":
    try:
        model, feature_cols = load_model()
        test_features = load_test_features()
 
        if "unit_number" not in test_features.columns:
            st.error("Test feature file has no 'unit_number' column to select by.")
        else:
            unit_ids = sorted(test_features["unit_number"].unique())
            selected_id = st.selectbox("Select Engine / Unit ID", unit_ids)
 
            row = test_features[test_features["unit_number"] == selected_id]
 
            if st.button("Predict"):
                result = run_prediction(row, model, feature_cols)
                if result is not None:
                    st.success(f"Prediction for Engine {selected_id}:")
                    st.dataframe(result, width="stretch")
 
                    csv_out = result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download prediction as CSV",
                        data=csv_out,
                        file_name=f"engine_{selected_id}_prediction.csv",
                        mime="text/csv",
                    )
    except FileNotFoundError:
        st.error(
            f"Test feature file not found at {TEST_FEATURES_PATH}. "
            "Make sure FD001_test_features.parquet exists in data/processed/."
        )
    except Exception as e:
        st.error(f"Error running prediction: {e}")
 
# ---------- Option 2: upload a CSV of new sensor data ----------
else:
    st.markdown(
        "Upload a CSV with the same 72 feature columns the model was trained on "
        "(e.g. export a few rows from FD001_test_features.parquet). "
        "The app will predict RUL, Health Score, and Risk Level for each row."
    )
    uploaded_file = st.file_uploader("Upload sensor feature CSV", type=["csv"])
 
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            model, feature_cols = load_model()
 
            result = run_prediction(new_data, model, feature_cols)
            if result is not None:
                st.success(f"Predicted {len(result)} rows successfully.")
                st.dataframe(result, width="stretch")
 
                csv_out = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions as CSV",
                    data=csv_out,
                    file_name="new_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")