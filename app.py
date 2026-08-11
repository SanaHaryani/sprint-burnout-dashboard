import streamlit as st
import joblib
import numpy as np
import pandas as pd

# PAGE CONFIG

st.set_page_config(
    page_title="AI-Based Agile Sprint Planning Dashboard",
    page_icon="📊",
    layout="centered",
)

# LOADING MODELS, SCALER, AND FEATURE COLUMN ORDER


@st.cache_resource
def load_artifacts():
    regressor = joblib.load("sprint_capacity_model.pkl")
    classifier = joblib.load("burnout_classifier_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return regressor, classifier, scaler, feature_cols


try:
    regressor, classifier, scaler, feature_cols = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Could not find one of the required model files "
        "(sprint_capacity_model.pkl, burnout_classifier_model.pkl, "
        "scaler.pkl, feature_columns.pkl). Make sure all four files are "
        "in the same folder as app.py."
    )
    st.stop()

# TITLE DISPLAY

st.title("AI-Based Agile Sprint Planning Dashboard")
st.write(
    "Predict sprint capacity and burnout risk from team and workload "
    "variables, using the trained models from Chapters 3-4 of the thesis."
)


# USER INPUTS ( 8 base variables from the final feature set)

st.subheader("Team and Sprint Inputs")

col1, col2 = st.columns(2)

with col1:
    team_size = st.slider("Team Size", 3, 10, 6)
    team_experience = st.slider("Team Experience (years, average)", 1, 10, 5)
    sprint_duration = st.select_slider(
        "Sprint Duration (weeks)", options=[2, 3, 4], value=2)
    avg_hours_per_week = st.slider(
        "Average Working Hours Per Week", 30, 60, 45)

with col2:
    tasks_assigned = st.slider("Tasks Assigned", 5, 30, 16)
    avg_task_complexity = st.slider("Average Task Complexity (1-5)", 1, 5, 3)
    past_velocity = st.slider("Past Velocity", 10, 60, 30)
    deadlines_missed = st.slider("Deadlines Missed (last sprint)", 0, 5, 2)


# FEATURE ENGINEERING 

workload_per_person = tasks_assigned / team_size
velocity_efficiency = past_velocity / (team_experience + 1)
input_row = pd.DataFrame([{
    "team_size": team_size,
    "team_experience": team_experience,
    "sprint_duration": sprint_duration,
    "avg_hours_per_week": avg_hours_per_week,
    "tasks_assigned": tasks_assigned,
    "avg_task_complexity": avg_task_complexity,
    "past_velocity": past_velocity,
    "deadlines_missed": deadlines_missed,
    "workload_per_person": workload_per_person,
    "velocity_efficiency": velocity_efficiency,
}])

# PREDICTIONS

st.subheader("Generate Predictions")

if st.button("Generate Predictions", type="primary"):
    # CRITICAL: apply the same StandardScaler used during training before
    # calling predict(). Skipping this step is the single most common cause
    # of a dashboard silently producing wrong predictions with no error.
    scaled_input = pd.DataFrame(
        scaler.transform(input_row), columns=feature_cols)

    sprint_prediction = regressor.predict(scaled_input)[0]
    burnout_prediction = classifier.predict(scaled_input)[0]
    burnout_proba = classifier.predict_proba(scaled_input)[0]
    proba_by_class = dict(zip(classifier.classes_, burnout_proba))

    st.markdown("### Results")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Predicted Sprint Capacity", f"{round(sprint_prediction)}")
    with res_col2:
        risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        st.metric(
            "Predicted Burnout Risk",
            f"{risk_colors.get(burnout_prediction, '')} {burnout_prediction}"
        )

    st.markdown("#### Burnout Risk Probability Breakdown")
    proba_df = pd.DataFrame({
        "Risk Level": list(proba_by_class.keys()),
        "Probability": [f"{v:.1%}" for v in proba_by_class.values()],
    })
    st.dataframe(proba_df, hide_index=True, use_container_width=True)

    st.caption(
        "These predictions come from Linear Regression (sprint capacity) and "
        "Logistic Regression (burnout risk) — the best-performing models "
        "identified in Chapter 4's model comparison, trained on 1,000 "
        "synthetic sprint observations. See Chapter 4 for full evaluation "
        "metrics and Chapter 5 for discussion of what these predictions do "
        "and do not mean in practice."
    )
else:
    st.info("Adjust the sliders above, then click **Generate Predictions**.")
