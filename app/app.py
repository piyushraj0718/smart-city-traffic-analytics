import streamlit as st
import pandas as pd
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go


# PAGE CONFIG
st.set_page_config(
    page_title="Smart City Traffic Intelligence",
    layout="wide",
    page_icon="🚦"
)


# CUSTOM DARK STYLING

st.markdown("""
<style>

/* ===== Deep Navy Premium Background ===== */
.stApp {
    background: linear-gradient(180deg, #0B1426 0%, #0F1C33 100%);
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1120 0%, #0E1A2F 100%);
}

/* Metric Card Base Style */
.metric-card {
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    color: #FFFFFF;
    background: rgba(20, 30, 50, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.05);
}

/* Subtle elevation */
.metric-card h2 {
    margin: 5px 0;
}

/* Headings */
h1, h2, h3 {
    color: #E6EDF7;
}

/* Hide default footer */
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# PATHS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "ml_ready_accidents.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_models", "best_model.pkl")
FORECAST_PATH = os.path.join(BASE_DIR, "reports", "forecast", "30_day_forecast.csv")
HEATMAP_PATH = os.path.join(BASE_DIR, "reports", "heatmap", "advanced_accident_heatmap.html")


# LOAD DATA

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()


# SIDEBAR NAVIGATION

st.sidebar.title("🚦 Smart City Dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["Executive Overview", "ML Prediction", "Risk Heatmap", "Forecast"]
)


# PAGE 1 — EXECUTIVE OVERVIEW

if page == "Executive Overview":

    st.title("🚦 Smart City Traffic Intelligence Overview")
    st.markdown("Data-driven national accident analytics for strategic planning.")

    total_accidents = len(df)
    high_severity = (df["Severity_Label"] == "High").mean() * 100
    avg_daily = df.groupby("Date").size().mean()
    risky_time = df["Time_Category"].value_counts().idxmax()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Accidents", f"{total_accidents:,}")
    col2.metric("High Severity %", f"{high_severity:.2f}%")
    col3.metric("Avg Daily Incidents", f"{avg_daily:.0f}")
    col4.metric("Highest Risk Time", risky_time)

    st.markdown("---")

    daily = df.groupby("Date").size().reset_index(name="Accidents")

    trend_fig = px.line(
        daily,
        x="Date",
        y="Accidents",
        template="plotly_dark",
        title="National Accident Trend (Historical)"
    )

    st.plotly_chart(trend_fig, use_container_width=True)

    st.markdown("---")

    severity_fig = px.pie(
        df,
        names="Severity_Label",
        template="plotly_dark",
        title="Accident Severity Distribution"
    )

    st.plotly_chart(severity_fig, use_container_width=True)



# PAGE 2 — ML PREDICTION

elif page == "ML Prediction":

    st.title("🤖 Accident Severity Prediction Engine")

    model = joblib.load(MODEL_PATH)

    st.markdown("Adjust parameters to simulate accident conditions.")

    col1, col2 = st.columns(2)

    vehicles = col1.number_input("Number of Vehicles", 1, 10, 2)
    casualties = col1.number_input("Number of Casualties", 0, 20, 1)
    speed_limit = col1.selectbox("Speed Limit (mph)", [20, 30, 40, 50, 60, 70])

    weather_index = col2.slider("Weather Severity Index", 1, 5, 1)
    road_risk = col2.slider("Road Risk Score", 1, 5, 3)
    weekend = col2.selectbox("Is Weekend?", [0, 1])

    if st.button("🚀 Predict Severity"):

        input_df = pd.DataFrame([{
    "Number_of_Vehicles": vehicles,
    "Number_of_Casualties": casualties,
    "Speed_limit": speed_limit,
    "Weather_Severity_Index": weather_index,
    "Road_Risk_Score": road_risk,
    "Is_Weekend": weekend
}])

        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]

        severity_map = {0: "High", 1: "Low", 2: "Medium"}

        st.success(f"Predicted Severity: **{severity_map.get(prediction, prediction)}**")

        prob_df = pd.DataFrame({
            "Severity": ["High", "Low", "Medium"],
            "Probability": probs
        })

        prob_fig = px.bar(
            prob_df,
            x="Severity",
            y="Probability",
            template="plotly_dark",
            title="Prediction Confidence Distribution",
            text_auto=True
        )

        st.plotly_chart(prob_fig, use_container_width=True)



# PAGE 3 — HEATMAP

elif page == "Risk Heatmap":

    st.title("🗺️ National Accident Risk Heatmap")

    if os.path.exists(HEATMAP_PATH):
        with open(HEATMAP_PATH, "r", encoding="utf-8") as f:
            heatmap_html = f.read()
        st.components.v1.html(heatmap_html, height=650)
    else:
        st.warning("Heatmap file not found. Generate it first.")



# PAGE 4 — FORECAST

elif page == "Forecast":

    st.title("📈 30-Day Accident Forecast")

    if os.path.exists(FORECAST_PATH):
        forecast = pd.read_csv(FORECAST_PATH)

        forecast["ds"] = pd.to_datetime(forecast["ds"])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(width=3)
        ))

        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="Confidence Interval"
        ))

        fig.update_layout(
            template="plotly_dark",
            title="Accident Forecast Projection with Confidence Band",
            xaxis_title="Date",
            yaxis_title="Predicted Daily Accidents"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🔎 Next 30-Day Forecast Summary")
        st.dataframe(forecast.tail(30))

    else:
        st.warning("Forecast file not found. Run forecasting module first.")


# FOOTER

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #9CA3AF; font-size: 14px;'>
        🚦 Smart City Traffic & Accident Risk Analytics System<br>
        Developed & Engineered by <b>Piyush Raj</b><br>
        <span style='font-size:12px;'>Python | XGBoost | SHAP | Prophet | Streamlit</span>
    </div>
    """,
    unsafe_allow_html=True
)