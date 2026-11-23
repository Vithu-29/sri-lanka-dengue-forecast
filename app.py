from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ASSETS = Path(__file__).resolve().parent / "dashboard_assets"

st.set_page_config(page_title="Sri Lanka dengue forecasts", layout="wide")


@st.cache_resource
def load_models():
    return joblib.load(ASSETS / "models.joblib")


@st.cache_data
def load_data():
    history = pd.read_csv(ASSETS / "district_cases.csv", index_col=0)
    history.index = pd.to_datetime(history.index)

    test = pd.read_csv(
        ASSETS / "test_results.csv",
        parse_dates=["week_start"],
    )
    scores = pd.read_csv(ASSETS / "model_summary.csv", index_col=0)

    return history.sort_index(), test, scores


models = load_models()
history, test, scores = load_data()

st.title("Weekly dengue forecasts in Sri Lanka")
district = st.selectbox("Choose a district", sorted(history.columns))

# Use the four exact weeks before the forecast date, in the same order as training.
target_week = history.index.max() + pd.Timedelta(weeks=1)
previous_weeks = [target_week - pd.Timedelta(weeks=n) for n in range(1, 5)]
recent_counts = history[district].reindex(previous_weeks)

st.subheader(f"Random Forest forecast for {district}")

if recent_counts.isna().any():
    st.info("A weekly count needed for this forecast is missing.")
else:
    inputs = {"district": district, "month": target_week.month}
    for n, count in enumerate(recent_counts, start=1):
        inputs[f"lag_{n}"] = count

    row = pd.DataFrame([inputs], columns=models["features"])
    forecast = models["rf_model"].predict(row)[0]

    reported_card, forecast_card = st.columns(2)
    reported_card.metric(
        label=f"Reported · week of {previous_weeks[0]:%d %b %Y}",
        value=f"{recent_counts.iloc[0]:,.0f} cases",
        border=True,
    )
    forecast_card.metric(
        label=f"Random Forest forecast · week of {target_week:%d %b %Y}",
        value=f"{forecast:,.0f} cases",
        border=True,
    )

st.subheader(f"Reported cases and forecasts in {district}")
st.caption("Each forecast was made one week ahead during the final test period.")

district_test = (
    test.loc[test["district"] == district]
    .set_index("week_start")
    .sort_index()
)

chart = district_test[["actual_cases", "rf_prediction", "xgb_prediction"]].rename(
    columns={
        "actual_cases": "Reported cases",
        "rf_prediction": "Random Forest",
        "xgb_prediction": "XGBoost",
    }
)

st.line_chart(
    chart,
    y=["Reported cases", "Random Forest", "XGBoost"],
    color=["#172F3D", "#0F766E", "#BC5A12"],
    x_label="Reporting week",
    y_label="Weekly dengue cases",
    height=360,
)

st.subheader(f"Which model was more accurate in {district}?")
st.caption(
    f"Average absolute error across {len(district_test)} test weeks. "
    "Lower is better."
)

actual = district_test["actual_cases"]
rf_mae = (district_test["rf_prediction"] - actual).abs().mean()
xgb_mae = (district_test["xgb_prediction"] - actual).abs().mean()

rf_display = f"{rf_mae:.2f}"
xgb_display = f"{xgb_mae:.2f}"

left, right = st.columns(2)
left.metric("Random Forest error", f"{rf_display} cases", border=True)
right.metric("XGBoost error", f"{xgb_display} cases", border=True)

if rf_display == xgb_display:
    st.write("Both models had the same average error when rounded to two decimal places.")
else:
    better_model = "Random Forest" if rf_mae < xgb_mae else "XGBoost"
    st.write(f"{better_model} had the lower average error in {district}.")

st.subheader("Model comparison across all 25 districts")
st.caption(
    "Mean absolute error (MAE) measures how far predictions were from reported "
    "counts, on average, in cases per district-week. Lower values are better."
)
st.dataframe(scores.style.format("{:.2f}"))