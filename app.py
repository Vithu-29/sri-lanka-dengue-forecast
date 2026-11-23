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

    test = pd.read_csv(ASSETS / "test_results.csv", parse_dates=["week_start"])
    scores = pd.read_csv(ASSETS / "model_summary.csv", index_col=0)
    return history.sort_index(), test, scores


models = load_models()
history, test, scores = load_data()

st.title("Weekly dengue forecasts in Sri Lanka")
district = st.selectbox("Choose a district", sorted(history.columns))

# Forecast the week immediately after the last report in the dataset.
target_week = history.index.max() + pd.Timedelta(weeks=1)
previous_weeks = [target_week - pd.Timedelta(weeks=n) for n in range(1, 5)]
recent_counts = history[district].reindex(previous_weeks)

st.subheader(f"Random Forest forecast for {district}")
st.write(f"Week starting {target_week:%d %B %Y}")

if recent_counts.isna().any():
    st.info("A weekly count needed for this forecast is missing.")
else:
    inputs = {"district": district, "month": target_week.month}
    for n, count in enumerate(recent_counts, start=1):
        inputs[f"lag_{n}"] = count

    row = pd.DataFrame([inputs], columns=models["features"])
    forecast = models["rf_model"].predict(row)[0]
    st.metric("Predicted reported cases", f"{forecast:.0f}")

st.subheader(f"Reported cases and forecasts in {district}")
st.caption("These are one-week-ahead forecasts from the final test period.")

district_test = test.loc[test["district"] == district].set_index("week_start")
chart = district_test[["actual_cases", "rf_prediction", "xgb_prediction"]].rename(
    columns={
        "actual_cases": "Reported cases",
        "rf_prediction": "Random Forest",
        "xgb_prediction": "XGBoost",
    }
)
st.line_chart(chart.sort_index())

st.subheader(f"Which model was more accurate in {district}?")
st.caption(
    f"Average forecast error across {len(district_test)} test weeks. "
    "Lower is better."
)

actual = district_test["actual_cases"]
rf_mae = (district_test["rf_prediction"] - actual).abs().mean()
xgb_mae = (district_test["xgb_prediction"] - actual).abs().mean()

left, right = st.columns(2)
left.metric("Random Forest error", f"{rf_mae:.2f} cases")
right.metric("XGBoost error", f"{xgb_mae:.2f} cases")

gap = abs(rf_mae - xgb_mae)
if round(gap, 2) == 0:
    st.write("The two models had the same average error to two decimal places.")
else:
    better_model = "Random Forest" if rf_mae < xgb_mae else "XGBoost"
    st.write(
        f"{better_model} had the lower average error in {district}, "
        f"by {gap:.2f} cases per week."
    )

st.subheader("Model comparison across all 25 districts")
st.caption(
    "Mean absolute error is the average number of cases a forecast missed by. "
    "Lower values are better."
)
st.dataframe(scores.style.format("{:.2f}"))