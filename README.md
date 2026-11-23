# Sri Lanka Dengue Forecast

How accurately can we forecast next week’s reported dengue cases for each district in Sri Lanka? This project uses weekly dengue reports from all 25 districts to compare six forecasting methods. A Streamlit dashboard runs a trained Random Forest and lets users explore how its past forecasts compared with XGBoost and the reported counts.

**[Open the dashboard](https://sri-lanka-dengue-forecast.streamlit.app/)**

## Try the dashboard

1. Choose a district, such as Colombo. The two cards show its latest reported count and the Random Forest forecast for the following week. In the project dataset, that forecast is for the week starting **14 September 2026**.
2. Read the line chart to see when Random Forest and XGBoost tracked or missed changes during the final test period. The error cards below the chart compare the two models for the selected district.
3. Switch districts to explore local differences. The table at the bottom summarises all six methods across the country.

## Run locally

Clone this repository or download and extract its ZIP. Open the folder containing `app.py` and the root `requirements.txt` in VS Code. The dashboard loads its trained model and CSV files from `dashboard_assets/`. If you have those files, you can skip model training.

Use the Python major and minor version printed by the final Colab export cell. The commands below assume that your selected Python matches that version. On Windows, for example, if Colab printed `3.12.x`, use `py -3.12 -m venv .venv` for the first command.

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS or Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually <http://localhost:8501>. Press `Ctrl+C` in the terminal to stop the app.

## Data and forecasting method

The notebook reads the `srilanka_weekly_data` dataset from **denguedatahub**, compiled from Sri Lanka’s Weekly Epidemiological Reports. The analysed snapshot contains 26,753 source rows and 1,029 reporting weeks, with week start dates from 23 December 2006 to 7 September 2026. The notebook combines the source labels `Ampara` and `Kalmune` to produce one series for each of the 25 districts. [Download the source CSV](https://thiyangt.r-universe.dev/denguedatahub/data/srilanka_weekly_data/csv)

Random Forest and XGBoost use six inputs per forecast: the district, the target month and the reported counts exactly one, two, three and four weeks earlier. Training uses earlier weeks. Validation and final testing each cover 52 later reporting weeks. For each one-week-ahead forecast, a model can use counts already reported during the preceding weeks. A district-week is scored only when the observed target and required inputs are available.

## Results

Mean absolute error (MAE) measures how far a forecast was from the reported count, in cases per district-week. Lower values indicate smaller errors. Validation covers **1,300** district-week forecasts. The final test covers **1,195** forecasts shared by all six methods.

| Method | How it forecasts | Validation MAE | Final test MAE |
|---|---|---:|---:|
| Previous-week baseline | Repeats the previous week’s count | 9.00 | 29.72 |
| Random Forest | Averages predictions from decision trees | 8.49 | 29.19 |
| XGBoost | Builds trees in sequence to reduce prediction error | **8.14** | 32.21 |
| Stacking | Learns a linear combination of Random Forest and XGBoost forecasts | 8.86 | 30.06 |
| ARIMA | Uses a time-series model fitted separately for each district | 8.59 | 28.54 |
| Random Forest + ARIMA | Averages the two forecasts with equal weights | 8.32 | **28.09** |

**XGBoost was selected on validation.** In the later test period, the Random Forest + ARIMA hybrid had the smallest error, and all six methods had larger errors than during validation. The ranking therefore depends on the period evaluated. The dashboard’s interactive forecast demonstrates Random Forest, while its historical comparison uses the saved test predictions.

## Reproduce the analysis in Google Colab

1. Open the project’s dengue analysis notebook (`.ipynb`) in Google Colab.
2. Choose **Runtime → Run all**. The notebook downloads the source data, prepares district-week records, trains the models and calculates the validation and test scores. An updated source CSV may produce different dates and results.
3. Run the final export cell to download `dashboard_assets.zip`. To run the app with that export, extract its contents directly into `dashboard_assets/` beside `app.py`.

| Dashboard asset | Purpose |
|---|---|
| `dashboard_assets/models.joblib` | Trained Random Forest pipeline and ordered input feature names |
| `dashboard_assets/district_cases.csv` | Weekly district counts used to build the forecast inputs |
| `dashboard_assets/test_results.csv` | Reported counts and historical model predictions for the test period |
| `dashboard_assets/model_summary.csv` | Validation and final test MAE for all six methods |

## Acknowledgements

The analysis was developed in Python using Google Colab, pandas, scikit-learn, XGBoost and statsmodels. The dashboard was developed in VS Code using Streamlit. ChatGPT assisted with drafting and debugging code, developing the dashboard, and explaining results. The reported scores come from the executed project notebook.
