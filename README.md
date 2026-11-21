# Sri Lanka Dengue Forecast

## Project question
How accurately can we forecast next week's reported dengue cases for each
district in Sri Lanka using historical weekly dengue reports?

The dashboard will let users select a district, view its reported case
history, and compare forecasts from Random Forest, XGBoost and stacking.

## Data
I plan to use the `srilanka_weekly_data` dataset from the `denguedatahub`
project. It is compiled from Sri Lanka's Weekly Epidemiological Reports.
I will check the available dates, district names and missing weeks before
selecting the final study period.

## Planned comparison
I will first try a simple forecast: next week's count equals this week's
count. I will then compare Random Forest, XGBoost and stacking against
that baseline, training on earlier weeks and testing on later weeks.

## Tools
Google Colab for the Python notebook and VS Code for the dashboard.