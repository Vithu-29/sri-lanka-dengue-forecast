# Sri Lanka Dengue Forecast

## Project question
Can previous weekly dengue case counts help predict the following week's
reported cases in Colombo?

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