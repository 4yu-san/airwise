# Airwise: Air Quality Monitoring, Analysis and Prediction (DWM Mini Project)

Flask + MySQL star-schema warehouse with ETL, OLAP, data mining and an interactive dashboard.

## Pipeline
Data sources (CSV) -> ETL (`/api/etl/upload`) -> MySQL warehouse -> Star schema -> OLAP -> Data mining -> Prediction -> Dashboard

## Star schema
Fact table `fact_air_quality` (pollutant readings, AQI) linked to `dim_date`, `dim_location`, `dim_pollutant`.
The schema is drawn on the Data Warehouse page.

## Features
- Login (hashed password), dashboard with 5 Chart.js charts, Leaflet map, filters (city, category, date range), dark mode
- ETL: validates columns, fills missing values with the city median, removes duplicates, calculates AQI, loads MySQL
- OLAP: roll-up, drill-down, slice, dice with chart + table
- Mining: K-Means (choose K), Decision Tree (accuracy, precision, recall, F1, confusion matrix), Linear Regression with animated gauge

## Setup
```
pip install -r requirements.txt
python generate_dataset.py        # optional: regenerates data/air_quality.csv and database.sql
mysql -u root -p < database.sql   # creates air_quality_dw with 5,848 sample rows
cp .env.example .env              # put your MySQL password in .env
python app.py                     # open http://127.0.0.1:5000  (admin / admin123)
```
If MySQL is not reachable the app reads `data/air_quality.csv` and the Dashboard shows which source is in use.

## AQI and categories
AQI = max of the PM2.5 and PM10 sub-indices (CPCB-style breakpoints). Good <=50, Moderate <=100, Poor <=200, Very Poor <=300, Severe >300.

## Notes
- The dataset is synthetic (seasonal and city-specific patterns), not measured data.
- The decision tree scores very high because the category is derived from PM2.5/PM10; mention this in your viva.

## Testing the filters
```
python test_filters.py              # uses MySQL if reachable, otherwise the CSV fallback
FORCE_CSV=1 python test_filters.py  # force the CSV fallback
```
39 checks: all 8 cities, city + date, All Cities, name variants (case, spaces, URL encoding), empty results, bad input, OLAP slice and dice.
The active source is shown in the top bar ("MySQL Data Warehouse" or "CSV Fallback") and logged in the terminal.
