"""Generates data/air_quality.csv (daily readings, 8 cities, 2024-2025) and database.sql."""
import numpy as np, pandas as pd, sys

CITIES = {  # city: (state, lat, lon, pm25 base, temp base, humidity base)
    "Mumbai": ("Maharashtra", 19.0760, 72.8777, 55, 29, 72),
    "Delhi": ("Delhi", 28.6139, 77.2090, 105, 25, 55),
    "Pune": ("Maharashtra", 18.5204, 73.8567, 48, 26, 58),
    "Bengaluru": ("Karnataka", 12.9716, 77.5946, 38, 24, 62),
    "Hyderabad": ("Telangana", 17.3850, 78.4867, 52, 27, 55),
    "Chennai": ("Tamil Nadu", 13.0827, 80.2707, 45, 30, 70),
    "Kolkata": ("West Bengal", 22.5726, 88.3639, 82, 27, 74),
    "Ahmedabad": ("Gujarat", 23.0225, 72.5714, 72, 28, 52),
}
POLLUTANTS = [(1, "PM2.5", "ug/m3", "Fine particles under 2.5 micrometres"),
              (2, "PM10", "ug/m3", "Coarse particles under 10 micrometres"),
              (3, "CO", "mg/m3", "Carbon monoxide from combustion"),
              (4, "NO2", "ug/m3", "Nitrogen dioxide from traffic and industry"),
              (5, "SO2", "ug/m3", "Sulphur dioxide from fuel burning"),
              (6, "O3", "ug/m3", "Ground-level ozone")]

PM25_BP = ([0, 30, 60, 90, 120, 250], [0, 50, 100, 200, 300, 400])
PM10_BP = ([0, 50, 100, 250, 350, 430], [0, 50, 100, 200, 300, 400])

def compute_aqi(pm25, pm10):
    a = np.interp(pm25, *PM25_BP); b = np.interp(pm10, *PM10_BP)
    return np.round(np.maximum(a, b)).astype(int)

def category(aqi):
    return pd.cut(aqi, [-1, 50, 100, 200, 300, 10000],
                  labels=["Good", "Moderate", "Poor", "Very Poor", "Severe"])

def generate(seed=42, start="2024-01-01", end="2025-12-31"):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end)
    rows = []
    for city, (st, lat, lon, pm_base, t_base, h_base) in CITIES.items():
        doy = days.dayofyear.values
        winter = np.cos((doy - 15) / 365 * 2 * np.pi)          # peaks in January
        monsoon = np.exp(-((doy - 200) / 40) ** 2)             # cleans air around July
        pm25 = pm_base * (1 + 0.55 * winter - 0.5 * monsoon) * rng.lognormal(0, 0.22, len(days))
        pm25 = np.clip(pm25, 8, 330)
        pm10 = np.clip(pm25 * rng.normal(1.65, 0.15, len(days)), 15, 480)
        temp = t_base - 7 * winter + 2 * monsoon + rng.normal(0, 1.8, len(days))
        hum = np.clip(h_base + 18 * monsoon + 6 * winter * (city == "Delhi") + rng.normal(0, 7, len(days)), 15, 98)
        co = np.clip(0.5 + pm25 / 90 + rng.normal(0, 0.15, len(days)), 0.2, 6)
        no2 = np.clip(14 + pm25 * 0.32 + rng.normal(0, 6, len(days)), 4, 140)
        so2 = np.clip(6 + pm25 * 0.08 + rng.normal(0, 3, len(days)), 1, 60)
        o3 = np.clip(35 + (temp - 20) * 1.6 - pm25 * 0.1 + rng.normal(0, 8, len(days)), 8, 160)
        rows.append(pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "city": city, "state": st,
            "latitude": lat, "longitude": lon, "temperature": temp.round(1), "humidity": hum.round(1),
            "pm25": pm25.round(1), "pm10": pm10.round(1), "co": co.round(2), "no2": no2.round(1),
            "so2": so2.round(1), "o3": o3.round(1)}))
    df = pd.concat(rows, ignore_index=True)
    df["aqi"] = compute_aqi(df.pm25, df.pm10)
    return df

def dominant(row):
    subs = {1: np.interp(row.pm25, *PM25_BP), 2: np.interp(row.pm10, *PM10_BP),
            3: row.co * 20, 4: row.no2 * 1.2, 5: row.so2 * 1.5, 6: row.o3 * 0.9}
    return max(subs, key=subs.get)

def write_sql(df, path="database.sql"):
    cities = list(CITIES)
    d = pd.to_datetime(df.date.unique())
    with open(path, "w") as f:
        f.write("""CREATE DATABASE IF NOT EXISTS air_quality_dw;
USE air_quality_dw;
SET FOREIGN_KEY_CHECKS=0;
DROP TABLE IF EXISTS fact_air_quality, dim_date, dim_location, dim_pollutant, users;
SET FOREIGN_KEY_CHECKS=1;
CREATE TABLE dim_date (date_id INT PRIMARY KEY, full_date DATE NOT NULL UNIQUE, day TINYINT, month TINYINT,
  month_name VARCHAR(12), quarter TINYINT, year SMALLINT);
CREATE TABLE dim_location (location_id INT PRIMARY KEY AUTO_INCREMENT, city VARCHAR(50) NOT NULL UNIQUE,
  state VARCHAR(50), country VARCHAR(50) DEFAULT 'India', latitude DECIMAL(9,6), longitude DECIMAL(9,6));
CREATE TABLE dim_pollutant (pollutant_id INT PRIMARY KEY, pollutant_name VARCHAR(10) NOT NULL,
  unit VARCHAR(10), description VARCHAR(100));
CREATE TABLE fact_air_quality (measurement_id INT PRIMARY KEY AUTO_INCREMENT, date_id INT NOT NULL,
  location_id INT NOT NULL, pollutant_id INT NOT NULL, temperature FLOAT, humidity FLOAT, pm25 FLOAT, pm10 FLOAT,
  co FLOAT, no2 FLOAT, so2 FLOAT, o3 FLOAT, aqi INT,
  FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
  FOREIGN KEY (location_id) REFERENCES dim_location(location_id),
  FOREIGN KEY (pollutant_id) REFERENCES dim_pollutant(pollutant_id),
  UNIQUE KEY uq_day_city (date_id, location_id), INDEX idx_aqi (aqi), INDEX idx_loc (location_id));
CREATE TABLE users (id INT PRIMARY KEY AUTO_INCREMENT, username VARCHAR(50) UNIQUE, password_hash VARCHAR(255));
""")
        f.write("INSERT INTO dim_pollutant VALUES " + ",".join(
            "(%d,'%s','%s','%s')" % p for p in POLLUTANTS) + ";\n")
        f.write("INSERT INTO dim_location (location_id,city,state,latitude,longitude) VALUES " + ",".join(
            "(%d,'%s','%s',%s,%s)" % (i + 1, c, CITIES[c][0], CITIES[c][1], CITIES[c][2]) for i, c in enumerate(cities)) + ";\n")
        f.write("INSERT INTO dim_date VALUES " + ",".join(
            "(%s,'%s',%d,%d,'%s',%d,%d)" % (x.strftime("%Y%m%d"), x.date(), x.day, x.month, x.strftime("%B"), x.quarter, x.year)
            for x in d) + ";\n")
        cid = {c: i + 1 for i, c in enumerate(cities)}
        vals = [f"({r.date.replace('-','')},{cid[r.city]},{dominant(r)},{r.temperature},{r.humidity},{r.pm25},{r.pm10},{r.co},{r.no2},{r.so2},{r.o3},{r.aqi})"
                for r in df.itertuples()]
        for i in range(0, len(vals), 500):
            f.write("INSERT INTO fact_air_quality (date_id,location_id,pollutant_id,temperature,humidity,pm25,pm10,co,no2,so2,o3,aqi) VALUES\n"
                    + ",\n".join(vals[i:i + 500]) + ";\n")

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    df = generate(seed)
    df.to_csv("data/air_quality.csv", index=False)
    write_sql(df)
    print(len(df), "rows written; AQI mean", round(df.aqi.mean(), 1))
