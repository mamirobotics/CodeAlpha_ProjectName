#  Muhammad Amir Mushtaq
# TASK 2: UNEMPLOYMENT ANALYSIS
# CodeAlpha Data Science Internship

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Here I load the dataset

df = pd.read_csv(r"C:\Users\Acer\Downloads\Code Alpha Internship\Task Code and Results\Data Scince Internship\Unemployment in India.csv")

print("\nDataset Shape:", df.shape)
print(df.head())
print(df.columns)

# 2) Clean Column Names
df.columns = [c.strip() for c in df.columns]

# Rename if needed (some datasets have extra spaces)
if "Region" not in df.columns:
    for col in df.columns:
        if "Region" in col:
            df.rename(columns={col: "Region"}, inplace=True)

if "Date" not in df.columns:
    for col in df.columns:
        if "Date" in col:
            df.rename(columns={col: "Date"}, inplace=True)

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

print("\nMissing Values:\n", df.isnull().sum())

# Drop missing
df.dropna(inplace=True)

# 3) Basic Stats

print("\nSummary:\n", df.describe())

# 4) Plot Overall Unemployment Trend
monthly = df.groupby("Date")["Estimated Unemployment Rate (%)"].mean().reset_index()

plt.figure(figsize=(10, 5))
plt.plot(monthly["Date"], monthly["Estimated Unemployment Rate (%)"])
plt.title("India Average Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5) Covid-19 Impact Analysis
covid_start = pd.to_datetime("2020-03-01")

before_covid = monthly[monthly["Date"] < covid_start]
after_covid = monthly[monthly["Date"] >= covid_start]

plt.figure(figsize=(10, 5))
plt.plot(before_covid["Date"], before_covid["Estimated Unemployment Rate (%)"], label="Before Covid")
plt.plot(after_covid["Date"], after_covid["Estimated Unemployment Rate (%)"], label="During/After Covid")
plt.axvline(covid_start, linestyle="--")
plt.title("Unemployment Rate - Covid Impact")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nAverage unemployment before Covid:",
      round(before_covid["Estimated Unemployment Rate (%)"].mean(), 2))

print("Average unemployment during/after Covid:",
      round(after_covid["Estimated Unemployment Rate (%)"].mean(), 2))


# 6) Region-wise Analysis

region_avg = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
region_avg.plot(kind="bar")
plt.title("Average Unemployment Rate by Region")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()


# 7) Seasonal Trend (Month Wise)
df["Month"] = df["Date"].dt.month
month_avg = df.groupby("Month")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(8, 5))
plt.plot(month_avg.index, month_avg.values, marker="o")
plt.title("Seasonal Trend - Average Unemployment by Month")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()
