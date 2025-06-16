# Install required package
# !pip install fastf1

# Imports
import os
import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Define cache directory
cache_dir = "/content/f1_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)
print("Cache enabled at:", cache_dir)

# Load FastF1 2024 Chinese GP race session
session_2024 = fastf1.get_session(2024, 'China', 'R')
session_2024.load()



# Extract lap times
laps_2024 = session_2024.laps[["Driver", "LapTime","Sector1Time","Sector2Time","Sector3Time"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# 2025 Qualifying Data (Update with actual qualifying results)
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
        "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda",
        "Alexander Albon", "Esteban Ocon", "Nico Hulkenberg"
    ],
    "QualifyingTime (s)": [
         75.096, 75.102, 75.109, 75.115, 75.121, 75.127, 75.133, 75.139, 75.145, 75.151, 75.890, 75.900
    ]
})

# Map driver names to FastF1 codes (ensure all codes are correct)
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR",
    "Max Verstappen": "VER", "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC",
    "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT", "Yuki Tsunoda": "TSU",
    "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hulkenberg": "HUL"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge 2025 qualifying data with 2024 race lap times
merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver", how="left")
merged_data.dropna(inplace=True)
merged_data["LapTime"] = merged_data["LapTime"].dt.total_seconds()

# Calculate the average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time", "Sector2Time", "Sector3Time"]].mean()

# Feature selection
X = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime"]
if X.shape[0] == 0:
    raise ValueError("No data available for training.")

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict 2025 race times
predicted_times = model.predict(qualifying_2025[["QualifyingTime (s)"]])
qualifying_2025["PredictedRaceTime"] = predicted_times

# Rank drivers
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime")
qualifying_2025["Position"] = range(1, len(qualifying_2025) + 1)

# Print final predictions
print("\n Predicted 2025 Chinese GP Results \n")
print(qualifying_2025[["Position", "Driver", "PredictedRaceTime"]])

# Evaluate model
y_pred = model.predict(X_test)
print(f"\n Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
