import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

weather = pd.read_csv("DailyDelhiClimateTest.csv")
print(weather.head())
weather["date"] = pd.to_datetime(weather["date"])
weather["Days"] = (
    weather["date"] - weather["date"].min()
).dt.days
X = weather[["Days"]]
y = weather["meantemp"]
model = LinearRegression()
model.fit(X, y)
print("Model trained successfully!")
predictions = model.predict(X)
weather["Predicted_Temp"] = predictions
print(weather.head())
plt.figure(figsize=(10,5))
plt.plot(
    weather["date"],
    weather["meantemp"],
    label="Actual Temperature"
)
plt.plot(
    weather["date"],
    weather["Predicted_Temp"],
    label="Predicted Temperature"
)
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Delhi Temperature Trend")
plt.legend()
plt.show()
future_days = np.arange(
    weather["Days"].max() + 1,
    weather["Days"].max() + 8
).reshape(-1, 1)
future_predictions = model.predict(future_days)
print("\nFuture Temperature Predictions:")
for i, value in enumerate(future_predictions, start=1):
    print(f"Day +{i}: {value:.2f}°C")
