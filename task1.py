import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime


df = pd.read_csv('Nat_Gas.csv')


print("Columns found:", df.columns.tolist())

if 'Dates' in df.columns:
    df.rename(columns={'Dates': 'Date'}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df['Price'] = pd.to_numeric(df['Prices'], errors='coerce')

df.dropna(inplace=True)

df = df.sort_values('Date')

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], marker='o', color='blue', label='Actual Prices')
plt.title('Monthly Natural Gas Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

def interpolate_price(date_str):
    """
    Estimate gas price for any past date within historical data range.
    """
    date = pd.to_datetime(date_str)
    if date < df['Date'].min() or date > df['Date'].max():
        return None
    return float(np.interp(date.timestamp(),
                           df['Date'].map(lambda x: x.timestamp()),
                           df['Price']))

prophet_df = df.rename(columns={'Date': 'ds', 'Price': 'y'})

model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
model.fit(prophet_df)


future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

fig1 = model.plot(forecast)
plt.title('Natural Gas Price Forecast')
plt.xlabel('Date')
plt.ylabel('Predicted Price')
plt.show()

def get_gas_price(date_str):
    """
    Returns estimated gas price for any given date.
    - Uses interpolation for past dates
    - Uses Prophet forecast for future dates
    - Falls back to linear extrapolation if beyond forecast horizon
    """
    date = pd.to_datetime(date_str)
    
    if date <= df['Date'].max():
        return interpolate_price(date_str)
    
    forecast_date = forecast[forecast['ds'] == date]
    if not forecast_date.empty:
        return float(forecast_date['yhat'])
    
    last_dates = df['Date'].tail(2)
    last_prices = df['Price'].tail(2)
    slope = (last_prices.iloc[1] - last_prices.iloc[0]) / ((last_dates.iloc[1] - last_dates.iloc[0]).days)
    days_ahead = (date - last_dates.iloc[1]).days
    return float(last_prices.iloc[1] + slope * days_ahead)

# print("Estimated price on 2023-05-15:", get_gas_price("2023-05-15"))
# print("Estimated price on 2025-06-30:", get_gas_price("2025-06-30"))
