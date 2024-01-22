import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.dates as mdates

# Load the CSV file using pandas and parse dates
data = pd.read_csv('data.csv', header=0, parse_dates=[3], index_col=3)

# Extract data for EU27_2020 and LV
eu27_data = data[data['geo'] == 'EU27']['OBS_VALUE']
lv_data = data[data['geo'] == 'LV']['OBS_VALUE']

# Fit ARIMA models and forecast for EU27_2020 and LV separately
def forecast_and_plot(data, geo):
    model = ARIMA(data, order=(5, 1, 1))  # Adjust the differencing terms (e.g., order=(p, d, q))
    model_fit = model.fit()

    future_steps = 5  # Change this to the number of future data points you want to forecast

    future_index = pd.date_range(start='2021-12-31', periods=future_steps, freq='Y')
    forecast = model_fit.forecast(steps=future_steps)

    plt.plot(data.index, data, label=f'{geo} Elektrības cenas')
    plt.plot(future_index, forecast, label=f'{geo} Prognozētās cenas', linestyle='dashed')

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Plot separate lines for EU27_2020 and LV
plt.figure(figsize=(10, 6))
forecast_and_plot(eu27_data, 'EU27')
forecast_and_plot(lv_data, 'LV')

plt.xlabel('Gadi')
plt.ylabel('Cena (EUR)')
plt.title('ARIMA elektrības cenas prognoze')
plt.legend()
plt.show()
