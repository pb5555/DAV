import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima

np.random.seed(42)
n = 100
t = np.arange(n)
data = 10 + 0.5 * t + np.random.normal(0, 1, n)

model = auto_arima(data, start_p=1, start_q=1, max_p=3, max_q=3, seasonal=False, trace=True)
model.fit(data)

print(model.summary())

forecast_steps = 15
forecast = model.predict(n_periods=forecast_steps)

plt.plot(t, data, label='Original Data')
plt.plot(np.arange(n, n + forecast_steps), forecast, label='Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
