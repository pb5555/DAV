from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

data = fetch_openml("Bike_Sharing_Demand", version= 3,as_frame=True)
df = data.frame
df.head()


plt.figure(figsize=(10,10))
year_count = df.groupby('year')['count'].sum().reset_index()
g = sns.barplot(data=year_count,x='year',y='count')
for v in year_count.itertuples():
    g.text(x=v.Index,y=v.count+1e4,s=str(v.count), size=7, ha="center")
plt.title('Bike Sharing count by Year')
plt.show()


plt.figure(figsize=(10,10))
year_month_count = df.groupby(['year','month'])['count'].sum().reset_index()
sns.barplot(data=year_month_count,x='month',y='count',hue='year')
plt.title('Bike Sharing count by Year and Month')
plt.show()


plt.figure(figsize=(10,10))
year_season_count = df.groupby(['year','season'])['count'].sum().reset_index()
sns.barplot(data=year_season_count,x='season',y='count',hue='year')
plt.title('Bike Sharing count by Year and Season')
plt.show()


plt.figure(figsize=(10,10))
holiday_count = df.groupby('holiday')['count'].sum().reset_index()
g = sns.barplot(data=holiday_count,x='holiday',y='count')
for v in holiday_count.itertuples():
    g.text(x=v.Index,y=v.count+1e4,s=str(v.count), size=7, ha="center")
plt.title('Bike Sharing count by Holiday')
plt.show()

X = df.drop(['windspeed','count','workingday',	'weather',	'temp',	'feel_temp',	'humidity',	'windspeed','season'], axis=1)
Y = df['count']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.01)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

df_predictions = pd.DataFrame({
    'month': X_test['month'],
    'actual_count': y_test,
    'predicted_count': y_pred
})

df_predictions = df_predictions.sort_values(by='month')

plt.figure(figsize=(10,10))
sns.lineplot(data=df_predictions, x='month', y='actual_count', label='Actual Count',ci=False)
sns.lineplot(data=df_predictions, x='month', y='predicted_count', label='Predicted Count',ci=False)
plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Actual vs. Predicted Bike Sharing Count by Month')
plt.show()