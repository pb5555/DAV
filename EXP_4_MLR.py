import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = {
    'X1':[1,2,3,4,5,6,7,8,9],
    'X2':[1,4,12,16,20,25,32,38,43],
    'Y':[1,2,4,8,12,15,22,29,34]
}
df = pd.DataFrame(data)

X = df[['X1','X2']]
Y = df['Y']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

print("RMSE = ",mean_squared_error(Y_test,Y_pred,squared=False))
print("Coefficients = ", model.coef_)
print("Intercept = ", model.intercept_)

diff = pd.DataFrame({'Actual value': Y_test, 'Predicted value': Y_pred})
print('\n')
print(diff.head())

plt.figure(figsize=(12, 6))
plt.title('Multiple Linear Regression')

plt.subplot(1, 2, 1)
plt.scatter(X['X1'], Y, color='blue', label='Data Points')
plt.plot(X_test['X1'], Y_pred, color='red', label='Linear Regression Line')
plt.scatter(X_test['X1'], Y_pred, color='green', label='Test Predictions')
plt.xlabel('Feature 1')
plt.ylabel('y')
plt.title('1st Feature with output')
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
plt.scatter(X['X2'], Y, color='blue', label='Data Points')
plt.plot(X_test['X2'], Y_pred, color='red', label='Linear Regression Line')
plt.scatter(X_test['X2'], Y_pred, color='green', label='Test Predictions')
plt.xlabel('Feature 2')
plt.ylabel('y')
plt.title('2nd Feature with output')
plt.legend()
plt.grid(True)

plt.show()
