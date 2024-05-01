import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

X = np.array([1,2,6,9,15,19,22,30,35,39,42,48,55,63,72,77]).reshape(-1,1)
Y = np.array([3,5,9,17,25,37,42,56,61,74,77,82,89,95,100,109])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

print("RMSE = ",mean_squared_error(Y_test,Y_pred,squared=False))
print("Coeffiecients = ",model.coef_)
print("Intercept = ", model.intercept_)

diff = pd.DataFrame({'Actual value': Y_test, 'Predicted value': Y_pred})
print('\n')
print(diff.head())

plt.scatter(X_test,Y_pred,color='lightcoral')
plt.plot(X_test,Y_pred,color='blue')
plt.title('Linear Regression')
plt.show()