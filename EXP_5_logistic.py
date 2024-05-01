from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model

# Define X and y with increased data range
X = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Train logistic regression model
logreg = linear_model.LogisticRegression()
logreg.fit(X, y)

# Function to convert logit to probability
def logit2prob(logreg, X):
    log_odds = logreg.coef_ * X + logreg.intercept_
    odds = np.exp(log_odds)
    probability = odds / (1 + odds)
    return probability

print(logit2prob(logreg, X))

# Plotting data points
plt.scatter(X, y, color='blue', label='Data Points')

# Plotting the logistic regression curve
X_values = np.linspace(0, 12, 100).reshape(-1, 1)
plt.plot(X_values, logreg.predict_proba(X_values)[:, 1], color='red', label='Logistic Regression Curve')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Logistic Regression')
plt.legend()

# Display the plot
plt.show()
