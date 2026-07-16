# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Generate a sample dataset (you would typically load your own dataset)
data = {'X1': [1, 2, 3, 4, 5,6,7,8,9,10],
        'X2': [2, 3, 4, 5, 6,7,8,9,10,11],
        'Y': [3, 5, 7, 9, 11,13,15,17,19,21]}

df = pd.DataFrame(data)

# Split the data into independent variables (X) and the dependent variable (Y)
X = df[['X1', 'X2']]
Y = df['Y']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = model.predict(X_test)

# Evaluate the model
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred, squared=False))

# Print the coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
