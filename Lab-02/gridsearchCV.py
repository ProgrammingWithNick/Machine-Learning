# Grid Search CV with SVM

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=0,
    test_size=0.25
)

# Parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
}

print("Parameter Grid:")
print(param_grid)

# Grid Search
grid_search = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

# Train model
grid_search.fit(X_train, y_train)

# Results
print("\nTest Set Accuracy: {:.2f}".format(
    grid_search.score(X_test, y_test)
))

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Score:")
print("{:.2f}".format(grid_search.best_score_))

print("\nBest Estimator:")
print(grid_search.best_estimator_)

# Convert results to DataFrame
results = pd.DataFrame(grid_search.cv_results_)

print("\nFirst 5 Rows of Results:")
print(results.head())

# Create score matrix
scores = np.array(results['mean_test_score']).reshape(
    len(param_grid['C']),
    len(param_grid['gamma'])
)

# Heatmap using matplotlib
plt.figure(figsize=(8, 6))

plt.imshow(scores, interpolation='nearest')
plt.colorbar()

plt.xticks(
    np.arange(len(param_grid['gamma'])),
    param_grid['gamma']
)

plt.yticks(
    np.arange(len(param_grid['C'])),
    param_grid['C']
)

plt.xlabel("Gamma")
plt.ylabel("C")
plt.title("Grid Search CV Heatmap")

plt.show()