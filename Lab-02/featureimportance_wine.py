import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Read dataset
df_wine = pd.read_csv(r"D:\A4087\ml02\wine (1).csv")

print("Dataset:")
print(df_wine)

# Features and target
X = df_wine.iloc[:, 1:].values
y = df_wine.iloc[:, 0].values

# Feature names
feat_labels = df_wine.columns[1:]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Feature Importances
feature_importances = clf.feature_importances_

print("\nFeature Importances:")
for feature, importance in zip(feat_labels, feature_importances):
    print(f"{feature}: {importance:.4f}")

# Sort feature importances
indices = np.argsort(feature_importances)[::-1]

print("\nRanked Feature Importances:")
for f in range(len(feat_labels)):
    print(
        f"{f+1:2d}) {feat_labels[indices[f]]:<30} "
        f"{feature_importances[indices[f]]:.4f}"
    )