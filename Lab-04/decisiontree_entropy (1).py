from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from dtreeviz.trees import dtreeviz
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy')  
# ID3 uses information gain, which is based on entropy
model.fit(X_train, y_train)

# Visualize the decision tree
viz = dtreeviz(model, X, y,feature_names=iris.feature_names,class_names=iris.target_names)
viz.view()
