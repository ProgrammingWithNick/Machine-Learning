from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

b_cancer=load_breast_cancer()
X=b_cancer.data
y=b_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

for k in range(6,17):
    # Instantiate learning model (k = 6 to 16)
    classifier = KNeighborsClassifier(n_neighbors=k)

    # Fitting the model
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)*100
    print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
