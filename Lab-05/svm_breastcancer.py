from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

b_cancer=load_breast_cancer()
X=b_cancer.data
y=b_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Apply gridsearch for different values of kernel, gamma and C
#for kernel: linear, rbf, poly, sigmoid
#for gamma: 0.001,0.01,0.1,1,10,100
#for C: 0.001,0.01,0.1,1,10,100 

svc_model=SVC(kernel='linear',C=1,gamma=10)

# Fitting the model
svc_model.fit(X_train, y_train)
    
# Predicting the Test set results
y_pred = svc_model.predict(X_test)

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')


svc_model=SVC(kernel='rbf',C=1)

# Fitting the model
svc_model.fit(X_train, y_train)
    
# Predicting the Test set results
y_pred = svc_model.predict(X_test)

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

svc_model=SVC(kernel='poly',C=1)

# Fitting the model
svc_model.fit(X_train, y_train)
    
# Predicting the Test set results
y_pred = svc_model.predict(X_test)

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
