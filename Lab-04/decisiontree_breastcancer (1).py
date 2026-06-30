from sklearn import datasets
import sklearn.model_selection as ms
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 


data = datasets.load_breast_cancer()

x_train, x_test, y_train, y_test = ms.train_test_split(data.data, data.target, test_size=0.2, random_state=42)

dtc = tree.DecisionTreeClassifier(random_state=42)

dtc.fit(x_train, y_train)

#predict the class labels for x_test
y_predict=dtc.predict(x_test)

#print accuracy
print('Accuracy of Decision Tree-Test: ', accuracy_score(y_predict, y_test))

#print confusion matrix
print('\n','Confusion Matrix - Test:','\n',confusion_matrix(y_test,y_predict))

#print precision, recall and f1-score
print(classification_report(y_test,y_predict))




