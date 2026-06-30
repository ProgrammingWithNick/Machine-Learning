import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.datasets import load_iris

#read data from csv file
iris=pd.read_csv(r"C:\Users\nickk\Downloads\ml\Lab-04\iris.csv")

# define data and target variables
y=iris["species"]
x=iris[["sepal_length","sepal_width","petal_length","petal_width"]]

print(y)
print(x)

# splitting of dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=50,test_size=0.2)

#model creation and fitting
dtree1=tree.DecisionTreeClassifier(max_depth=3, random_state=1)
dtree1.fit(x_train,y_train)

#predict the class labels for x_test
y_predict=dtree1.predict(x_test)

#print accuracy
print('Accuracy of Decision Tree-Test: ', accuracy_score(y_predict, y_test))

#print confusion matrix
print('\n','Confusion Matrix - Test:','\n',confusion_matrix(y_test,y_predict))

#print precision, recall and f1-score
print(classification_report(y_test,y_predict))

