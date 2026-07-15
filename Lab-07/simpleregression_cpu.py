import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\nickk\Downloads\ml\Lab7\cputime.csv")

x=data.iloc[:,0:1]
y=data.iloc[:,1]

print(x)
print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Creating and fitting the model
model = LinearRegression()
model.fit(X_train,y_train)

# Making predictions
y_pred_test = model.predict(X_test)     # predicted value of y_test
y_pred_train = model.predict(X_train)   # predicted value of y_train

y_pred=model.predict([[40]])

print("Value of CPU time for disk I/O = 40 is",y_pred)

# Plotting results
plt.scatter(X_train, y_train, color = 'lightcoral')
plt.plot(X_train, y_pred_train, color = 'firebrick')
plt.title('CPU time vs disk I/O)')
plt.xlabel('Disk I/O')
plt.ylabel('CPU time')
plt.legend(['X_train/Pred(y_test)', 'X_train/y_train'], title = 'disk/cpu', loc='best', facecolor='white')
plt.box(False)
plt.show()


#domain classification,regression,cluster
#50k records   10 

