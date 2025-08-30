# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SHARMILA P
RegisterNumber:  212224220094
*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('data.csv')
print(dataset.head())
dataset=pd.read_csv('data.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```

## Output:
<img width="244" height="797" alt="Screenshot 2025-08-30 143454" src="https://github.com/user-attachments/assets/3b5b71fe-0fab-448c-baf6-8b4478cb7fce" />
<img width="867" height="322" alt="Screenshot 2025-08-30 143543" src="https://github.com/user-attachments/assets/a103f62f-7e7e-4ead-a29c-4a5a4dbe2f69" />
<img width="836" height="682" alt="Screenshot 2025-08-30 143555" src="https://github.com/user-attachments/assets/53bed8fb-7601-4296-9cbe-904d7776be18" />
<img width="862" height="804" alt="Screenshot 2025-08-30 143622" src="https://github.com/user-attachments/assets/3f981099-d5cb-4963-969e-5d19c8de1161" />








## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
