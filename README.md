# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Salary.csv dataset, display the initial rows (head), and examine the data structure (info) to check for missing values.

2.Encode the "Position" column using LabelEncoder to convert it into numeric format, making it suitable for regression modeling.

3.Define the feature set x (including "Position" and "Level") and target y ("Salary"), and split these into training and testing sets with an 80-20 ratio using train_test_split.

4.Train a DecisionTreeRegressor on the training data to model the relationship between features and salary.

5.Evaluate the model's performance using the Mean Squared Error and R² score, and make predictions for new sample inputs, e.g., dt.predict([[5,6]]).
## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: RAMYA S
RegisterNumber:  212222040130

import pandas as pd

data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])

data.head()

x=data[["Position", "Level"]]

y=data["Salary"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=2)

from sklearn.tree import DecisionTreeRegressor

dt=Decision TreeRegressor()

dt.fit(x_train, y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

mse=metrics.mean_squared_error(y_test,y_pred) mse

462500000.0

r2=metrics.r2_score(y_test,y_pred)

r2

0.48611111111111116

dt.predict([[5,6]])

/usr/local/lib/python3.7/dist-packages/sklearn/

"X does not have valid feature names, but" array ([200000.])
```

## Output:
## MSE & R2 & PREDICTED
![image](https://github.com/user-attachments/assets/a2c0edad-b1c2-4737-b88e-5e5b396f45fc)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
