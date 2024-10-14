# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: RAMYA S
RegisterNumber:  212222040130
*/
```
```
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
![image](https://github.com/user-attachments/assets/b7adc02f-3d9f-402e-8449-bbeac322d8a7)
![image](https://github.com/user-attachments/assets/2065aa35-23e8-4ba2-823b-4db77be1a90f)
![image](https://github.com/user-attachments/assets/f9000e5c-a572-4d45-94c2-0b81399be0d0)










## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
