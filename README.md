# EX 7 Implementation of Decision Tree Regressor Model for Predicting the Salary of the Employee
## DATE:
## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Dataset: Import the data (e.g., employee salary data) into a pandas DataFrame.
2. Handle Missing Values: Identify and either fill or remove missing values.
3. Encode Categorical Variables: Convert categorical columns (e.g., department, gender) into numerical form using label encoding or one-hot
encoding.
4. Split the Dataset: Define your features (X) and target (y), then split the data into training and testing sets.
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: NITHIN BILGATES C
RegisterNumber:  2305001022
*/
```import pandas as pd
df=pd.read_csv("/content/Salary_EX7.csv")
df

from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data=df.copy()
data.describe()

data.info()
data

data.isnull().sum()
data

le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
data.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:

![image](https://github.com/user-attachments/assets/d6a763e5-1ccd-4e0b-b619-ce961ff376ff)
![image](https://github.com/user-attachments/assets/cc046e5f-9fbe-415d-9824-e8a1fc441b08)
![image](https://github.com/user-attachments/assets/c0a9ec13-32bd-45c8-ba05-61b2e7a5c1f9)
![image](https://github.com/user-attachments/assets/a805a0c1-4f72-4013-9758-6954aafdbc00)
![image](https://github.com/user-attachments/assets/ba97c115-57d6-4aba-8c68-721a3daeef98)
![image](https://github.com/user-attachments/assets/18e96d63-79fd-4d51-9491-dd544a86fb6c)
![image](https://github.com/user-attachments/assets/3f3e06ab-0430-4654-9d70-13b44129a7ba)
![image](https://github.com/user-attachments/assets/d8240c78-7ea3-4126-ad46-056953e7c41d)






## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
