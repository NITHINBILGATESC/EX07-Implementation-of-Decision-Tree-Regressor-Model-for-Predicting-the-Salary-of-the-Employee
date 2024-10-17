# EX 7 Implementation of Decision Tree Regressor Model for Predicting the Salary of the Employee
## DATE:
## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.We handled categorical variables and split the data into training and test sets. 
2.We trained a Decision Tree Regressor with depth control to avoid overfitting.
3.The model was evaluated using Mean Squared Error (MSE) and R-squared (R²) metrics.
4.We used GridSearchCV to optimize the model.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: NITHIN BILGATES C
RegisterNumber:  2305001022
*/
```
import pandas as pd
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
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
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
## Output:
![image](https://github.com/user-attachments/assets/e0c04ad9-3a43-4288-9212-1aa933c6c7d5)
![image](https://github.com/user-attachments/assets/d0a89f1f-a53c-4e21-bce5-aac91e33ce1f)
![image](https://github.com/user-attachments/assets/5e70db60-b834-48c4-8701-f90e19a08aa2)
![image](https://github.com/user-attachments/assets/8fbd8f4e-1144-4ffb-b002-b66723571d57)
![image](https://github.com/user-attachments/assets/4ef5edcd-3e2b-4600-b4fe-2bd3fb510877)
![image](https://github.com/user-attachments/assets/ff7fdc44-6582-44b1-8115-09d0787f3807)
![image](https://github.com/user-attachments/assets/28063497-fc2f-4d26-93f2-8f52b7433eab)
![image](https://github.com/user-attachments/assets/3ed457c3-10bf-4c08-ae6c-b66fb0fa5392)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
