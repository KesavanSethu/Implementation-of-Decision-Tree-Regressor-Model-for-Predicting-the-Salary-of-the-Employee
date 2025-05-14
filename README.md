# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries for data handling, preprocessing, modeling, and evaluation.
2. Load the dataset from the CSV file into a pandas DataFrame.
3. Check for null values and inspect data structure using .info() and .isnull().sum().
4. Encode the categorical "Position" column using LabelEncoder.
5. Split features (Position, Level) and target (Salary), then divide into training and test sets.
6. Train a DecisionTreeRegressor model on the training data.
7. Predict on test data, calculate mean squared error and R² score, and make a sample prediction.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Elavarasan M
RegisterNumber:  212224040083
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
```
```
# load the dataframe
data = pd.read_csv("Salary.csv")
```
```
# display head values
data.head()
```
```
# display dataframe information
data.info()
```
```
# display the count of null values
data.isnull().sum()
```
```
# encode postion using label encoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
```
```
# defining x and y and splitting them
x = data[["Position", "Level"]]
y = data["Salary"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
```
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
```
```
# predicting test values with model
y_pred = dt.predict(x_test)
```
```
mse = metrics.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error : {mse}")
```
```
r2 = metrics.r2_score(y_test, y_pred)
print(f"R Square : {r2}")
```
```
dt.predict(pd.DataFrame([[5,6]], columns=["Position", "Level"]))
```

## Output:

**Head Values**

![Screenshot 2025-05-12 181141](https://github.com/user-attachments/assets/8ab7b5e8-8629-4307-bc22-442333ca09e6)

**DataFrame Info**

![Screenshot 2025-05-12 181146](https://github.com/user-attachments/assets/0ba175f2-3b75-4987-863a-7aa381ddf408)

**Sum - Null Values**

![Screenshot 2025-05-12 181150](https://github.com/user-attachments/assets/1ca53118-d3bc-45b8-ab6c-264401c2e7b8)

**Encoding position feature**

![Screenshot 2025-05-12 181157](https://github.com/user-attachments/assets/bba7a0d9-acf4-43bc-a9ac-3786184be3f3)

**Training the model**

![Screenshot 2025-05-12 181211](https://github.com/user-attachments/assets/2013bd39-a609-4ca3-a521-75ac1b69d587)

**Mean Squared Error**

![Screenshot 2025-05-12 181219](https://github.com/user-attachments/assets/1ce4f60a-cf3d-462d-8455-dd4a8b4035de)

**R2 Score**

![Screenshot 2025-05-12 181225](https://github.com/user-attachments/assets/78d467e3-d966-489d-b8c9-9a74631d2ded)

**Final Prediction on Untrained Data**

![Screenshot 2025-05-12 181230](https://github.com/user-attachments/assets/c7781d0e-f2f6-41f0-a2a6-8453d3c06fac)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
