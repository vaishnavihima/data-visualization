import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv(r'/Users/Himavaishnavi/Documents/statestics/Salary_Data.csv')
# divide independent ,dependent variabeles
x = dataset.iloc[:,:-1]  #independent variable years of exp
y= dataset.iloc[:, -1]  #dependent salary

#split data into training,testing sets
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=0)
#from dataset we randomly pull the 80% data ,20% from actual data, random state is must

#reshape the x_train,x_test into 2D arrays if they are single feature columns
#no need y_train,test as its the target variable
x_train =x_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)

#if there are any null values
dataset.isnull().sum()

#fit the linear regression into training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  #create object regressor and fit the model to x_train,y_train historical data
regressor.fit(x_train, y_train)

#predicting the results for test set
y_pred = regressor.predict(x_test)

#visuvalizing training set results
plt.scatter(x_test, y_test, color ='red')
plt.plot(x_train, regressor.predict(x_train),color= 'blue')
plt.title('Salary vs experience(Test set)')
plt.xlabel('Years of expereinces')
plt.ylabel('Salary')
plt.show()

#visulazing the test set values
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('salary vs exp(training set)')
plt.xlabel('exp')
plt.ylabel('salary')
plt.show()

#slop is required to predict future values
m_slope = regressor.coef_
print(m_slope)
# FOR constant value
c_intercept = regressor.intercept_
print(c_intercept)

#predicting the employee wth 15 years of experience
y_15 = m_slope * 15 + c_intercept
y_15
y_25 = m_slope * 25 + c_intercept
y_25

#
#compare the actual an predicted
comparison = pd.DataFrame({'Actual' :y_test,'predicted':y_pred})
print(comparison)

#to check weather it is good model or not -r2
#stastics in ML
import pickle   #converts 30 lines into single code
filename = 'linear_regression_model.pkl'
with open(filename,'wb') as file: #wb:write binary,rb:read binary
    pickle.dump(regressor,file) #dump parameter,created regressor 
print("model has been pickled and saved as linear_regression_model.pkl")

import os
print(os.getcwd())


