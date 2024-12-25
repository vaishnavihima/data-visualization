#ML data preprocessing pipeline using sklearn models using transformers 
#transformers to fill missing value, categorical to numerical

import numpy as np #array

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv(r'/Users/Himavaishnavi/Documents/statestics/data.csv')

dataset.describe()

#independent variable - childern, iloloc means location :-1 means remove that column

x= dataset.iloc[:,:-1].values 
#dependent variable- father
y=dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer
#sklearn: ml frame work|| imput:to fill missin gvalue
#SimpleImputer:library

imputer =SimpleImputer() #gives mean
#imputer = SimpleImputer(strategy="most_frequent")#or median

#in null places mean value is filled ,x[:,1:3] 2d array so used ,
imputer =imputer.fit(x[:,1:3])

x[:,1:3]= imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder
#label encoder convorts catogerical data to numerical

LabelEncoder_x =LabelEncoder()

LabelEncoder_x.fit_transform(x[:,0])
#x[:,0] strating column in x is places convert it into numbers
x[:,0] =LabelEncoder_x.fit_transform(x)

#convert y into numerical
LabelEncoder_y =LabelEncoder()

y = LabelEncoder_y.fit_transform(y)

#split data
from sklearn.model_selection import train_test_split
#in train test split split into 4 xtrain,test,y train,test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, train_size=0.7)

# or test_size=0.8, train_size=0.2 it is 80 -20





