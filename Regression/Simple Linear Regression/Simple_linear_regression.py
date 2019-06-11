# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:45:27 2019

@author: Kibria
"""

#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset = pd.read_csv("Beton.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values



#Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting Simple Linear Regression in the training set
from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(X_train, y_train)

#Predicting the test set result
y_pred = linearRegression.predict(X_test)

#Visualizing the training set data
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train,linearRegression.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the test set data
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test,linearRegression.predict(X_test), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()