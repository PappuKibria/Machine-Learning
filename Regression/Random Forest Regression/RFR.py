# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 02:53:39 2019

@author: Kibria
"""

#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,[2]].values

#Fitting Decision Tree Regression to the Dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=30, random_state=0)
regressor.fit(X,y)

#Predicting a new result
y_pred = regressor.predict([[6.5]])

#Visualising the RFR results
plt.scatter(X, y, color = 'red')
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()