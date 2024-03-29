# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:32:06 2019

@author: Kibria
"""

#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values


#Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
''''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)''''
