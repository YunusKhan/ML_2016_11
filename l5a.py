#Sample code predicts salary 

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import tensorflow as tf
import pdb

data = pd.read_csv('/home/khany1/allcode/datasets/d2.csv')
trX = data['Age'].values
trY = data['Salary'].values
n_samples = trX.shape[0]

trX=np.array(trX).reshape(-1,1)
trY=np.array(trY).reshape(-1,1)

# Split the data into training/testing sets
trX_train = trX[2:20]
trX_test = trX[21:300]

# Split the targets into training/testing sets
trY_train = trY[2:20]
trY_test = trY[21:300]


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(trX_train, trY_train)

age1 = input("what is your age?\n")

sal1 = regr.coef_ * age1 + regr.intercept_
print('Salary for ', age1, ' year old is', sal1)




