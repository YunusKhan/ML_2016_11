import numpy as np
import os
import pandas as pd
import sklearn


df=pd.read_csv('/home/khany1/allcode/usele/haberman.txt', header=None, names= ['age', 'op_year', 'nodes', 'survival'])
feat = df[['age', 'op_year', 'survival']]
target = df[['survival']]
from sklearn import cross_validation as cv
splits = cv.train_test_split(feat, target, test_size = 0.2)
xtrain, xtest,ytrain, ytest = splits
expected = [2,2,1,1]

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report as CSR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RFC

model_svc = SVC()
model_rfc = RFC()
model_LogRegr = LogisticRegression()
model_LinRegr  = LinearRegression()

model_LinRegr.fit(xtrain, np.ravel(ytrain))
model_LogRegr.fit(xtrain, np.ravel(ytrain))
model_svc.fit(xtrain, np.ravel(ytrain))
model_rfc.fit(xtrain, np.ravel(ytrain))

pd1 = pd.read_csv('/home/khany1/allcode/datasets/t1.txt', header=None, names=['aa','bb','cc'])
pr_linregr = model_LinRegr.predict(pd1)
pr_logregr = model_LogRegr.predict(pd1)
pr_svc = model_svc.predict(pd1)
pr_rfc = model_rfc.predict(pd1)


print ' --------------------------- '
print 'Array of predictions - lin regr / log regr / SVC / RFC'
print pr_linregr
print pr_logregr
print pr_svc
print pr_rfc

print ' --------------------------- '
print 'Accuracy - r2 - lin regr / log regr / SVC / RFC '
print r2_score(expected, pr_linregr)
print r2_score(expected, pr_logregr)
print r2_score(expected, pr_svc)
print r2_score(expected, pr_rfc)
print ' --------------------------- '

print 'Accuracy - mse - lin regr / log regr / SVC / RFC '
print mse(expected, pr_linregr)
print mse(expected, pr_logregr)
print mse(expected, pr_svc)
print mse(expected, pr_rfc)

print ' --------------------------- '





