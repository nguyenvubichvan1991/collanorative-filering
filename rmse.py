# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:20:45 2018

@author: WIN 10
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:43:59 2018

@author: WIN 10
"""
import pandas
import numpy
import scipy
import sys
import sklearn
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd 
from sklearn.metrics import mean_squared_error
from math import sqrt
names =['userid','idnl']
dataset = pandas.read_csv('D:\\Data\\schools.csv',names=names) ## names: la tieu de trong file 
print(dataset.shape)
print(dataset.head())
print(dataset.describe())
print(dataset.groupby('idnl').size())

import pandas as pd 
from sklearn.metrics import mean_squared_error
from math import sqrt
#Reading user file:
u_cols =  ['user_id', 'name', 'nl']
users = pd.read_csv('D:/Data/tblusers.csv', sep='|', names=u_cols,
 encoding='latin-1')
n_users = users.shape[0]
print( 'Number of users:', n_users)
# users.head() #uncomment this to see some few examples
r_cols = ['user_id', 'nlid','rating','unix_timestamp']
 
ratings_base = pd.read_csv('D:/Data/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('D:/Data/ua.test', sep='\t', names=r_cols, encoding='latin-1')
print ('Number of traing rates:', ratings_base.shape[0])
print ('Number of test rates:', ratings_test.shape[0])
i_cols = ['nlid', 'nltitle', 'math1', 'english','health', 'science', 'doctor', 'technology', 'paint', 'sings', 'marketing',
 'travel', 'business']
items = pd.read_csv('D:/Data/nangluc.csv', sep='|', names=i_cols, encoding='latin-1')
n_items = items.shape[0]
print ('Number of items:', n_items)

import numpy as np
nl = [0.99, 0.91, 0.95, 0.01, 0.03]
sc = [0.02, 0.11, 0.05, 0.99, 0.98]

print("nl  is: " + str(["%.8f" % elem for elem in nl]))
print("sc  is: " + str(["%.8f" % elem for elem in sc]))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse_val = rmse(np.array(nl), np.array(sc))
print("RMSE " + str(rmse_val))