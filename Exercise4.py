
# coding: utf-8

# In[1]:


from __future__ import division
get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier


# In[2]:


dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter = ',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter = ',')
XTrain = dataTrain[:, : -1]
YTrain = dataTrain[:, -1]
XTest = dataTest[:, : -1]
YTest = dataTest[:, -1]


# In[3]:


def normalization(train_data, transform_data):
    mean_row = np.mean(train_data, axis = 0)
    std_row = np.std(train_data, axis = 0)
    mean_matrix = np.tile(mean_row, transform_data.shape[0]).reshape(transform_data.shape[0],-1)
    #std_list = [(1/i) for i in std_row]
    std_list = 1/np.array(std_row)
    std_matrix = np.tile(std_list, transform_data.shape[0]).reshape(transform_data.shape[0],-1)
    dif_matrix = np.subtract(transform_data, mean_matrix)
    std_matrix_diag = np.diag(np.diag(std_matrix))
    normalizated_matrix = np.dot(dif_matrix, std_matrix_diag)
    return normalizated_matrix


# In[4]:


nor_xtrain = normalization(XTrain, XTrain)
nor_xtest = normalization(XTrain, XTest)


# In[5]:


clf = RandomForestClassifier(n_estimators = 50)
clf.fit(XTrain, YTrain)
clf.predict(XTest)


# In[6]:


#Exercise 5 (Applying random forrest ).
re = []
for i in range(100):
    clf = RandomForestClassifier(n_estimators = 50)
    clf.fit(XTrain, YTrain)
    clf.predict(XTest)
    ac = clf.score(XTest, YTest)
    re.append(ac)
np.array(re).mean()


# In[9]:


np.array(re).max()


# In[10]:


np.array(re).min()


# In[8]:


clf2 = RandomForestClassifier(n_estimators = 50)
clf2.fit(nor_xtrain, YTrain)
clf2.score(nor_xtest, YTest)

