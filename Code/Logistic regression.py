#!/usr/bin/env python
# coding: utf-8

# ### logistic regression

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('breast-cancer.csv')
dataset.drop('id', axis =1, inplace=True)


# In[20]:


dataset.head(3)


# In[35]:


X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0:1].values.flatten()

# Normalize the  data 
scaler =StandardScaler()
Xs = scaler.fit_transform(X)


# In[36]:


Xs


# In[37]:


#Encoder y labels from M and B to 0 and 1
LE = LabelEncoder()
y = LE.fit_transform(y)


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size = 0.25, random_state = 0)


# In[39]:


classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[40]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[ ]:




