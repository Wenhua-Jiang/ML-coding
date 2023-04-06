#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[48]:


dataset = pd.read_csv('diabetes.csv')


# In[49]:


dataset.head()


# In[50]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values.flatten()


# In[51]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[52]:


classifier = KNeighborsClassifier(n_neighbors = 15, metric = 'euclidean', p = 2)
classifier.fit(X_train, y_train)


# In[53]:


y_pred = classifier.predict(X_test)


# In[54]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[55]:


cm


# In[56]:


print(accuracy_score(y_test,y_pred))

plt.show()


# In[ ]:




