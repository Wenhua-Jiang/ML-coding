#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[10]:


data = pd.read_csv('spam_ham_dataset.csv')


# In[11]:


data.head()


# In[12]:


X_train, X_test, Y_train, Y_test = train_test_split(data['text'],
data['label_num'])


# In[13]:


cv = CountVectorizer()
vectorizer = cv.fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)


# In[14]:


X_train_vectorized


# In[15]:


model = MultinomialNB(alpha=0.1)
model.fit(X_train_vectorized, Y_train)


# In[16]:


predictions = model.predict(vectorizer.transform(X_test))
print("Accuracy:", 100 * sum(predictions == Y_test) / len(predictions), '%')


# In[17]:


model.predict(vectorizer.transform(
[
"Hello Mike, I can came across your profile on Indeed, are you available for a short chat over the weekend.",
])
)


# In[ ]:




