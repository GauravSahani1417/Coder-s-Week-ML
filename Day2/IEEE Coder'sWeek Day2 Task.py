#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd


# In[2]:


df=pd.read_csv('Social_Network_Ads.csv.txt')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


plt.hist(df['Age'], color = 'peru', edgecolor = 'black',bins=6)


# In[6]:


sns.set_style('darkgrid')
sns.countplot(x = 'Purchased', hue = 'Gender', data = df)


# In[7]:


sns.catplot(x="Age", y="EstimatedSalary", data=df,kind="swarm",hue='Gender',height=5,aspect=3)


# In[8]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=df['EstimatedSalary'], y=df['Age']);


# In[9]:


df.Gender[df.Gender == 'Male'] = 1
df.Gender[df.Gender == 'Female'] = 2

df['Gender'] = df['Gender'].astype(int) 


# In[10]:


df.head()


# In[11]:


df.info()


# In[12]:


X=df[['User ID','Gender','Age','EstimatedSalary']]
y=df[['Purchased']]


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[14]:


df.Purchased.unique()


# In[15]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X_train,y_train)


# In[16]:


from sklearn.metrics import accuracy_score,confusion_matrix
pred=LR.predict(X_test)
print(accuracy_score(pred,y_test))
print(confusion_matrix(pred,y_test))


# In[17]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)


# In[18]:


pred1=knn.predict(X_test)
print(accuracy_score(pred1,y_test))
print(confusion_matrix(pred1,y_test))


# In[19]:


from sklearn.tree import DecisionTreeClassifier
DTC=DecisionTreeClassifier()
DTC.fit(X_train,y_train)


# In[20]:


pred2=DTC.predict(X_test)
print(accuracy_score(pred2,y_test))
print(confusion_matrix(pred2,y_test))


# In[22]:


pred2


# We can see that, DecisionTreeClassifier gives the much better accuracy than other.

# Thankyou!

# In[ ]:




