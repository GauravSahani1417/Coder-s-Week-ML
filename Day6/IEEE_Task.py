# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:51:05 2020

@author: gaurav sahani
"""


import pandas as pd

df=pd.read_csv('Social_Network_Ads.csv.txt')
df.head()

df.Gender[df.Gender == 'Male'] = 1
df.Gender[df.Gender == 'Female'] = 2

df['Gender'] = df['Gender'].astype(int) 
df.head()

X=df[['Gender','Age','EstimatedSalary']]
y=df[['Purchased']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix
pred=LR.predict(X_test)
print(accuracy_score(pred,y_test))
print(confusion_matrix(pred,y_test))

import joblib
joblib.dump(LR, "IEEE.pkl")