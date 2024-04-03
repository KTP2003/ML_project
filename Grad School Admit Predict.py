#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import seaborn as sns 
import os


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


df=pd.read_csv('admission_data.csv')


# In[5]:


df.head()


# In[6]:


df.columns


# In[7]:


df.isnull().any()


# In[8]:


df.describe()


# In[9]:


sns.countplot(x='Research',data=df)


# In[10]:


df.plot.scatter('Chance of Admit ','SOP', color = 'lightgreen')


# In[11]:


df.plot.scatter('Chance of Admit ','GRE Score', color = 'red')


# In[12]:


# Chance of Admit increases after increase in cgpa
#ideal CGPA is 8.5 CGPA+


# In[13]:


df.plot.scatter('Chance of Admit ','TOEFL Score', color='darkturquoise')


# In[14]:


#chance of Admit increases with TOEFL score
#ideal is 110+


# In[15]:


df.plot.scatter('GRE Score','TOEFL Score', color = 'blue')


# In[16]:


'''
The Trend that is observed is that who have scored better in GRE have scored 
better in TOEFL and VICE VERSA
'''


# In[17]:


df.plot.scatter('GRE Score','CGPA', color = 'green')


# In[18]:


'''The Trend that is observed is that who have better CGPA have scored 
better in GRE and VICE VERSA
'''


# In[19]:


X=df.drop(["Chance of Admit ",],axis=1)
y=df['Chance of Admit ']


# In[20]:


plt.figure(figsize=(10,10),clear=False)
sns.heatmap(df.corr(),linewidths=0.5,annot=True)
plt.show()


# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2003)


# In[22]:


X_train


# In[23]:


X_test


# In[24]:


y_test


# In[25]:


from sklearn.ensemble import RandomForestRegressor
rf_classifier=RandomForestRegressor(n_estimators=100).fit(X_train,y_train)
predictions=rf_classifier.predict(X_test)


# In[26]:


from sklearn.metrics import r2_score;
r2_score(y_test,predictions)


# In[27]:


'''Hyperparameter Tuning'''


# In[30]:


y_test_rf=rf_classifier.predict(X_test)
y_train_rf=rf_classifier.predict(X_train)


# In[32]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[33]:


print("Mean Squared Error testing data:",mean_absolute_error(y_test,y_test_rf))
print("Root mean Square error in testing data:",np.sqrt(mean_absolute_error(y_test,y_test_rf)))


# In[34]:


print("Mean Squared Error training data:",mean_absolute_error(y_train,y_train_rf))
print("Root mean Square error in training data:",np.sqrt(mean_absolute_error(y_train,y_train_rf)))


# In[ ]:




