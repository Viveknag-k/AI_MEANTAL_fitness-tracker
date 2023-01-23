#!/usr/bin/env python
# coding: utf-8

# <div style="color:black;
#            display:fill;
#            border-radius:50px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;">
#             IMPORTING REQUIRED LIBRARIES
# </p>
# </div>

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# <div style="color:black;
#            display:fill;
#            border-radius:25px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;">
#             READING DATA
# </p>
# </div>

# In[2]:


df1 = pd.read_csv("C:\\Users\\Vivek Nag Kanuri\\Downloads\\prevalence-by-mental-and-substance-use-disorder.csv")
df2=pd.read_csv("C:\\Users\\Vivek Nag Kanuri\\Downloads\\mental-and-substance-use-as-share-of-disease.csv")


# In[3]:


df1.head()


# In[4]:


df2.head()


# <div style="color:black;
#            display:fill;
#            border-radius:25px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;">
#             MERGING TWO DATASETS
# </p>
# </div>

# In[5]:


data = pd.merge(df1, df2)


# In[6]:


data.head()


# <div style="color:black;
#            display:fill;
#            border-radius:25px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;">
#             DATA CLEANING
# </p>
# </div>

# In[7]:


data.isnull().sum()


# In[8]:


data.drop('Code',axis=1,inplace=True)


# In[9]:


data.head()


# In[10]:


data.size


# In[11]:


data.shape


# In[12]:


data.set_axis(['Country','Year','Schizophrenia', 'Bipolar_disorder', 'Eating_disorder','Anxiety','drug_usage','depression','alcohol','mental_fitness'], axis='columns', inplace=True)


# In[13]:


data.head()


# In[14]:


plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True,cmap='Blues')
plt.plot()


# In[15]:


sns.jointplot('Schizophrenia','mental_fitness',data,kind='reg',color='m')
plt.show()


# In[16]:


sns.jointplot('Bipolar_disorder','mental_fitness',data,kind='reg',color='blue')
plt.show()


# In[17]:


sns.pairplot(data,corner=True)
plt.show()


# In[19]:


fig = px.pie(data, values='mental_fitness', names='Year')
fig.show()


# In[ ]:





# In[20]:


df = data.copy()


# In[21]:


df.head()


# In[22]:


df.info()


# In[23]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in df.columns:
    if df[i].dtype == 'object':
        df[i]=l.fit_transform(df[i])


# In[24]:


X = df.drop('mental_fitness',axis=1)
y = df['mental_fitness']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)


# ### Linear Regression

# In[27]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression()
lr.fit(xtrain,ytrain)

# model evaluation for training set
ytrain_pred = lr.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
ytest_pred = lr.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# ### Decision Tree

# In[26]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(xtrain, ytrain)

# model evaluation for training set
ytrain_pred = rf.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
ytest_pred = rf.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[ ]:




