#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#Importing pandas


# In[2]:


a=pd.read_csv('parkinsons.csv')
#reading csv file 


# In[44]:


a.shape
#printing the dimentions of data set


# In[4]:


a.dtypes
# Attributes in data set
#here we have 24 attributes and status is the result, 1 for positive and 0 for negative for ParkinsonDisease


# In[5]:


a.head()
#pring the head of the data


# In[6]:


import seaborn as sns
#Importing Seaborn for better visulization of data


# In[7]:


sns.catplot(x='status',kind='count',data=a)
#printing how many of them are positive for Parkinsons


# In[8]:



for i in a:
    if i != 'status' and i != 'name':
        sns.catplot(x='status',y=i,kind='box',data=a)
        
#The boxplot shown below helps in identifying the difference in values with respect to the 'status' of the patient.


# In[9]:





# In[10]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
#importing numpy, MinMaxScaler


# In[11]:


features=a.drop(['status','name'],axis=1)
labels=a['status']

#droping "name" attribute as it provides no useful insight.
#taking status values in labels variable


# In[12]:


scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels
#normalize the data using the minmax scaler to bring the feature variables within the range -1 to 1


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=5)
#Splitting the data into traning set and test set


# In[25]:


from sklearn.svm import SVC

#Importing Support vector machine algorithm


# In[28]:


svm = SVC(kernel='linear', C=1, random_state=0)
svm.fit(x_train, y_train)
#fitting the data in svm algorithm
y_pred=svm.predict(x_test)
# print('misclassified samples: %d'%(y_test!=y_pred).sum())
# print('correctly classified samples: %d'%(y_test==y_pred).sum())


# In[29]:


from sklearn.metrics import accuracy_score


# In[45]:


print('Accuracy:',accuracy_score(y_test,y_pred)*100)


# In[46]:


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,y_pred))


# In[48]:


y_pred=pd.DataFrame(y_pred)
y_pred


# In[ ]:





# In[ ]:




