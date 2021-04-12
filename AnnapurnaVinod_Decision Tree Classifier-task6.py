#!/usr/bin/env python
# coding: utf-8

# *Based on the datasets given, we classify a flower group(Iris) into different Species by using the Decision Tree Classifier Algorithm*

# In[1]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


e_iris=pd.read_csv(r"C:\Users\Annapurna Vinod\Downloads\Iris.csv")
e_iris.head()


# In[3]:


e_iris.describe()


# In[4]:


e_iris["Species"].value_counts()


# In[5]:


e_iris.isna().sum()


# In[6]:


e_iris.drop('Id',axis=1,inplace=True)


# 2D Scatterplot Rep

# In[7]:


e_iris.plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm')
plt.show()


# In[8]:


sns.set_style("dark")
sns.FacetGrid(e_iris,hue="Species", height=4).map(plt.scatter,"SepalLengthCm", "SepalWidthCm").add_legend();
plt.show()


# 3D Scatterplot Rep

# In[9]:


plt.close()
sns.set_style('dark')
sns.pairplot(e_iris,hue="Species",height=3)
plt.show()


# Training & Testing the dataset

# In[29]:


train,test=train_test_split(e_iris,test_size=0.4)
print(train.shape)
print(test.shape)


# In[30]:


train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]#training data 
train_y=train.Species#training data ouput
test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] #test data
test_y =test.Species#testdata output


# Defining the Decision Tree Algorithm

# In[31]:


decisiont=DecisionTreeClassifier(max_depth=5,random_state=0)
model = DecisionTreeClassifier().fit(train_X,train_y)
print("decision tree classifier is set ")


# Now, we can predict using a sample data

# In[32]:


predict=model.predict(test_X)


# In[33]:


X=[[2.5,3.4,1.6,2.9]]
learn=model.predict(X)
print(learn)


# In[34]:


print('accuracy level of classifier is',"{:.4f}".format(metrics.accuracy_score(predict,test_y)))


# DECISION TREE CLASSIFIER HAS 96% ACCURACY

# In[16]:


fig=plt.figure(figsize=(40,30))
a=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
b=['Iris-Setosa','Iris-virginica','Iris-versicolor']
plot_tree(model,feature_names=a, class_names =b,filled=True)

