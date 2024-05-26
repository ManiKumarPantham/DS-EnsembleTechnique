#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics


# In[2]:


# Load the dataset
iris = load_iris()

iris


# In[3]:


# Create the dataframe
df_features = pd.DataFrame(data = iris.data, columns = iris.feature_names)

df_target = pd.DataFrame(data = iris.target, columns = ['species'])

final = pd.concat([df_features, df_target], axis = 1)


# # Save the Dataframe into a CSV file
# final.to_csv('iris.csv', index = False)
# 
# final = pd.read_csv('iris.csv')

# In[4]:


final


# In[5]:


X = np.array(final.iloc[:, :4]) # Predictors 
Y = np.array(final['species']) # Target


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    stratify = Y, 
                                                    random_state = 42)


# In[7]:


X_train[0:5]


# In[8]:


y_train[0:5]


# In[9]:


# Base estimators

estimators = [('rf', RandomForestClassifier(n_estimators = 10, random_state = 42)),
              ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state = 42)))]


# In[10]:


# Meta Model stacked on top of base estimators

clf = StackingClassifier(estimators = estimators, final_estimator = LogisticRegression())


# In[11]:


# Fit the model on traing data

stacking = clf.fit(X_train, y_train)


# In[12]:


# Accuracy

stacking.score(X_test, y_test)


# In[13]:


# Save the Stacking model 
pickle.dump(stacking, open('stacking_iris.pkl', 'wb'))


# In[14]:


# Load the saved model

model = pickle.load(open('stacking_iris.pkl', 'rb'))
model


# In[15]:


# Load test dataset
test = pd.read_csv(r'C:\Users\Bharani Kumar\Desktop\Data Science using Python & R\Version 2 slides\stacking_classify_flask_new\stacking_classify_flask_new\iris_test.csv')
test


# In[16]:


pred = model.predict(test)


# In[17]:


pred


# In[18]:


y_test


# In[ ]:


# score = metrics.accuracy_score(y_test, pred)
# score

