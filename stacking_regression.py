#!/usr/bin/env python
# coding: utf-8

# In[20]:


# Stacking Regression Using scikit-learn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics


# In[21]:


# Load the dataset
diabetes = load_diabetes()


# In[22]:


# # Load the dataframe
df_features = pd.DataFrame(data = diabetes.data, columns = diabetes.feature_names)

df_target = pd.DataFrame(data = diabetes.target, columns = ['target'])


# In[23]:


final = pd.concat([df_features, df_target], axis = 1)

final


# ### Save the data frame into csv file
# final.to_csv('diabetes_new.csv', index = False)
# 
# final = pd.read_csv('diabetes_new.csv')
# 
# final
# final.info()

# In[24]:


X = np.array(final.iloc[:, :10]) # Predictors 

y = np.array(final['target']) # Target


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


# In[26]:


# Base estimators

estimators = [("lr", RidgeCV()), ("svr", LinearSVR(random_state = 42))]


# In[27]:


# Meta Model stacked on top of base estimators

reg = StackingRegressor(estimators = estimators,
                        final_estimator = RandomForestRegressor(n_estimators = 10,
                                                                random_state = 42))


# In[28]:


stacking_reg = reg.fit(X_train, y_train)
stacking_reg


# In[29]:


# Save the ML model
pickle.dump(stacking_reg, open('stacking_reg_diabetes.pkl', 'wb'))

# Load the saved model
model = pickle.load(open('stacking_reg_diabetes.pkl','rb'))


# In[30]:


pred = model.predict(X_test)

pred


# In[31]:


r2_score = model.score(X_test, y_test)


# In[32]:


print(r2_score)


# In[33]:


test = pd.read_csv(r'C:\Users\Bharani Kumar\Desktop\Data Science using Python & R\Version 2 slides\stacking_regression_flask_new\stacking_regression_flask_new\diabetes_test.csv')


# In[34]:


test_pred = model.predict(test)


# In[35]:


test_pred


# In[ ]:




