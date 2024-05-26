#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required libraries
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes, ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle


# In[2]:


# Load the dataset
breast_cancer = datasets.load_breast_cancer()

# breast_cancer=pd.read_csv('breast_cancer.csv')
breast_cancer


# In[3]:


X, y = breast_cancer.data, breast_cancer.target


# In[4]:


X


# In[5]:


y


# In[6]:


# Split the train and test samples

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    stratify = y, random_state = 123)


# In[7]:


### k-Nearest Neighbors (k-NN) with GridSearchCV
knn = neighbors.KNeighborsClassifier()

params_knn = {'n_neighbors': np.arange(1, 25)}

knn_gs = GridSearchCV(knn, params_knn, cv = 5)

knn_gs.fit(X_train, y_train)
knn_best = knn_gs.best_estimator_

knn_gs_predictions = knn_gs.predict(X_test)


# In[8]:


### Random Forest Classifier with GridSearchCV
rf = ensemble.RandomForestClassifier(random_state = 0)

params_rf = {'n_estimators': [50, 100, 200]}

rf_gs = GridSearchCV(rf, params_rf, cv = 5)

rf_gs.fit(X_train, y_train)
rf_best = rf_gs.best_estimator_

rf_gs_predictions = rf_gs.predict(X_test)


# In[9]:


### Logistic Regression with GridSearchCV
log_reg = linear_model.LogisticRegression(random_state = 123, solver = "liblinear", 
                                         penalty = "l2", max_iter = 5000)

C = np.logspace(1, 4, 10)
params_lr = dict(C = C)

lr_gs = GridSearchCV(log_reg, params_lr, cv = 5, verbose = 0)

lr_gs.fit(X_train, y_train)
lr_best = lr_gs.best_estimator_

log_reg_predictions = lr_gs.predict(X_test)


# In[10]:


# Combine all 3 models using VotingClassifier with voting = "soft" parameter
estimators = [('knn', knn_best), ('rf', rf_best), ('log_reg', lr_best)]

ensemble_S = VotingClassifier(estimators, voting = "soft")

soft_voting = ensemble_S.fit(X_train, y_train)


# In[11]:


# Save model
pickle.dump(soft_voting, open('soft_voting.pkl', 'wb'))


# Load the saved model
model = pickle.load(open('soft_voting.pkl', 'rb'))
model


# In[12]:


print("knn_gs.score: ", knn_gs.score(X_test, y_test))
# Output: knn_gs.score:

print("rf_gs.score: ", rf_gs.score(X_test, y_test))
# Output: rf_gs.score:

print("log_reg.score: ", lr_gs.score(X_test, y_test))
# Output: log_reg.score:


# In[13]:


print("ensemble.score: ", ensemble_S.score(X_test, y_test))
# Output: ensemble.score:


# In[ ]:




