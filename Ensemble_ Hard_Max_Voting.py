#!/usr/bin/env python
# coding: utf-8
# CRISP-ML(Q):

Business Problem: There are a lot of assumptions in the diagnosis pertaining to cancer. In a few cases radiologists, pathologists and oncologists go wrong in diagnosing whether tumor is benign (non-cancerous) or malignant (cancerous). Hence team of physicians want us to build an AI application which will predict with confidence the presence of cancer in a patient. This will serve as a compliment to the physicians.

Business Objective: Maximize Cancer Detection

Business Constraints: Minimize Treatment Cost & Maximize Patient Convenience

Success Criteria:

Business success criteria: Increase the correct diagnosis of cancer in at least 96% of patients
Machine Learning success criteria: Achieve an accuracy of atleast 98%
Economic success criteria: Reducing medical expenses will improve trust of patients and thereby hospital will see an increase in revenue by atleast 12%

Data Collection:

Data is collected from the hospital for 569 patients. 30 features and 1 label comprise the feature set. Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)
# In[17]:


# Import the required libraries
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes, ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import pickle


# In[18]:


# Load the dataset
breast_cancer = datasets.load_breast_cancer()

breast_cancer


# In[19]:


X, y = breast_cancer.data, breast_cancer.target


# In[20]:


X


# In[21]:


y


# In[22]:


# Split the train and test samples

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    stratify = y, random_state = 123)


# In[24]:


### k-Nearest Neighbors (k-NN) with GridSearchCV
knn = neighbors.KNeighborsClassifier()

params_knn = {'n_neighbors': np.arange(1, 25)}

knn_gs = GridSearchCV(knn, params_knn, cv = 5)

knn_gs.fit(X_train, y_train)
knn_best = knn_gs.best_estimator_
knn_best


# In[25]:


### Random Forest Classifier with GridSearchCV
rf = ensemble.RandomForestClassifier(random_state = 0)

params_rf = {'n_estimators': [50, 100, 200]}

rf_gs = GridSearchCV(rf, params_rf, cv = 5)

rf_gs.fit(X_train, y_train)
rf_best = rf_gs.best_estimator_


# In[26]:


### Logistic Regression with GridSearchCV
log_reg = linear_model.LogisticRegression(random_state = 123, solver = "liblinear", 
                                          penalty = "l2", max_iter = 5000)
C = np.logspace(1, 4, 10)
params_lr = dict(C = C)

lr_gs = GridSearchCV(log_reg, params_lr, cv = 5, verbose = 0)

lr_gs.fit(X_train, y_train)
lr_best = lr_gs.best_estimator_


# In[27]:


# Combine all three Voting Ensembles
estimators = [('knn', knn_best), ('rf', rf_best), ('log_reg', lr_best)]

# Instantiate the voting classifier
ensemble_H = VotingClassifier(estimators, voting = "hard")

# Fit classifier with the training data
hard_voting = ensemble_H.fit(X_train, y_train)


# In[28]:


# Save the voting classifier
pickle.dump(hard_voting, open('hard_voting.pkl', 'wb'))

# Loading a saved model
model = pickle.load(open('hard_voting.pkl', 'rb'))
model


# In[15]:


print("knn_gs.score: ", knn_best.score(X_test, y_test))
# Output: knn_gs.score:

print("rf_gs.score: ", rf_best.score(X_test, y_test))
# Output: rf_gs.score:

print("log_reg.score: ", lr_best.score(X_test, y_test))
# Output: log_reg.score:


# In[16]:


print("ensemble.score: ", ensemble_H.score(X_test, y_test))
# Output: ensemble.score:


# In[ ]:




