from sklearn import datasets, naive_bayes, ensemble, neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

breast_cancer = datasets.load_breast_cancer()

X, y = breast_cancer.data, breast_cancer.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

param_grid = {'n_neighbors' : np.arange(1, 25)}

grid_knn = GridSearchCV(knn, param_grid,  cv = 5)

grid_knn.fit(x_train, y_train)

grid.best_estimator_
grid.best_score_
grid.best_params_


rf = ensemble.RandomForestClassifier(random_state = 0)

param_grid_rf = {'n_estimators' : [50, 100, 200]}

grid_rf = GridSearchCV(rf, param_grid_rf, cv = 5)

grid_rf.fit(x_train, y_train)

grid_rf.best_estimator_
grid_rf.best_score_
grid_rf.best_params_

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state = 123, solver = "liblinear", 
                                          penalty = "l2", max_iter = 5000)

C = np.logspace(1, 4, 10)
params_lr = dict(C = C)

lr_gs = GridSearchCV(log_reg, params_lr, cv = 5, verbose = 0)

lr_gs.fit(x_train, y_train)
lr_gs.best_estimator_
lr_gs.best_params_

estimators = [('knn', grid.best_estimator_), ('rf', grid_rf.best_estimator_), ('log_reg', lr_gs.best_estimator_)]

hard_voting = ensemble.VotingClassifier(estimators)
hard_votingfit = hard_voting.fit(x_train, y_train)

grid.score(x_test, y_test)
grid_rf.score(x_test, y_test)
lr_gs.score(x_test, y_test)

hard_voting.score(x_test, y_test)

############################ SOFT VOTING #################################

from sklearn import ensemble, naive_bayes, neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

df = datasets.load_breast_cancer()


X = df.data
Y = df.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 0)
knn = neighbors.KNeighborsClassifier()

knn_grid = {'n_neighbors' : np.arange(1, 50)}

knn_grid = GridSearchCV(knn, knn_grid, cv = 5)

knn_model = knn_grid.fit(x_train, y_train)

knn_grid.best_params_
knn_model.best_params_

rf = ensemble.RandomForestClassifier(random_state = 0)

rf_parm = {'n_estimators' : [25, 50, 100, 150]}

rf_grid = GridSearchCV(rf, rf_parm, cv = 5)

rf_model = rf_grid.fit(x_train, y_train)

rf_model.best_estimator_
rf_model.best_params_

from sklearn import linear_model

lr = linear_model.LogisticRegression(random_state = 123, solver = "liblinear", 
                                         penalty = "l2", max_iter = 5000)

C = np.logspace(1, 4, 10)
params_lr = dict(C = C)

lr_grid = GridSearchCV(lr, params_lr, cv = 5)

lr_model = lr_grid.fit(x_train, y_train)

lr_model.best_estimator_
lr_model.best_params_

from sklearn.ensemble import VotingClassifier

estimators = [('knn', knn_grid.best_estimator_), ('rf', rf_grid.best_estimator_), ('lr', lr_grid.best_estimator_)]

voting = VotingClassifier(estimators, voting = 'soft')

vot_model = voting.fit(x_train, y_train)


knn_model.score(x_test, y_test)
rf_model.score(x_test, y_test)
lr_model.score(x_test, y_test)
vot_model.score(x_test, y_test)
