#############################################################################################

1.	Given is the diabetes dataset. Build an ensemble model to correctly classify the outcome 
variable and improve your model prediction by using GridSearchCV. 
You must apply Bagging, Boosting, Stacking, and Voting on the dataset.  

#############################################################################################

# Importing requried libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier  
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb

data = pd.read_csv("D:/Hands on/19_Ensembling Technique_2/Assignment/Diabeted_Ensemble.csv")

# Information of the dataset
data.info()

# Statistical calculations of the dataset
data.describe()

# First moment business decession
data.mean()

data.median()

data.mode()

# Second moment business decession
data.var()

data.std()

# Third moment business decession
data.skew()

# Fourth moment business decession
data.kurt()

# Pairplot
sns.pairplot(data)

# Correlation coefficient
data.corr()

# Checking for duplicates
data.duplicated().sum()

# Checking for null values
data.isnull().sum()

# Spliting the data into X and Y
X = data.iloc[:, 0:8]

Y = data.iloc[:, -1]

# Converting Dependent variable into 1 and 0
Y = np.where(Y == 'YES', 1, 0)

# Boxplot
X.plot(kind = 'box', sharey = False, subplots = True)
plt.subplots_adjust(wspace = 0.5)

# Winsorization
winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = list(X.columns))

X_winsor = winsor.fit_transform(X)

# Boxplot
X.plot(kind = 'box', sharey = False, subplots = True)
plt.subplots_adjust(wspace = 0.5)

# Creating a Robust scaler object
robust = RobustScaler()

X_robust = pd.DataFrame(robust.fit_transform(X_winsor), columns = X_winsor.columns)

# Spliting the into train and test
x_train, x_test, y_train, y_test = train_test_split(X_robust, Y, random_state = 0, stratify = Y)

# Converting into DataFrame
pd.DataFrame(y_train).value_counts()

pd.DataFrame(y_test).value_counts()

# Voting
# Creating a KNeighborsClassifier object
knn = KNeighborsClassifier()

# Creating param set
knn_parm = {'n_neighbors' : np.arange(5, 15)}

# Creating a GridSearchCV object
grid_knn = GridSearchCV(knn, param_grid = knn_parm, verbose = 0, cv = 5)
knn_model = grid_knn.fit(x_train, y_train)

# Best estimator and best Score
knn_model.best_estimator_
knn_model.best_score_

# RandomeForestClassifier object
rf = RandomForestClassifier()

# Creating param grid
rf_parm = {'n_estimators' : [25, 50, 75, 100, 125, 150]}

# Creating a GridSearchCV object
rf_grid = GridSearchCV(rf, param_grid = rf_parm, verbose = 0, cv = 5)

# Building a model
rf_model = rf_grid.fit(x_train, y_train)
rf_model.best_estimator_
rf_model.best_params_
rf_model.best_score_

# Creating a LogisticRegression object
lr = LogisticRegression(random_state = 123, solver = "liblinear", 
                                          penalty = "l2", max_iter = 5000)
C = np.logspace(1, 4, 10)
lr_parm = dict(C = C)

# Creating a GridSearchCV object
lr_grid = GridSearchCV(lr, param_grid = lr_parm, verbose = 0, cv = 5)

# Building a model 
lr_model = lr_grid.fit(x_train, y_train)
lr_model.best_estimator_
lr_model.best_params_
lr_model.best_score_

# Creating estimatores
vote_estimators = [('knn', knn_model.best_estimator_), ('rf', rf_model.best_estimator_), ('lr', lr_model.best_estimator_)]

# Creating a VotingClassifier object
Vote = VotingClassifier(estimators = vote_estimators, voting = 'soft')

# Creating a model
Vote_model = Vote.fit(x_train, y_train)

# Scoring on each building
grid_knn.score(x_test, y_test)
rf_grid.score(x_test, y_test)
lr_grid.score(x_test, y_test)
Vote_model.score(x_test, y_test)

# Stacking
estimators = [('knn', KNeighborsClassifier()), ('svm', LinearSVC()), ('DT', DecisionTreeClassifier())]

# StackingClassifier object
stack = StackingClassifier(estimators = estimators, final_estimator = RandomForestClassifier())

# Building a model
stack_model = stack.fit(x_train, y_train)

# Prediction on Test data
test_pred = stack_model.predict(x_test)

# Confusion matrix
confusion_matrix(y_test, test_pred)

# Accuracy on test
accuracy_score(y_test, test_pred)

# Prediction on Train data
train_pred = stack_model.predict(x_train)

# Confusion matrix
confusion_matrix(y_train, train_pred)

# Accuracy on Training data
accuracy_score(y_train, train_pred)

# Bagging
# Creating a DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion = 'entropy', max_features = 'sqrt')

# Creating a BaggingClassifier
bagging = BaggingClassifier(base_estimator = dtree, n_estimators = 100, bootstrap = True, n_jobs = 1)

# Building a model
bag_model = bagging.fit(x_train, y_train)

# Prediction on test data
test_pred = bag_model.predict(x_test)

# Consfusion matrix
confusion_matrix(y_test, test_pred)

# Accuracy on test data
accuracy_score(y_test, test_pred)

# Prediction on train dadta
train_pred = bag_model.predict(x_train)

# Confusion matrix
confusion_matrix(y_train, train_pred)

# Accuracy on training data
accuracy_score(y_train, train_pred)

# Boosting
# Creating a AdaBoostClassifier object
adaboost = AdaBoostClassifier(n_estimators = 10, learning_rate = 0.02)

# Building a model on train data
ada_model = adaboost.fit(x_train, y_train)

# Prediction on test data
test_pred = ada_model.predict(x_test)

# Confusion matrix
confusion_matrix(y_test, test_pred)

# Accuracy on test data
accuracy_score(y_test, test_pred)

# Prediction train data
train_pred = ada_model.predict(x_train)

# Confusion matrix
confusion_matrix(y_train, train_pred)

# Accuracy score on training
accuracy_score(y_train, train_pred)

#Gradient Boosting
# Creating a GradientBoostingClassifier object
gradient = GradientBoostingClassifier()

# Builing a model on train data
grad_model = gradient.fit(x_train, y_train)

# Prediction on test data
test_pred = grad_model.predict(x_test)

# Confusion matrix
confusion_matrix(y_test, test_pred)

# Accuracy score on test dadta
accuracy_score(y_test, test_pred)

# Prediction on train data
train_pred = grad_model.predict(x_train)

# Confusion matrix
confusion_matrix(y_train, train_pred)

# Accuracy on train data
accuracy_score(y_train, train_pred)

# XGboosting
# Creating XGBClassifier object
xgb1 = xgb.XGBClassifier(max_depths = 5, n_estimators = 100, learning_rate = 0.02, n_jobs = -1)

# Building a model
xgb_model = xgb1.fit(x_train, y_train)

# Prediction on test data
test_pred = xgb_model.predict(x_test)

# Confusion matrix
confusion_matrix(y_test, test_pred)

# Accuracy
accuracy_score(y_test, test_pred)

# Prediction on train data
train_pred = xgb_model.predict(x_train)

# Confusion matrix
confusion_matrix(y_train, train_pred)

# Accuracy on train data
accuracy_score(y_train, train_pred)

#############################################################################################
2.	Most cancers form a lump called a tumour. But not all lumps are cancerous. Doctors extract a 
sample from the lump and examine it to find out if it’s cancer or not. Lumps that are not 
cancerous are called benign (be-NINE). Lumps that are cancerous are called 
malignant (muh-LIG-nunt). Obtaining incorrect results (false positives and false negatives) 
especially in a medical condition such as cancer is dangerous. So, perform Bagging, 
Boosting, Stacking, and Voting algorithms to increase model performance and provide your 
insights in the documentation.

############################################################################################

# Importing required libraries into Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

import xgboost as xb

# importing data into Python
data = pd.read_csv("D:/Hands on/19_Ensembling Technique_2/Assignment/Tumor_Ensemble.csv")

# Droping id column
data.drop(['id'], axis = 1, inplace = True)

# Information of the dataset
data.info()

# Statistical calculations of the dataset
data.describe()

# First moment business decession
data.mean()

data.median()

data.mode()

# Second moment business decession
data.var()

data.std()

# Third moment business decession
data.skew()

# Fourth moment business decession
data.kurt()

# Pairplot
sns.pairplot(data)

# Correlation coefficient
data.corr()

# Checking for duplicates
data.duplicated().sum()

# Checking for null values
data.isnull().sum()

# Spliting the data into X and Y
X = data.iloc[:, 1:]

Y = data.iloc[:, 0]

# Converting output variables into Binary
Y = np.where(Y == 'M', 1, 0)

# Boxplot 
for i in X.columns:
    plt.boxplot(X[i])
    plt.title('Box plot for ' + str(i))
    plt.show()
    

# Creating a Winsorizer object
winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = list(X.columns))

X_winsor = winsor.fit_transform(X)

# Boxplot  
for i in X.columns:
    plt.boxplot(X[i])
    plt.title('Box plot for ' + str(i))
    plt.show()
    

# Creating a RobustScaler object
robust = RobustScaler()

X_scale = pd.DataFrame(robust.fit_transform(X), columns = X_winsor.columns)

# Spliting into train and test
x_train, x_test, y_train, y_test = train_test_split(X_scale, Y, stratify = Y, random_state = 0)

# Voting
# Creating KNeighborsClassifier object
knn = KNeighborsClassifier()
knn_parm = {'n_neighbors' : np.arange(1, 20, 2)}

# Creating GridSearchCV object
knn_grid = GridSearchCV(knn, param_grid = knn_parm, cv = 4, verbose = 0)

# Building a train model
knn_model = knn_grid.fit(x_train, y_train)

# model parameters
knn_model.best_estimator_
knn_model.best_params_
knn_model.best_score_

# Creating a DecisionTreeClassifier
dt = DecisionTreeClassifier()

# Creating param list
dt_parm = {'max_depth' : np.arange(1, 10), 'min_samples_leaf' : np.arange(1, 5), 'min_samples_split' : np.arange(1, 5)}

# Creating a GridSearchCV obejct
dt_grid = GridSearchCV(dt, dt_parm, cv = 5, verbose = 0)

# Building a model
dt_model = dt_grid.fit(x_train, y_train)

# Model best estimators
dt_model.best_estimator_
dt_model.best_params_
dt_model.best_score_

# Creating a LogisticRegression object
lr = LogisticRegression(random_state = 123, solver = "liblinear", 
                                          penalty = "l2", max_iter = 5000)
C = np.logspace(1, 4, 10)
lr_parm = dict(C = C)

# Creating a GridSearchCV object
lr_grid = GridSearchCV(lr, lr_parm, cv = 5, verbose = 0)

# Building a model
lr_model = lr_grid.fit(x_train, y_train)

lr_model.best_estimator_
lr_model.best_params_
lr_model.best_score_

# Creating a estimators
estimators = [('knn', knn_model.best_estimator_), ('dt', dt_model.best_estimator_), ('lr', lr_model.best_estimator_)]

# Voting object
vote = VotingClassifier(estimators = estimators, voting = 'hard')

# Building a model train data
vote_model = vote.fit(x_train, y_train)

knn_model.score(x_test, y_test)
dt_model.score(x_test, y_test)
lr_model.score(x_test, y_test)
vote_model.score(x_test, y_test)

# Stacking
# Creating estimators
estimators = [('knn', KNeighborsClassifier()), ('lr', LogisticRegression())]

# Creating a StackingClassifier object
stack = StackingClassifier(estimators = estimators, 
                           final_estimator = RandomForestClassifier(n_estimators = 10, random_state = 0))


# Building a model on train data
stack_model = stack.fit(x_train, y_train)

# Prediction on test data
test_pred = stack_model.predict(x_test)

# Confusion matrix
confusion_matrix(y_test, test_pred)

# Accuracy score
accuracy_score(y_test, test_pred)

# Prediction on train data
train_pred = stack_model.predict(x_train)

# Confusion matrix
confusion_matrix(y_train, train_pred)

# Accuracy score
accuracy_score(y_train, train_pred)

# Bagging
# Creating a KNeighborsClassifier object
knn = KNeighborsClassifier(n_neighbors = 5)

# Creating a BaggingClassifier object
bag = BaggingClassifier(base_estimator = knn, n_estimators = 5, bootstrap = True, random_state = 0)

# Building a model
bag_model = bag.fit(x_train, y_train)

# Prediction on test data
test_pred = stack_model.predict(x_test)
confusion_matrix(y_test, test_pred)
accuracy_score(y_test, test_pred)

# Prediction on train data
train_pred = stack_model.predict(x_train)
confusion_matrix(y_train, train_pred)
accuracy_score(y_train, train_pred)

# Adaboost 
# AdaBoostClassifier obejct
ada = AdaBoostClassifier(n_estimators = 10, learning_rate = 0.3)

# Building a model
ada_model = ada.fit(x_train, y_train)

# Prediction on test data
test_pred = ada_model.predict(x_test)
confusion_matrix(y_test, test_pred)
accuracy_score(y_test, test_pred)

# Prediction on train data
train_pred = ada_model.predict(x_train)
confusion_matrix(y_train, train_pred)
accuracy_score(y_train, train_pred)

# GradientBoosting
# GradientBoosting object
gradboost = GradientBoostingClassifier()

# Building a model
gradboost_model = gradboost.fit(x_train, y_train)

# Prediction on test data
test_pred = gradboost_model.predict(x_test)
confusion_matrix(y_test, test_pred)
accuracy_score(y_test, test_pred)

# Prediction on train data
train_pred = gradboost_model.predict(x_train)
confusion_matrix(y_train, train_pred)
accuracy_score(y_train, train_pred)

#XGBoost
# XGBoost object
xgb = xb.XGBClassifier(learning_rate = 0.02, random_state = 0, n_estimators = 10)

# Building a model
xgb_model = xgb.fit(x_train, y_train)

# Prediction on test data
test_pred = xgb_model.predict(x_test)
confusion_matrix(y_test, test_pred)
accuracy_score(y_test, test_pred)

# Prediction on train data
train_pred = xgb_model.predict(x_train)
confusion_matrix(y_train, train_pred)
accuracy_score(y_train, train_pred)

# Voting
# Creating estimators
estimators = [('bagging', bag), ('ada', ada), ('gboost', gradboost), ('xgb', xgb), ('stack', stack)]

# Voting object
vote1 = VotingClassifier(estimators = estimators, voting = 'hard')

# Builing a model on train data
vote1_model = vote1.fit(x_train, y_train)

# Prediction on test data
test_pred = vote1_model.predict(x_test)
confusion_matrix(y_test, test_pred)
accuracy_score(y_test, test_pred)

# Predction on train data
train_pred = vote1_model.predict(x_train)
confusion_matrix(y_train, train_pred)
accuracy_score(y_train, train_pred)

##############################################################################################

3.	A sample of global companies and their ratings are given for the cocoa bean production 
along with the location of the beans being used. Identify the important features in 
the analysis and accurately classify the companies based on their ratings and draw 
insights from the data. Buildensemble models such as Bagging, Boosting, Stacking, 
and Voting on the dataset given.

#############################################################################################

# Importing required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from feature_engine.outliers import Winsorizer
from sklearn.metrics import r2_score

from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

import xgboost as xb

# Reading a data into Python
data = pd.read_excel(r"D:/Hands on/19_Ensembling Technique_2/Assignment/Coca_Rating_Ensemble.xlsx")

# Information of the dataset
data.info()

# Statistical calculations of the dataset
data.describe()

# First moment business decession
data.mean()

data.median()

data.mode()

# Second moment business decession
data.var()

data.std()

# Third moment business decession
data.skew()

# Fourth moment business decession
data.kurt()

# Pairplot
sns.pairplot(data)

data.corr()

# Auto EDA
import dtale
d = dtale.show(data)
d.open_browser()

# Data types
data.dtypes

# Droping columns
data1 = data.drop(labels = (['Origin', 'Bean_Type']), axis = 1)

# Checking for duplicates
data1.duplicated().sum()

# Checking for null values
data1.isna().sum()

'''from sklearn.impute import SimpleImputer

mode_impute = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

data['Bean_Type'] = mode_impute.fit_transform(data[['Bean_Type']])

data['Origin'] = mode_impute.fit_transform(data[['Origin']])

data.isna().sum()
'''
# Boxplot
data1.plot(kind = 'box', sharey = False, subplots = True)

# Spliting into num and cat features
cat_features = data1.select_dtypes(include = object)
num_features = data1.select_dtypes(exclude = object)

# Creating dummy columns
cat_dummies = pd.get_dummies(cat_features, drop_first = True)

# Boxplot
num_features.plot(kind = 'box', sharey = False, subplots = True)

# Winsorization for outlier treatment
pwinsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['Cocoa_Percent'])
rwinsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['Rating'])

num_features['Cocoa_Percent'] = pd.DataFrame(pwinsor.fit_transform(num_features[['Cocoa_Percent']]))
num_features['Rating'] = pd.DataFrame(rwinsor.fit_transform(num_features[['Rating']]))

# Re-aranging the columns
num_features = num_features[['Rating', 'REF', 'Review', 'Cocoa_Percent']]

# Creating a Scaling object
min = MinMaxScaler()

num_trans = pd.DataFrame(min.fit_transform(num_features), columns = num_features.columns)

# Concating two data frame
new_data = pd.concat([num_trans, cat_dummies], axis = 1)

# Spliting into X and Y
Y = new_data['Rating']

X = new_data.iloc[:, 1:]

# Spliting into train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.20, stratify = Y)

# Stacking
# Creating estimators
estimators = [('knn', KNeighborsRegressor()), ('dt', DecisionTreeRegressor())]

# Stacking object
stack = StackingRegressor(estimators = estimators, 
                           final_estimator = RandomForestRegressor(n_estimators = 10, random_state = 0))


# Building model on train data
stack_model = stack.fit(x_train, y_train)

# Prediction on test data
test_pred = stack_model.predict(x_test)
test_resid = y_test - test_pred
test_rmse = np.sqrt(np.mean(test_resid * test_resid))

# R2 score
r2_score(y_test, test_pred)

# Prediction on train data
train_pred = stack_model.predict(x_train)
train_resid = y_train - train_pred
train_rmse = np.sqrt(np.mean(train_resid * train_resid))

r2_score(y_train, train_pred)

# Bagging
# Creating a KNeighborsRegressor obejct
knn = KNeighborsRegressor(n_neighbors = 5)

# Creating BaggingRegressor object
bag = BaggingRegressor(base_estimator = knn, n_estimators = 5, bootstrap = True, random_state = 0)

# Building a model
bag_model = bag.fit(x_train, y_train)

# Prediction on test data
test_pred = bag_model.predict(x_test)
test_resid = y_test - test_pred
test_rmse = np.sqrt(np.mean(test_resid * test_resid))

r2_score(y_test, test_pred)

# Prediction on train data
train_pred = bag_model.predict(x_train)
train_resid = y_train - train_pred
train_rmse = np.sqrt(np.mean(train_resid * train_resid))

r2_score(y_train, train_pred)

# Adaboost
# Creating a AdaBoostRegressor object
ada = AdaBoostRegressor(n_estimators = 10, learning_rate = 0.3)

# Building a model
ada_model = ada.fit(x_train, y_train)

# Prediction on test data
test_pred = ada_model.predict(x_test)
test_resid = y_test - test_pred
test_rmse = np.sqrt(np.mean(test_resid * test_resid))

r2_score(y_test, test_pred)

# Prediction on train data
train_pred = ada_model.predict(x_train)
train_resid = y_train - train_pred
train_rmse = np.sqrt(np.mean(train_resid * train_resid))

r2_score(y_train, train_pred)

# GradientBoosting
# GradientBoostingRegressor object
gradboost = GradientBoostingRegressor()

# Building a model
gradboost_model = gradboost.fit(x_train, y_train)

# Prediction on test data
test_pred = gradboost_model.predict(x_test)
test_resid = y_test - test_pred
test_rmse = np.sqrt(np.mean(test_resid * test_resid))

r2_score(y_test, test_pred)

# Prediction on train data
train_pred = gradboost_model.predict(x_train)
train_resid = y_train - train_pred
train_rmse = np.sqrt(np.mean(train_resid * train_resid))

r2_score(y_train, train_pred)

#XGBoost
# XGBRegressor object
xgb = xb.XGBRegressor(learning_rate = 0.02, random_state = 0, n_estimators = 10)

# Building a model 
xgb_model = xgb.fit(x_train, y_train)

# Prediction on test data
test_pred = xgb_model.predict(x_test)
test_resid = y_test - test_pred
test_rmse = np.sqrt(np.mean(test_resid * test_resid))

r2_score(y_test, test_pred)

# Prediction on train data
train_pred = xgb_model.predict(x_train)
train_resid = y_train - train_pred
train_rmse = np.sqrt(np.mean(train_resid * train_resid))

r2_score(y_train, train_pred)

# Voting
# Creating estimators
estimators = [('bagging', bag), ('ada', ada), ('gboost', gradboost), ('xgb', xgb), ('stack', stack)]

# Voting object
vote1 = VotingRegressor(estimators = estimators)

# Model building
vote1_model = vote1.fit(x_train, y_train)

# Prediction on test data
test_pred = vote1_model.predict(x_test)
test_resid = y_test - test_pred
test_rmse = np.sqrt(np.mean(test_resid * test_resid))

r2_score(y_test, test_pred)

# Prediction on train data
train_pred = vote1_model.predict(x_train)
train_resid = y_train - train_pred
train_rmse = np.sqrt(np.mean(train_resid * train_resid))

r2_score(y_train, train_pred)
    
###########################################################################################

4.	Data privacy is always an important factor to safeguard their customers' details. 
For this, password strength is an important metric to track. 
Build anensemble model to classify the user’s password strength.

############################################################################################

# Importing required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

import xgboost as xb

# Reading a data into python
data = pd.read_excel(r"D:/Hands on/19_Ensembling Technique_2/Assignment/Ensemble_Password_Strength.xlsx")

# Columns of the data set
data.columns

# Values and its count
data.characters.value_counts()

# Number of unique values
data.characters.nunique()

# Information of the dataset
data.info()

# Statistical calculations of the dataset
data.describe()

# First moment business decession
data.mean()

data.median()

data.mode()

# Second moment business decession
data.var()

data.std()

# Third moment business decession
data.skew()

# Fourth moment business decession
data.kurt()

# Pairplot
sns.pairplot(data)

# Correlation coefficient
data.corr()

# Chekcing for duplicates
data.duplicated().sum()

# Checking null values
data.isna().sum()

# Values and its count
data.characters.value_counts()

# Values and its count
data.characters_strength.value_counts()

# Five sample
data.sample(5)

# Creating dummy columns
data1 = pd.get_dummies(data, drop_first = True)

# Spliting the data into X and Y
X = data1.iloc[:, 1:]

Y = pd.DataFrame(data1.characters_strength)

# Spliting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 0, stratify = Y)

# Stacking
# Estimators
estimators = [('knn', KNeighborsClassifier()), ('dt', DecisionTreeClassifier())]

# Stacking Object
stack = StackingClassifier(estimators = estimators, 
                        final_estimator = RandomForestClassifier(n_estimators = 10, random_state = 0))

# Building model
stack_model = stack.fit(x_train, y_train)

# Testing on prediction
test_pred = stack_model.predict(x_test)

# Confusion matrix
confusion_matrix(y_test, test_pred)

# Accuracy on test data
accuracy_score(y_test, test_pred)

# Prediction on train data
train_pred = stack_model.predict(x_train)

confusion_matrix(y_train, train_pred)

accuracy_score(y_train, train_pred)

# Bagging
# DecisionTreeClassifier object
dt = DecisionTreeClassifier()
bag = BaggingClassifier(base_estimator = dt, n_estimators = 10, bootstrap =True, random_state = 0)

# Building a model
bag_model =  bag.fit(x_train, y_train)

# Prediction on test data
test_pred = bag_model.predict(x_test)

confusion_matrix(y_test, test_pred)

accuracy_score(y_test, test_pred)

# Prediction on train data
train_pred = bag_model.predict(x_train)

confusion_matrix(y_train, train_pred)

accuracy_score(y_train, train_pred)

# Adaboost
# AdaBoostClassifier object
ada = AdaBoostClassifier(n_estimators = 10, learning_rate = 0.3)

# Building a model
ada_model = ada.fit(x_train, y_train)

# Prediction on test data
test_pred = ada_model.predict(x_test)
confusion_matrix(y_test, test_pred)
accuracy_score(y_test, test_pred)

# Prediction on train data
train_pred = ada_model.predict(x_train)
confusion_matrix(y_train, train_pred)
accuracy_score(y_train, train_pred)

# GradientBoosting
# GradientBoostingClassifier object
gradboost = GradientBoostingClassifier()

# Building a model
gradboost_model = gradboost.fit(x_train, y_train)

# Prediction on test data
test_pred = gradboost_model.predict(x_test)
confusion_matrix(y_test, test_pred)
accuracy_score(y_test, test_pred)

# Prediction on train data
train_pred = gradboost_model.predict(x_train)
confusion_matrix(y_train, train_pred)
accuracy_score(y_train, train_pred)

#XGBoost
# XGBClassifier object
xgb = xb.XGBClassifier(learning_rate = 0.02, random_state = 0, n_estimators = 10)

# Building model
xgb_model = xgb.fit(x_train.values, y_train.values)

# Prediction on test values
test_pred = xgb_model.predict(x_test.values)
confusion_matrix(y_test, test_pred)
accuracy_score(y_test, test_pred)

# Prediction on train data
train_pred = xgb_model.predict(x_train.values)
confusion_matrix(y_train, train_pred)
accuracy_score(y_train, train_pred)

# Voting
# Estimators
estimators = [('bagging', bag), ('ada', ada), ('gboost', gradboost), ('xgb', xgb), ('stack', stack)]

# Voting object
vote1 = VotingClassifier(estimators = estimators, voting = 'hard')

# Building a model
vote1_model = vote1.fit(x_train.values, y_train.values)

# Prediction on test data
test_pred = vote1_model.predict(x_test.values)
confusion_matrix(y_test, test_pred)
accuracy_score(y_test, test_pred)

# Prediction on train data
train_pred = vote1_model.predict(x_train.values)
confusion_matrix(y_train, train_pred)
accuracy_score(y_train, train_pred)
























