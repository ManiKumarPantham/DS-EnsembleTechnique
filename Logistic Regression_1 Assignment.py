################################################################################################
1.	A psychological study has been conducted by a team of students at a university on married 
couples to determine the cause of having an extra marital affair. They have surveyed and 
collected a sample of data on which they would like to do further analysis. 
Apply Logistic Regression on the data to correctly classify whether a given person will have an 
affair or not given the set of attributes. Convert the naffairs column to discrete binary type 
before proceeding with the algorithm.


###############################################################################################

#Importing required libraries into Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, auc, classification_report

# Reading the data into Python
data = pd.read_csv("D:/Hands on/25_Logistic Regression_1/Assignement/Affairs.csv")

# Droping un related features
data.drop(['Unnamed: 0'], axis = 1, inplace = True)

# Converting naffairs into binary values
data['naffairs'] = np.where(data['naffairs'] == 0, 0, 1)

# Checking for duplicates
data.duplicated().sum()

# Checking for null values
data.isnull().sum()

# Spliting the data into X and Y
Y = pd.DataFrame(data['naffairs'])

X = data.iloc[:, 1:]

# Boxplot
X.plot(kind = 'box', subplots = True, sharey = False)

# Building a model
logit_model = sm.Logit(Y, X).fit()

# Summary of the model
logit_model.summary2()
logit_model.summary()

# Prediction in input data
pred = logit_model.predict(X)

# ROC curve
fpr, tpr, threshold = roc_curve(Y, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = threshold[optimal_idx]
optimal_threshold

# Area under the curve
auc = auc(fpr, tpr)
print('Area under the curve is %f' %auc)

# Creating a new Predi column having all zeros
X['Predi'] = np.zeros(601)

# If the pred value is greater than optimal_threshold then make it as 1
X.loc[pred > optimal_threshold, 'Predi'] = 1

# Confusion matrix
confusion_matrix(Y, X.Predi)

# Accuracy of model
accuracy_score(Y, X.Predi)

# confusion matrix
confusion_matrix(X.Predi, Y)

# Accuracy of the model
accuracy_score(X.Predi, Y)

# Classification report
classification = classification_report(X.Predi, Y)

### PLOT FOR ROC
plt.plot(fpr, tpr, label = "AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 2)
plt.show()


# Spliting the data into train test 
x_train, x_test, y_train, y_test = train_test_split(X.iloc[:, 0:17], Y, random_state = 0, stratify = Y)

# Building the model on training data
final_model = sm.Logit(y_train, x_train).fit() 

# Summary of the model
final_model.summary()
final_model.summary2()

# Prediction on test data
test_pred = final_model.predict(x_test)
y_test["y_test_pred"] = np.zeros(151)

# converting the values into zeros and once
y_test.loc[test_pred > optimal_threshold, "y_test_pred"] = 1

# Area under the curve
auc3 = roc_auc_score(y_test.y_test_pred, y_test.naffairs)

# Confusion matrix
confusion_matrix(y_test.y_test_pred, y_test.naffairs)

# Accuracy 
accuracy_score(y_test.y_test_pred, y_test.naffairs)
class3 = classification_report(y_test.y_test_pred, y_test.naffairs)

# Prediction on training data
train_pred = final_model.predict(x_train)

# Creating a new feature with all zeros
y_train['train_pred'] = np.zeros(450)

# Converting into zeros and ones
y_train.loc[train_pred > optimal_threshold, 'train_pred'] = 1

# Area under the curve
auc1 = roc_auc_score(y_train["train_pred"], y_train.naffairs)

# Classification report
class1 = classification_report(y_train["train_pred"], y_train.naffairs)

# Confusion matrix
confusion_matrix(y_train["train_pred"], y_train.naffairs)

# Accuracy score
accuracy_score(y_train["train_pred"], y_train.naffairs)

############################################################################################

2.	In this time and age of widespread internet usage, effective and targeted marketing 
plays a vital role. A marketing company would like to develop a strategy by analyzing 
their customer data. For this, data like age, location, time of activity, etc. has been 
collected to determine whether a user will click on an ad or not. 
Perform Logistic Regression on the given data to predict whether a user will click on an ad 
or not.

############################################################################################

# Importing required libraries into Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report, roc_auc_score

# Reading the data into python
data = pd.read_csv('D:/Hands on/25_Logistic Regression_1/Assignement/advertising.csv') 

# Information of the dataset
data.info()

# Statistical values of the dataset
data.describe()

# Columns of the data set
data.columns

# Returns top 5 records
data.head()

# Droping un related columns
data.drop(['Ad_Topic_Line', 'City', 'Country', 'Timestamp'], axis = 1, inplace = True)

# Spliting the data into X and Y
X = data.iloc[:, 0:5]

Y = data['Clicked_on_Ad']

# Checking for duplicates
X.duplicated().sum()

# Checking for null values
X.isnull().sum()

# Boxplot
X.plot(kind = 'box', subplots = True, sharey = False)

# creating a Winsorizer object
winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['Area_Income'])

# fit and transform the Area_Income feature
X['Area_Income'] = pd.DataFrame(winsor.fit_transform(X[['Area_Income']]))

# Boxplot
X.plot(kind = 'box', subplots = True, sharey = False)

# Sreating StandardScaler object
stndrd = StandardScaler()

# Scaling the data
X = pd.DataFrame(stndrd.fit_transform(X), columns = X.columns)

# Creating a model
logit_model = sa.Logit(Y, X).fit()

# Prediction on input
pred = logit_model.predict(X)

# ROC Curve
fpr, tpr, thshld = roc_curve(Y, pred)
opt_indx = np.argmax(tpr - fpr)
opt_val = thshld[opt_indx]
opt_val

# AUC 
auc = auc(fpr, tpr)

# Creating a new feature with zero values
X['pred'] =  np.zeros(1000)

# Converting into probability values
X.loc[pred > opt_val, 'pred'] = 1

confusion_matrix(Y, X.pred)

accuracy_score(Y, X.pred)

classification_report(Y, X.pred)

### PLOT FOR ROC
plt.plot(fpr, tpr, label = "AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 4)
plt.show()


x_train, x_test, y_train, y_test = train_test_split(X.iloc[:, 0:5], Y, random_state = 0, stratify = Y)

final_model = sa.Logit(y_train, x_train).fit()

test_pred = final_model.predict(x_test)
y_test = pd.DataFrame(y_test)
y_test['test_pred'] = np.zeros(250)
y_test.loc[test_pred > opt_val, 'test_pred'] = 1
confusion_matrix(y_test.test_pred, y_test.Clicked_on_Ad)
accuracy_score(y_test.test_pred, y_test.Clicked_on_Ad)
classification_report(y_test.test_pred, y_test.Clicked_on_Ad)

train_pred = final_model.predict(x_train)
y_train = pd.DataFrame(y_train)
y_train['train_pred'] = np.zeros(750)
y_train.loc[train_pred > opt_val, 'train_pred'] = 1
confusion_matrix(y_train.train_pred, y_train.Clicked_on_Ad)
accuracy_score(y_train.train_pred, y_train.Clicked_on_Ad)
classification_report(y_train.train_pred, y_train.Clicked_on_Ad)

############################################################################################

3.	Perform Logistic Regression on the dataset to predict whether a candidate will win or lose 
the election based on factors like amount of money spent and popularity rank. 

#############################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import statsmodels.api as sma
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, classification_report


data = pd.read_csv('D:/Hands on/25_Logistic Regression_1/Assignement/election_data.csv') 

data.info()

data.describe()

data.columns

data.drop(['Election-id'], axis = 1, inplace = True)

data.info()

data.duplicated().sum()

data.isnull().sum()



impute = SimpleImputer(missing_values = np.nan, strategy = 'mean')

data['Year'] = pd.DataFrame(impute.fit_transform(data[['Year']]))
data['Amount Spent'] = pd.DataFrame(impute.fit_transform(data[['Amount Spent']]))

mode_impute = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
data['Result'] = pd.DataFrame(mode_impute.fit_transform(data[['Result']]))
data['Popularity Rank'] = pd.DataFrame(mode_impute.fit_transform(data[['Popularity Rank']]))

data.isnull().sum()

data.rename({'Amount Spent' : 'AmountSpent', 
                   'Popularity Rank' : 'PopularityRank'}, axis = True, inplace = True)

X = data.iloc[:, 1:]

Y = pd.DataFrame(data['Result'])

X.plot(kind = 'box', subplots = True, sharey = False)


winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['AmountSpent'])

X['AmountSpent'] = pd.DataFrame(winsor.fit_transform(X[['AmountSpent']]))

X.plot(kind = 'box', subplots = True, sharey = False)



mmscale = MinMaxScaler()

X_scale = pd.DataFrame(mmscale.fit_transform(X), columns = X.columns)



logit_model = sma.Logit(Y, X_scale).fit()

logit_model.summary()
logit_model.summary2()

pred = logit_model.predict(X_scale)


# ROC Curve to identify the appropriate cutoff value
fpr, tpr, thresholds = roc_curve(Y, pred)
optimal_indx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_indx]
optimal_threshold

auc = auc(fpr, tpr)

X_scale['predict'] = np.zeros(11)
#X.drop(['prediction'], axis = 1, inplace = True)

X_scale.loc[pred > optimal_threshold, 'predict'] = 1 

confusion_matrix(X_scale.predict, Y)

accuracy_score(X_scale.predict, Y)

classification = classification_report(X_scale.predict, Y)

plt.plot(fpr, tpr, label = "AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 4)
plt.show()
###
final_data = pd.concat([X_scale.iloc[:, 0:3], Y], axis = 1)
final_data1 = pd.concat([X, Y], axis = 1)

from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(X_scale.iloc[:, 0:3], Y, random_state = 0, stratify = Y)

train, test = train_test_split(final_data1, random_state = 0, stratify = final_data['Result'])

import statsmodels.formula.api as sfa
final_model = sfa.logit("Result ~  AmountSpent + Year", data = train).fit()

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
### Checking accuracy

test_pred = final_model.predict(test)

test['pred'] = np.zeros(3)

test.loc[pred > optimal_threshold, 'pred'] = 1 

confusion_matrix(test.Result, test.pred)

accuracy_score(test.Result, test.pred)

classification_report(test.Result, test.pred)

train_pred = final_model.predict(train)
train['pred'] = np.zeros(8)
train.loc[pred > optimal_threshold, 'pred'] = 1 
confusion_matrix(train.Result, train.pred)
accuracy_score(train.Result, train.pred)
classification_report(train.Result, train.pred)

############################################################################################
4.	It is vital for banks that customers put in long term fixed deposits as they use it to 
pay interest to customers and it is not viable to ask every customer if they will put 
in a long-term deposit or not. So, build a Logistic Regression model to predict whether a 
customer will put in a long-term fixed deposit or not based on the different variables 
given in the data. The output variable in the dataset is Y which is binary. 
Snapshot of the dataset is given below.

#############################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_engine.outliers import Winsorizer

data = pd.read_csv("D:/Hands on/25_Logistic Regression_1/Assignement/bank_data.csv")

data.info()

data.describe()

data.duplicated().sum()

data.drop_duplicates(inplace = True)

data.duplicated().sum()

data.isnull().sum()

data.var()

data.drop(['default', 'housing', 'loan', 'poutfailure', 'poutother', 'poutsuccess',
'poutunknown', 'con_cellular', 'con_telephone', 'con_unknown',
'divorced', 'married', 'single', 'joadmin.', 'joblue.collar',
'joentrepreneur', 'johousemaid', 'jomanagement', 'joretired',
'joself.employed', 'joservices', 'jostudent', 'jotechnician',
'jounemployed', 'jounknown'], inplace = True, axis = 1)

data.info()

data.plot(kind = 'box', subplots = True, sharey = False)

age_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['age'])
bal_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['balance']) 
dur_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['duration']) 
cam_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['campaign']) 
pdy_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['pdays'])
pre_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['previous']) 
           

data['age'] = age_winsor.fit_transform(data[['age']]) 
data['balance'] = bal_winsor.fit_transform(data[['balance']]) 
data['duration'] = dur_winsor.fit_transform(data[['duration']]) 
data['campaign'] = cam_winsor.fit_transform(data[['campaign']]) 
data['pdays'] = pdy_winsor.fit_transform(data[['pdays']]) 
data['previous'] = pre_winsor.fit_transform(data[['previous']]) 

data.plot(kind = 'box', subplots = True, sharey = False)

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

X = data.iloc[:, 0:6]

Y = pd.DataFrame(data['y'])

min = MinMaxScaler()

X_scale = pd.DataFrame(min.fit_transform(X))

import statsmodels.api as sms
import statsmodels.formula.api as sfa

bank1 = pd.concat([X_scale, Y], axis = 1)

#logit_model = sms.Logit(Y, X).fit()

logit_model = sms.Logit(Y, X_scale).fit()

sfa.logit('y ~ age + balance + duration + campaign + pdays + previous', data = data).fit()

