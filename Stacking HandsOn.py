import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

df = datasets.load_diabetes()

X = pd.DataFrame(df.data, columns = df.feature_names) 

Y = pd.DataFrame(df.target, columns = ['target'])

data = pd.concat([X, Y], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 1)
estimators = [('dt', DecisionTreeClassifier()), ('knn', KNeighborsClassifier())]

stack = StackingClassifier(estimators = estimators, 
        final_estimator = RandomForestClassifier(n_estimators = 50, random_state = 0))

stack_model = stack.fit(x_train, y_train)

test_pred = stack_model.predict(x_test)

r2score = stack_model.score(y_test, x_test)
    


