import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

df = datasets.load_breast_cancer()

X = pd.DataFrame(df.data, columns = df.feature_names)

Y = pd.DataFrame(df.target, columns = ['target'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 0)

dtree = DecisionTreeClassifier(random_state = 0)

bagging = BaggingClassifier(base_estimator = dtree, n_estimators = 100, random_state = 0, n_jobs = -1, bootstrap = True)

bad_model = bagging.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, bad_model.predict(x_test))
accuracy_score(y_test, bad_model.predict(x_test))

confusion_matrix(y_train, bad_model.predict(x_train))
accuracy_score(y_train, bad_model.predict(x_train))
