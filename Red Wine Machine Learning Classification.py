from random import random

import pandas as pd
import numpy as np
import matplotlib
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tensorboard.notebook import display
from tensorflow.tools.docs.doc_controls import header

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = pd.read_csv('/Users/Suli/Documents/Python Practice/Datasets/wine+quality/winequality-red.csv', delimiter=';')

bins = [0, 5.5, 7.5, 10]
labels = [0, 1, 2]
df['quality'] = pd.cut(df['quality'], bins=bins, labels=labels)

x = df[df.columns[:-1]]
y = df['quality']
sc = StandardScaler()
x = sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(df)

# K Nearest Neighbour Classifier

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)
y_prediction = knn_model.predict(x_test)
print(classification_report(y_test, y_prediction))

# Random Forrest Classifier

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y2_prediction = rf.predict(x_test)
print(classification_report(y_test, y2_prediction))

# Decision Tree Classifier

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y3_prediction = dt.predict(x_test)
print(classification_report)

# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y4_prediction = sgd.predict(x_test)
print(classification_report(y_test, y4_prediction))