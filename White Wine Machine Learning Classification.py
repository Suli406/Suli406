import pandas as pd
import numpy as np
import matplotlib as mpt
import sklearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = pd.read_csv('/Users/Suli/Documents/Python Practice/Datasets/wine+quality/winequality-white.csv', delimiter=';')
bins = [0, 3, 6, 10]
labels = [0, 1, 2]
df['quality'] = pd.cut(df['quality'], bins=bins, labels=labels)

x = df[df.columns[:-1]]
y = df['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=80)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
ros = RandomOverSampler()
x_train, y_train = ros.fit_resample(x_train, y_train)

# KNN Model

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)
y_prediction = knn_model.predict(x_test)
print(classification_report(y_test, y_prediction))

# Random Forest

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
y2_prediction = rf_model.predict(x_test)
print(classification_report(y_test, y2_prediction))

# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y3_prediction = sgd.predict(x_test)
print(classification_report(y_test, y3_prediction))

# Decision Tree

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y4_prediction = dt.predict(x_test)
print(classification_report(y_test, y4_prediction))