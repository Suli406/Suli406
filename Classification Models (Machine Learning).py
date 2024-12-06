import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.python.feature_column.feature_column import linear_model

# Setting terminal display options.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Naming the columns and setting up a dataframe using pandas.

cols = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3trans', 'fAlpha', 'fDist', 'class']
df = pd.read_csv('/Users/Suli/Documents/Python Practice/Datasets/magic+gamma+telescope/magic04.data', names=cols)

# Binary encoding the class column to make it true or false. 1 is true, 0 is false
df['class'] = (df['class'] == 'g').astype(int)

# Making histograms for each column in the dataframe, where one is gamma and one is hadron for each column.
# for label in cols[:-1]:
#     plt.hist(df[df['class']==1][label], color = 'blue', label = 'gamma', alpha = 0.7, density=True)
#     plt.hist(df[df['class'] ==0][label], color = 'red', label = 'hadron', alpha = 0.7, density=True)
#     plt.title(label)
#     plt.ylabel('Probability')
#     plt.xlabel(label)
#     plt.legend()
#     plt.show()

# Train, validation and test datasets.

train, test, valid = np.split(df.sample(frac=1), [(int(0.6*len(df))), int(0.8*len(df))])
def scale_dataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (-1, 1))))

    return data, x, y

train, x_train, y_train = scale_dataset(train, oversample=True)
valid, x_valid, y_valid = scale_dataset(valid, oversample=False)
test, x_test, y_test = scale_dataset(test, oversample=False)

# kNN (K Nearst Neighbours)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model = knn_model.fit(x_train, y_train)
y_prediction = knn_model.predict(x_test)
print(classification_report(y_prediction, y_test))

# Naive Bayes

from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model = nb_model.fit(x_train, y_train)
y2_prediction = nb_model.predict(x_test)
print(classification_report(y2_prediction, y_test))

# Logistic Regression

from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression()
lg_model = lg_model.fit(x_train, y_train)
y3_prediction = lg_model.predict(x_test)
print(classification_report(y3_prediction, y_test))

# Support Vector Machines

from sklearn.svm import SVC

svm_model = SVC()
svm_model = svm_model.fit(x_train, y_train)
y4_prediction = svm_model.predict(x_test)
print(classification_report(y4_prediction, y_test))


