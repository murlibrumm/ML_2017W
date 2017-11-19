from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import graphviz
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn import preprocessing

# ---------------------------------------------------------
# global variables
# ---------------------------------------------------------
labelEncoderDic = defaultdict(preprocessing.LabelEncoder)
df = None
testdata_id = None

# ---------------------------------------------------------
# functions
# ---------------------------------------------------------
def convertCSV(filename,trainingsFile):
    global labelEncoderDic
    global df
    global testdata_id

    # comma delimited is the default
    df = pd.read_csv(filename, header=0)

    # put the original column names in a python list
    original_headers = list(df.columns.values)

    testdata_id = df.get('ID').as_matrix()

    df = df.drop(columns=['ID'])

    if trainingsFile:
        # Encoding the variable
        fit = df.apply(lambda x: labelEncoderDic[x.name].fit_transform(x))
    else:
        # Using the dictionary to label future data
        fit = df.apply(lambda x: labelEncoderDic[x.name].transform(x))

    numpy_array = fit.as_matrix()

    if trainingsFile:
        X, y = numpy_array[:, 1:], numpy_array[:, 0]
    else:
        X, y = numpy_array, None

    print(type(X))
    print(X.shape)

    return (X, y)


# ---------------------------------------------------------
# executed part
# ---------------------------------------------------------

# read in test data
X_train, y_train = convertCSV("CongressionalVotingID.shuf.train.csv", True)

# define trainings algorithm
clf = RandomForestClassifier(n_estimators=10)

# train the model
print(clf.fit(X_train, y_train))

# obtain the test data
X_test, y_test = convertCSV("CongressionalVotingID.shuf.test.csv", False)

# prediction of the test data
y_pred = clf.predict(X_test)
print(y_test)

y_pred_trans = labelEncoderDic["class"].inverse_transform(y_pred)
prediction = np.c_[testdata_id, y_pred_trans]
np.savetxt("prediction.csv", prediction, delimiter=",", fmt='%s', header='ID,"class"')

# print(confusion_matrix(y_test, y_pred))

''' store a model
joblib.dump(clf, "save.pkl")

clf2 = joblib.load("save.pkl")
y_true = y
y_pred = clf2.predict(X[:])
'''

'''
shows the decision tree
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

dot_data = tree.export_graphviz(clf, out_file='an',
                         feature_names=numeric_headers,
                         class_names=np.array(['democrat','republican']),
                         filled=True, rounded=True,
                         special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("iris_col", view=True)
'''
