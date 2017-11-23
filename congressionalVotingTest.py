from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import tree
import graphviz
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict


import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn import preprocessing

# ---------------------------------------------------------
# global variables
# ---------------------------------------------------------
labelEnc = preprocessing.LabelEncoder()
labelEnc.fit(['n','y','unknown'])

classEnc = preprocessing.LabelEncoder()
classEnc.fit(['democrat','republican'])

df = None
testdata_id = None

# ---------------------------------------------------------
# functions
# ---------------------------------------------------------
def convertCSV(filename,trainingsFile):
    global labelEnc
    global df
    global testdata_id

    # comma delimited is the default
    df = pd.read_csv(filename, header=0)

    # put the original column names in a python list
    original_headers = list(df.columns.values)

    if not trainingsFile:
        testdata_id = df.get('ID').as_matrix()

    df = df.drop(columns=['ID'])

    if trainingsFile:
        # Encoding the variable
        fit = df.apply(lambda x: classEnc.transform(x) if x.name == 'class'
                                        else labelEnc.transform(x))
    else:
        # Using the dictionary to label future data
        fit = df.apply(lambda x: classEnc.transform(x) if x.name == 'class'
                                        else labelEnc.transform(x))

    # dealing with missing values
    unknownValue = labelEnc.transform(['unknown']).item(0)
    imp = preprocessing.Imputer(missing_values=unknownValue,
                                strategy='mean', axis=0)
    imp.fit(fit)

    numpy_array = fit.as_matrix()

    if trainingsFile:
        X, y = numpy_array[:, 1:], numpy_array[:, 0]
    else:
        X, y = numpy_array, None

    return (X, y)


# ---------------------------------------------------------
# executed part
# ---------------------------------------------------------

# read in train data
X, y = convertCSV("./datasets/CongressionalVotingID.shuf.train.csv", True)

# split data into test/training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0)

# define trainings algorithm
# clf = RandomForestClassifier(n_estimators=10)
# clf = KNeighborsClassifier(n_neighbors=10)
clf = GaussianNB()
# clf = LinearSVC(random_state=0)
# clf = tree.DecisionTreeClassifier()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                 hidden_layer_sizes=(5, 2), random_state=1)

scores = cross_val_score(clf, X, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

y_pred = cross_val_predict(clf, X, y, cv=5)
conf_mat = metrics.confusion_matrix(y,y_pred)
print(conf_mat)

# train the model
print(clf.fit(X_train, y_train))

# read in the provided test data
X_provided_test, y_provided_test = convertCSV("./datasets/CongressionalVotingID.shuf.test.csv", False)


# prediction of the test data
# y_pred = clf.predict(X_test)

# prediction of the provided test data
y_provided_test = clf.predict(X_provided_test)
y_provided_pred_trans = classEnc.inverse_transform(y_provided_test)
prediction = np.c_[testdata_id, y_provided_pred_trans]
np.savetxt("prediction.csv", prediction, delimiter=",", fmt='%s', header='ID,"class"', comments='')

'''
# evaluate the model
# summarize the fit of the model
print("evaluation:")
print(metrics.classification_report(y_test, y_pred))
print("confusion matirx:\n" + str(metrics.confusion_matrix(y_test, y_pred)))
'''

''' store a model
joblib.dump(clf, "save.pkl")

clf2 = joblib.load("save.pkl")
y_true = y
y_pred = clf2.predict(X[:])
'''

'''
print the structure of a decision tree
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
