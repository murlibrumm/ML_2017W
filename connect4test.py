#!/bin/env python
"""Script for playing around with the connect-4 data set"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split


def split_samples(dataset, target_columns):
    """Split the dataset into sample and target datasets.

    :param dataset: the dataset to split
    :param target_columns: the columns to use as target
    """
    # drop the target columns from the sample data
    samples = dataset.drop(target_columns, axis=1)
    # as target, use only the target columns
    # .ravel() against the DataConversionWarning
    target = dataset[target_columns].values.ravel()
    return (samples, target)


def show_summary(targets, predictions, classifier):
    """Print the result of calculated predictions vs. actual targets for the classifier algorithm"""
    print("=" * 80)
    print()
    print("Statistics for applied classifier ({0}):".format(classifier))
    print()
    print("Classification Report:")
    print(metrics.classification_report(targets, predictions))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(targets, predictions))


# the column names for the Connect 4 Dataset, since they're not included in the data file
column_names = ["a1", "a2", "a3", "a4", "a5", "a6", "b1", "b2", "b3", "b4", "b5", "b6",
                "c1", "c2", "c3", "c4", "c5", "c6", "d1", "d2", "d3", "d4", "d5", "d6",
                "e1", "e2", "e3", "e4", "e5", "e6", "f1", "f2", "f3", "f4", "f5", "f6",
                "g1", "g2", "g3", "g4", "g5", "g6", "Class"]

# read the connect-4 data set => result: DataFrame
data = pd.read_csv('datasets/connect-4.data', header=None, names=column_names)

# since we have string values ('x', 'o', 'b') and not only numeric values:
# https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
data = data.apply(LabelEncoder().fit_transform)
print("Dataset: Connect 4")
print('Dataset Size:', data.shape)
print()

# split into 80% training data, 20% test data
(train, test) = train_test_split(data, test_size=0.2)

# split training and test data sets into (samples, targets) each
# samples are the columns to be looked at by the classifier, targets are the classifications
(training_samples, training_target) = split_samples(train, ["Class"])
(test_samples, test_target) = split_samples(test, ["Class"])

# add the various classifiers
classifiers = []
classifiers.append((RandomForestClassifier(n_estimators=10), "Random Forests (n=10)"))
classifiers.append((KNeighborsClassifier(n_neighbors=5), "kNN (n=5)"))
classifiers.append((GaussianNB(priors=None), "Naive Bayes (priors=None)"))
classifiers.append((MLPClassifier(hidden_layer_sizes=(100,)), "Neural Network (hidden layers=100)"))

for (model, name) in classifiers:
    # for each classifier, do the training and evaluation
    model.fit(training_samples, training_target)

    # predict the samples
    predictions = model.predict(test_samples)

    # summarize the fit of the model
    show_summary(test_target, predictions, name)
