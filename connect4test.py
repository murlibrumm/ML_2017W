#!/bin/env python
"""Script for playing around with the connect-4 data set"""

from sys import argv
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


def add_result(targets, predictions, results, kind, param):
    """Add the result to the given list of results"""
    (precision, recall, fscore, support) = metrics.precision_recall_fscore_support(targets, predictions)

    def avg(vals, weights):
        """Calculate the weighted average of vals"""
        res = 0
        sum_weights = 0
        min_len = min(len(vals), len(weights))
        for i in range(0, min_len):
            res += vals[i] * weights[i]
            sum_weights += weights[i]

        return res / sum_weights

    # important!!
    # the average has to be weighed, because there are not equally many samples in each class
    precision = avg(precision, support)
    recall = avg(recall, support)
    fscore = avg(fscore, support)

    result_tuple = (param, precision, recall, fscore)

    new_results = results[kind]
    new_results.append(result_tuple)
    results[kind] = new_results


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


def append(list_, item):
    """Append item to the list if it isn't contained in the list yet"""
    if item not in list_:
        list_.append(item)


algorithms = []

if len(argv) == 2 and (argv[1] == "-h" or argv[1] == "--help"):
    # print help
    print("Usage: {0} [ALGORITHMS]".format(argv[0]))
    print()
    print("\tALGORITHMS")
    print("\t\tSpecifies which classification algorithms to run:")
    print("\t\t\t* random forest")
    print("\t\t\t* knn")
    print("\t\t\t* naive bayes")
    print("\t\t\t* neural network")
    print("\t\t\t* all")
    print("\t\tIf no algorithm is specified, all will be run")
    exit(0)

elif len(argv) >= 2:
    for alg in argv[1:]:
        alg = alg.lower()

        # check the arguments if we can recognise some algorithm names
        if alg in "random forest" and (alg[0] == "r" or alg[0] == "f"):
            append(algorithms, "forest")
        elif alg in "knn" and alg[0] == "k":
            append(algorithms, "knn")
        elif alg in "naive bayes" and (alg[0] == "n" or alg[0] == "b"):
            append(algorithms, "bayes")
        elif alg in "neural networks" and alg[0] == "n":
            append(algorithms, "neural")
        elif alg in "mlp" and alg[0] == "m":
            append(algorithms, "neural")
        elif alg in "all" and alg[0] == "a":
            algorithms = ["forest", "knn", "bayes", "neural"]

elif len(argv) == 1:
    # if the program was started without arguments, just run all algorithms
    algorithms = ["forest", "knn", "bayes", "neural"]

# the column names for the Connect 4 Dataset, since they're not included in the data file
# Class: ["win", "loss", "draw"]
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
print("Dataset Size: ", data.shape)
print("Predicted Classes: win / loss / draw")
print()

# split into 80% training data, 20% test data
(train, test) = train_test_split(data, test_size=0.2)

# split training and test data sets into (samples, targets) each
# samples are the columns to be looked at by the classifier, targets are the classifications
(training_samples, training_target) = split_samples(train, ["Class"])
(test_samples, test_target) = split_samples(test, ["Class"])

iterations = range(0, 10)
results = {"forest": [],
           "knn":    [],
           "bayes":  [],
           "neural": []}

set_rf_n_estimators = [10, 40, 60, 80, 100, 150, 200, 250, 300, 400]
set_knn_n_neighbors = [5, 10, 25, 50, 100, 200, 300, 500, 1000, 2000]
set_mlp_layers = [(100,) * (i + 1) for i in iterations]


for i in iterations:

    rf_n_estimators = set_rf_n_estimators[i]
    knn_n_neighbors = set_knn_n_neighbors[i]
    nb_priors = None
    mlp_layers = set_mlp_layers[i]

    # add the various classifiers
    classifiers = []
    if "forest" in algorithms:
        kind = "forest"
        param = rf_n_estimators
        name = "Random Forests (n={0})".format(rf_n_estimators)
        classifiers.append((RandomForestClassifier(n_estimators=rf_n_estimators), name, kind, param))
    if "knn" in algorithms:
        kind = "knn"
        param = knn_n_neighbors
        name = "kNN (n={0})".format(knn_n_neighbors)
        classifiers.append((KNeighborsClassifier(n_neighbors=knn_n_neighbors), name, kind, param))
    if "bayes" in algorithms:
        kind = "bayes"
        param = 0
        name = "Naive Bayes (priors={0})".format(nb_priors)
        classifiers.append((GaussianNB(priors=nb_priors), name, kind, param))

        # why should nb run more than once, if we only have None as interesting parameter?
        algorithms.remove("bayes")
    if "neural" in algorithms:
        kind = "neural"
        param = len(mlp_layers)
        name = "Neural Network (layers={0})".format(mlp_layers)
        classifiers.append((MLPClassifier(hidden_layer_sizes=mlp_layers), name, kind, param))

    for (model, name, kind, param) in classifiers:
        # for each classifier, do the training and evaluation
        model.fit(training_samples, training_target)

        # predict the samples
        predictions = model.predict(test_samples)

        # summarize the fit of the model
        show_summary(test_target, predictions, name)

        # add the result to the results list
        add_result(test_target, predictions, results, kind, param)

with open("c4-results.txt", "w") as result_file:
    for kind in results:
        result_file.write("{0}\n".format(kind))

        for (param, precision, recall, f1score) in results[kind]:
            result_file.write("{0} {1} {2} {3}\n".format(param, precision, recall, f1score))

        result_file.write("end {0}".format(kind))
        result_file.write("\n")