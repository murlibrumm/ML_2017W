import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neural_network import BernoulliRBM, MLPClassifier, MLPRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn import metrics
from sklearn import preprocessing
import sys
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import warnings
from functools import reduce


# TODO:
# 1) different ways to treat missing values
# 2) test with less columns (relevant ones)
# 3) different models (kNN, RF, Bayesian Networks, NN) => georg's models
# 4) different params
# maybe print the model & the missing value strategy at the end?
# maybe do the whole process X times => report on best/worst/avg results?

# GLOBALS
LOGGING          = False    # verbose logging output
SPLIT_DATA       = 0.2      # split dataset into training and testdata
DATAFRAME        = None     # our dataframe
MISSING_VALUES   = 'median' # how to deal with missing values (delete, mean, median, most_frequent)
SIGNIFICANT_COLS = False    # significant columns only (like APM, PACs, #Hotkeys)
NUMBER_OF_RUNS   = 5
RESULT_FORMAT    = '({:0.2f}, {:0.2f}, {:0.2f})'
LABEL_ENC = preprocessing.LabelEncoder()
LABEL_ENC.fit(['n','y','unknown'])

CLASS_ENC = preprocessing.LabelEncoder()
CLASS_ENC.fit(['democrat','republican'])

# change the classifier values here!
ALGORITHMS = ['forest', 'knn', 'bayes', 'neural'] #algorithms to use
rf_n_estimators = 10 # default: 10
knn_n_neighbors = 5 # default: 5
nb_priors = None # default: None
mlp_layers = (100) # default: (100,)

# filter warnings of the type UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def main():
    readDataset()
    encodeDataset()
    handleMissingValues()
    classifiers = getClassifiers()
    trainAndPredict(classifiers)


def readDataset():
    global DATAFRAME
    # csv => DataFrame
    DATAFRAME = pd.read_csv('datasets/CongressionalVotingID.shuf.train.csv')

    printlog('dataset size:' + str(DATAFRAME.shape))
    print(DATAFRAME)

def encodeDataset():
    global DATAFRAME
    DATAFRAME = DATAFRAME.apply(lambda x: encodeLabels(x))

def handleMissingValues():
    global DATAFRAME

    unknownVal = LABEL_ENC.transform(['unknown']).item(0)
    featureVal = DATAFRAME.drop(columns=['class'])

    if (MISSING_VALUES == 'delete'):
        # filter out missing values
        # https://stackoverflow.com/questions/27428954/drop-row-if-any-column-value-does-not-a-obey-a-condition-in-pandas
        featureVal = featureVal[~(featureVal == unknownVal).any(1)]

    if (MISSING_VALUES == 'median' or MISSING_VALUES == 'mean' or MISSING_VALUES == 'most_frequent'):
        # deal with missing values => mean
        imp = preprocessing.Imputer(missing_values=unknownVal, strategy=MISSING_VALUES, axis=0)
        imputed_DF = pd.DataFrame(imp.fit_transform(featureVal))
        imputed_DF.columns = featureVal.columns
        imputed_DF.index = featureVal.index
        featureVal = imputed_DF
        printlog('dataset size after handling missing values:' + str(featureVal.shape))

    DATAFRAME = DATAFRAME.loc[:, 'class':'class'].join(featureVal)

    print(DATAFRAME)

def trainAndPredict(classifiers):
    resultsPerClassifier = {}
    for (model, name) in classifiers:
        resultsPerClassifier[name] = []
    for i in range (0, NUMBER_OF_RUNS):
        # split into 80% training data, 20% test data
        train, test = train_test_split(DATAFRAME, test_size=SPLIT_DATA)

        # get training & test samples/targets
        training_samples, training_target = getSamplesAndTargets(train)
        test_samples,     actual_class  = getSamplesAndTargets(test)

        for (model, name) in classifiers:
            # for each classifier, do the training and evaluation
            model.fit(training_samples, training_target)

            # predict the samples
            predicted_class = model.predict(test_samples)

            # perform cross validation
            X, y = getSamplesAndTargets(DATAFRAME)
            cross_scores = cross_val_score(model, X, y, cv=10)

            # summarize the fit of the model
            printResults(cross_scores, actual_class, predicted_class, name)
            resultsPerClassifier[name].append(
                metrics.precision_recall_fscore_support(actual_class, predicted_class, average='weighted'))

    printClassifierReport(resultsPerClassifier)

def encodeLabels(column):
    if column.name == 'class':
        return CLASS_ENC.transform(column)
    elif column.name == 'ID':
        return column
    else:
        return LABEL_ENC.transform(column)

def getClassifiers():
    # add the various classifiers
    classifiers = []
    if "forest" in ALGORITHMS:
        name = "Random Forests (n={0})".format(rf_n_estimators)
        classifiers.append(
            (RandomForestClassifier(n_estimators=rf_n_estimators, criterion='entropy', max_depth=100), name))
    if "knn" in ALGORITHMS:
        name = "kNN (n={0})".format(knn_n_neighbors)
        classifiers.append(
            (KNeighborsClassifier(n_neighbors=knn_n_neighbors), name))
    if "bayes" in ALGORITHMS:
        name = "Naive Bayes (priors={0})".format(nb_priors)
        classifiers.append(
            (GaussianNB(priors=nb_priors), name))
    if "neural" in ALGORITHMS:
        name = "Neural Network (layers={0})".format(mlp_layers)
        classifiers.append(
            (MLPClassifier(hidden_layer_sizes=mlp_layers), name))
    return classifiers

def getSamplesAndTargets(data):
    # get training samples (without class and ID)
    samples = data.drop(['ID', 'class'], axis=1)

    # get training target (class)
    targets  = data['class'].values
    return samples, targets


def printResults(cross_score, actual_class, predicted_class, classifier):
    print("\n", "=" * 80, "\n")
    print("=== Classifier:", classifier, "===\n")
    print("=== Classification Report: ===\n"
          "precision (How many selected elements are relevant?): TP / (TP + FP)\n"
          "recall (How many relevant elements are selected?): TP / (TP + FN)\n"
          "f1 score to measure a test's accuracy (considers both precision and recall): 2*((PR * RC)/(PR + RC))\n"
          "support: #elements in this class\n", metrics.classification_report(actual_class, predicted_class))
    print("=== Cross Validation Results: ===\n"
          "Accuracy: %0.2f (+/- %0.2f)" % (cross_score.mean(), cross_score.std() * 2))
    print("=== Confusion Matrix: ===\n"
          "top: predicted values, left: actual values\n",
          metrics.confusion_matrix(actual_class, predicted_class))
    print()
    # here we can use 'weighted' or 'macro' => weighted adjusts for the number of instances per label
    # print("f1-score:        %0.2f" % metrics.f1_score(actual_class, predicted_class, average='weighted'))
    # print("recall-score:    %0.2f" % metrics.recall_score(actual_class, predicted_class, average='weighted'))
    # print("precision-score: %0.2f" % metrics.precision_score(actual_class, predicted_class, average='weighted'))
    # print("accuracy-score:  %0.2f" % metrics.accuracy_score(actual_class, predicted_class))


def printClassifierReport(resultsPerClassifier):
    print()
    print("=" * 80)
    print("=== Report per Classifier: ===")
    printlog(resultsPerClassifier)
    for name, results in resultsPerClassifier.items():
        print("=== %s ===" % name)
        # determine the best / worst result based on f1-score
        bestRow  = reduce((lambda x, y: x if x[2] > y[2] else y), results)[:-1]
        print("best (P, R, F):    ", RESULT_FORMAT.format(*bestRow))
        worstRow = reduce((lambda x, y: x if x[2] < y[2] else y), results)[:-1]
        print("worst (P, R, F):   ", RESULT_FORMAT.format(*worstRow))
        # calculate the average result.
        summedRows = results[0][:-1] if len(results) == 1 else reduce((lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2])), results)
        averageRow = list(map((lambda x: x / NUMBER_OF_RUNS), summedRows))
        print("average (P, R, F): ", RESULT_FORMAT.format(*averageRow))


def printlog(message):
    if (LOGGING):
        print(message)


if __name__ == '__main__':
    exit = main()
    sys.exit(exit)
