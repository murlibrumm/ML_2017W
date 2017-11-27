import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neural_network import BernoulliRBM, MLPClassifier, MLPRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn import metrics
from sklearn import preprocessing
import sys
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import numpy as np
import warnings
from functools import reduce
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
from matplotlib import lines

class LastUpdatedOrderedDict(OrderedDict):
    "Store items in the order the keys were last added."

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        OrderedDict.__setitem__(self, key, value)


# TODO:
# 1) different ways to treat missing values
# 2) test with less columns (relevant ones)
# 3) different models (kNN, RF, Bayesian Networks, NN) => georg's models
# 4) different params
# maybe print the model & the missing value strategy at the end?
# maybe do the whole process X times => report on best/worst/avg results?

# GLOBALS
LOGGING          = True    # verbose logging output
SPLIT_DATA       = 0.2      # split dataset into training and testdata
DATAFRAME        = None     # our dataframe
MISSING_VALUES   = 'mean' # how to deal with missing values (delete, mean, median, most_frequent)
SIGNIFICANT_COLS = False    # significant columns only (like APM, PACs, #Hotkeys)
NUMBER_OF_RUNS   = 2
RESULT_FORMAT    = '({:0.2f}, {:0.2f}, {:0.2f})'
LABEL_ENC = preprocessing.LabelEncoder()
LABEL_ENC.fit(['n','y','unknown'])
EXPORT_PLOT = True

CLASS_ENC = preprocessing.LabelEncoder()
CLASS_ENC.fit(['democrat','republican'])

# change the classifier values here!
# ALGORITHMS = ['forest', 'knn', 'bayes', 'neural'] #algorithms to use
ALGORITHMS = ['neural'] #algorithms to use
algorithmParameter = (1, 10, 1)
rf_n_estimators = 10 # default: 10
knn_n_neighbors = 5 # default: 5
nb_priors = None # default: None
mlp_layers = (5,2) # default: (100,)
mlp_solver = 'lbfgs' # solver lbfgs is good for small datasets

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
    global countUnknown

    DATAFRAME = DATAFRAME.apply(lambda x: encodeLabels(x))
    print(countUnknown)

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
    resultsPerClassifier = LastUpdatedOrderedDict()
    for (model, name) in classifiers:
        resultsPerClassifier[name] = []


    # split into 80% training data, 20% test data
    train, test = train_test_split(DATAFRAME, test_size=SPLIT_DATA)

    # get training & test samples/targets
    training_samples, training_target = getSamplesAndTargets(train)
    test_samples,     actual_class  = getSamplesAndTargets(test)

    for (model, name) in classifiers:
        # for each classifier, do the training and evaluation
        #model.fit(training_samples, training_target)

        # predict the samples
        #predicted_class = model.predict(test_samples)

        crossScoresPrecisionSum = 0
        crossScoresRecallSum    = 0
        crossScoresF1Sum        = 0

        for i in range (0, NUMBER_OF_RUNS):
            # perform cross validation
            X, y = getSamplesAndTargets(DATAFRAME)
            crossScoresPrecisionSum += cross_val_score(model, X, y, cv=10, scoring='recall_weighted').mean() #std = accuracy
            crossScoresRecallSum    += cross_val_score(model, X, y, cv=10, scoring='precision_weighted').mean() #std = accuracy
            crossScoresF1Sum        += cross_val_score(model, X, y, cv=10, scoring='f1_weighted').mean() #std = accuracy
            cross_y_pred = cross_val_predict(model, X, y, cv=10)

        crossScoresPrecision = crossScoresPrecisionSum/NUMBER_OF_RUNS
        crossScoresRecall    = crossScoresRecallSum/NUMBER_OF_RUNS
        crossScoresF1        = crossScoresF1Sum/NUMBER_OF_RUNS

        resultsPerClassifier[name].append(
            (crossScoresPrecision, crossScoresRecall, crossScoresF1, 0))

        # summarize the fit of the model
        printResults(y, crossScoresPrecision, cross_y_pred, actual_class, None, name)
        # resultsPerClassifier[name].append(
        #    metrics.precision_recall_fscore_support(actual_class, predicted_class, average='weighted'))

    printClassifierReport(resultsPerClassifier)

    if EXPORT_PLOT:
        printPlot(resultsPerClassifier)

countUnknown = 0

def encodeLabels(column):
    global countUnknown

    if column.name == 'class':
        dic = defaultdict(int)
        for val in column:
            dic[val] += 1
        print(dic)

        return CLASS_ENC.transform(column)
    elif column.name == 'ID':
        return column
    else:
        for val in column:
            if val == 'unknown':
                countUnknown += 1
        return LABEL_ENC.transform(column)

def getClassifiers():
    # add the various classifiers
    classifiers = []
    for i in range(*algorithmParameter):
        if "forest" in ALGORITHMS:
            name = "Random Forests (n={0})".format(i)
            classifiers.append(
                (RandomForestClassifier(n_estimators=i, criterion='entropy', max_depth=100), name))
        if "knn" in ALGORITHMS:
            name = "kNN (n={0})".format(i)
            classifiers.append(
                (KNeighborsClassifier(n_neighbors=i), name))
        if "bayes" in ALGORITHMS:
            name = "Naive Bayes (priors={0})".format(i)
            classifiers.append(
                (GaussianNB(priors=i), name))
        if "neural" in ALGORITHMS:
            name = "Neural Network (layers={0},{1})".format(i*2,i)
            classifiers.append(
                (MLPClassifier(solver=mlp_solver, hidden_layer_sizes=(i*2,i)), name))
    return classifiers

def getSamplesAndTargets(data):
    # get training samples (without class and ID)
    samples = data.drop(['ID', 'class'], axis=1)

    # get training target (class)
    targets  = data['class'].values
    return samples, targets


def printResults(cross_actual_class, cross_score, cross_pred_class, actual_class, predicted_class, classifier):
    print("\n", "=" * 80, "\n")
    print("=== Classifier:", classifier, "===\n")
    print("=== Cross Validation Results: ===\n"
          "Accuracy: %0.2f (+/- %0.2f)" % (cross_score.mean(), cross_score.std() * 2), "\n"
          "Confusion matrix:\n",
          metrics.confusion_matrix(cross_actual_class, cross_pred_class))
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

def printPlot(resultsPerClassifier):
    precision = [results[0][0] for name, results in resultsPerClassifier.items()]
    recall    = [results[0][1] for name, results in resultsPerClassifier.items()]
    f1Score   = [results[0][2] for name, results in resultsPerClassifier.items()]
    xAxis = list(range(*algorithmParameter))

    printlog(list(range(*algorithmParameter)))
    printlog(precision)
    printlog(recall)
    printlog(f1Score)

    fig = plt.figure(figsize=(8, 8))

    # p_line = lines.Line2D([], [], label="Precision", linestyle="--", color="red")
    # r_line = lines.Line2D([], [], label="Recall", linestyle="-.", color="green")
    # f_line = lines.Line2D([], [], label="F1", linestyle=None, color="blue")

    precisionLine, = plt.plot(xAxis, precision, "r--", label='Precision')
    recallLine,    = plt.plot(xAxis, recall,    "g-.", label='Recall')
    f1ScoreLine,   = plt.plot(xAxis, f1Score,   "b",    label='F1')
    plt.legend(handles=[precisionLine, recallLine, f1ScoreLine])
    plt.ylabel('F1 Score')

    # random forrest
    # plt.xlabel('number of trees in the forest')

    # knn
    # plt.xlabel('neighbors used')

    # neural
    plt.xlabel('number of perceptrons of the second layer and half the number of perceptrons of the first layer')

    # plt.show()
    plt.savefig('figures/voting_neural_1-10 2nd nodes_missing value treatment.png')
    plt.close(fig)

def printlog(message):
    if (LOGGING):
        print(message)


if __name__ == '__main__':
    exit = main()
    sys.exit(exit)
