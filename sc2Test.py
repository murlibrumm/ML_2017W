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
import matplotlib.pyplot as plt
from collections import OrderedDict


class LastUpdatedOrderedDict(OrderedDict):
    "Store items in the order the keys were last added."

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        OrderedDict.__setitem__(self, key, value)


# GLOBALS
LOGGING          = True    # verbose logging output
SPLIT_DATA       = 0.2      # split dataset into training and testdata
DATAFRAME        = None     # our dataframe
MISSING_VALUES   = 'delete' # how to deal with missing values (delete, mean, median, most_frequent)
SIGNIFICANT_COLS = False    # significant columns only (like APM, PACs, #Hotkeys)
NUMBER_OF_RUNS   = 1
EXPORT_PLOT      = False
X_LABEL          = 'hidden layer sizes'
PLOT_FILE_NAME   = 'figures/neural_network_2.png'

# change the classifier values here!
ALGORITHMS = ['neural'] #algorithms to use ['forest', 'knn', 'bayes', 'neural']
algorithmParameter = (5, 100+1, 5) # set a parameter in range(start, end, jump)

# forest params (algorithmParameter controls n_estimators)
forestCriterion = 'gini' # "gini" (default) for the Gini impurity 2) "entropy" for the information gain.
forestMaxDepth  = None      # how deep can a tree be max; default: none

# knn params (algorithmParameter control n_neighbors)
knnWeights   = 'uniform'   # weights: 1) 'uniform' (default): weighted equally. 2) 'distance': closer neighbors => more influence
knnAlgorithm = 'brute'      # algorithm to compute the NN: {'ball_tree', 'kd_tree', 'brute', 'auto}

# bayes params TODO

# neural MLP params (algorithmParameter controls hidden_layer_sizes, default: (100,))
neuralActivation = 'relu' # (activation function for the hidden layer) : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
neuralSolver = 'adam' # (for the weight optimization): {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
neuralLearningRate = 'constant'# (Learning rate schedule for weight updates).: {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
neuralMaxIter = 200 # max_iter : int, optional, default 200

# filter warnings of the type UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def main():
    readDataset()
    handleMissingValues()
    classifiers = getClassifiers()
    trainAndPredict(classifiers)


def readDataset():
    global DATAFRAME
    # csv => DataFrame
    DATAFRAME = pd.read_csv('datasets/SkillCraft1_Dataset.csv')
    printlog('dataset size:' + str(DATAFRAME.shape))


def handleMissingValues():
    global DATAFRAME
    if (MISSING_VALUES == 'delete'):
        # filter out missing values
        # https://stackoverflow.com/questions/27428954/drop-row-if-any-column-value-does-not-a-obey-a-condition-in-pandas
        DATAFRAME = DATAFRAME[~(DATAFRAME == '?').any(1)]

    if (MISSING_VALUES == 'median' or MISSING_VALUES == 'mean' or MISSING_VALUES == 'most_frequent'):
        # deal with missing values => mean
        DATAFRAME.replace({'?': np.nan}, inplace=True)
        fill_NaN = preprocessing.Imputer(missing_values='NaN', strategy=MISSING_VALUES, axis=0)
        imputed_DF = pd.DataFrame(fill_NaN.fit_transform(DATAFRAME))
        imputed_DF.columns = DATAFRAME.columns
        imputed_DF.index = DATAFRAME.index
        DATAFRAME = imputed_DF
        printlog('dataset size after handling missing values:' + str(DATAFRAME.shape))


def trainAndPredict(classifiers):
    resultsPerClassifier = LastUpdatedOrderedDict()
    for (model, name) in classifiers:
        resultsPerClassifier[name] = []
    for i in range (0, NUMBER_OF_RUNS):
        # split into 80% training data, 20% test data
        train, test = train_test_split(DATAFRAME, test_size=SPLIT_DATA)

        # get training & test samples/targets
        training_samples, training_target = getSamplesAndTargets(train)
        test_samples,     actual_leagues  = getSamplesAndTargets(test)

        for (model, name) in classifiers:
            # for each classifier, do the training and evaluation
            model.fit(training_samples, training_target)

            # predict the samples
            predicted_leagues = model.predict(test_samples)

            # perform cross validation
            X, y = getSamplesAndTargets(DATAFRAME)
            crossScoresPrecision = cross_val_score(model, X, y, cv=10, scoring='recall_weighted')
            crossScoresRecall    = cross_val_score(model, X, y, cv=10, scoring='precision_weighted')
            crossScoresF1        = cross_val_score(model, X, y, cv=10, scoring='f1_weighted')
            crossScoresAccuracy  = cross_val_score(model, X, y, cv=10, scoring='accuracy')

            # summarize the fit of the model
            crossScoresMean = (crossScoresPrecision.mean(), crossScoresRecall.mean(), crossScoresF1.mean(), crossScoresAccuracy.mean())
            crossScoresStd  = (crossScoresPrecision.std() * 2, crossScoresRecall.std() * 2, crossScoresF1.std() * 2, crossScoresAccuracy.std() * 2)
            printResults(crossScoresMean, crossScoresStd, actual_leagues, predicted_leagues, name)
            resultsPerClassifier[name].append(
                crossScoresMean)
                #(metrics.precision_recall_fscore_support(actual_leagues, predicted_leagues, average='weighted'))

    printClassifierReport(resultsPerClassifier)
    if EXPORT_PLOT:
        printPlot(resultsPerClassifier)


def getClassifiers():
    # add the various classifiers
    classifiers = []
    for i in range(*algorithmParameter):
        if "forest" in ALGORITHMS:
            name = "Random Forests (n={0})".format(i)
            classifiers.append(
                # n_estimators: number of trees in the forest, default: 10
                # criterion: 1) "gini" (default) for the Gini impurity 2) "entropy" for the information gain.
                # max_depth: how deep can a tree be max; default: none
                (RandomForestClassifier(n_estimators=i, criterion=forestCriterion, max_depth=forestMaxDepth), name))
        if "knn" in ALGORITHMS:
            name = "kNN (n={0})".format(i)
            classifiers.append(
                # n_neighbors: number of neighbours to use, default: 5
                # weights: 1) 'uniform' (default): weighted equally. 2) 'distance': closer neighbors => more influence
                # algorithm to compute the NN: 1) 'ball_tree' will use BallTree 2) 'kd_tree' will use KDTree
                (KNeighborsClassifier(n_neighbors=i, weights=knnWeights, algorithm=knnAlgorithm), name))
        if "bayes" in ALGORITHMS:
            name = "Naive Bayes (priors={0})".format('None')
            classifiers.append(
                # priors: prior probabilities of the classes; default: 'none'
                (GaussianNB(priors=None), name))
        if "neural" in ALGORITHMS:
            name = "Neural Network (layers={0})".format(i)
            classifiers.append(
                # hidden_layer_sizes (ith element = number of neurons in the ith hidden layer), default: (100,)
                # activation (activation function for the hidden layer) : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
                # solver (for the weight optimization): {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
                # learning_rate (Learning rate schedule for weight updates).: {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
                # max_iter : int, optional, default 200
                (MLPClassifier(hidden_layer_sizes=(i, ), activation=neuralActivation, solver=neuralSolver,
                               learning_rate=neuralLearningRate, max_iter=neuralMaxIter), name))
    return classifiers

def getSamplesAndTargets(data):
    if (SIGNIFICANT_COLS):
        # select only some of the most significant columns
        samples = data[['NumberOfPACs','ActionLatency','SelectByHotkeys','AssignToHotkeys', 'APM']]
    else:
        # get training samples (without LeagueIndex and GameID
        samples = data.drop(['GameID', 'LeagueIndex'], axis=1)

    # get training target (LeagueIndex)
    targets  = data['LeagueIndex'].values
    return samples, targets


def printResults(crossScoresMean, crossScoresStd, actual_leagues, predicted_leagues, classifier):
    print("\n", "=" * 80, "\n")
    print("=== Classifier:", classifier, "===\n")
    print("=== Classification Report: ===\n"
          "precision (How many selected elements are relevant?): TP / (TP + FP)\n"
          "recall (How many relevant elements are selected?): TP / (TP + FN)\n"
          "f1 score to measure a test's accuracy (considers both precision and recall): 2*((PR * RC)/(PR + RC))\n"
          "support: #elements in this class\n", metrics.classification_report(actual_leagues, predicted_leagues))
    print("=== Cross Validation Results: ===\n",
          "Precision: %0.2f (+/- %0.2f)\n" % (crossScoresMean[0], crossScoresStd[0]),
          "Recall:    %0.2f (+/- %0.2f)\n" % (crossScoresMean[1], crossScoresStd[1]),
          "F1 Score:  %0.2f (+/- %0.2f)\n" % (crossScoresMean[2], crossScoresStd[2]),
          "Accuracy:  %0.2f (+/- %0.2f)\n" % (crossScoresMean[3], crossScoresStd[3]))
    print("=== Confusion Matrix: ===\n"
          "top: predicted values, left: actual values\n",
          metrics.confusion_matrix(actual_leagues, predicted_leagues))
    print()
    # here we can use 'weighted' or 'macro' => weighted adjusts for the number of instances per label
    # print("f1-score:        %0.2f" % metrics.f1_score(actual_leagues, predicted_leagues, average='weighted'))
    # print("recall-score:    %0.2f" % metrics.recall_score(actual_leagues, predicted_leagues, average='weighted'))
    # print("precision-score: %0.2f" % metrics.precision_score(actual_leagues, predicted_leagues, average='weighted'))
    # print("accuracy-score:  %0.2f" % metrics.accuracy_score(actual_leagues, predicted_leagues))


def printClassifierReport(resultsPerClassifier):
    print()
    print("=" * 80)
    print("=== Report per Classifier: ===")
    printlog(resultsPerClassifier)
    resultFormat = '({:0.2f}, {:0.2f}, {:0.2f})'
    for name, results in resultsPerClassifier.items():
        print("=== %s ===" % name)
        # determine the best / worst result based on f1-score
        bestRow  = reduce((lambda x, y: x if x[2] > y[2] else y), results)[:-1]
        print("best (P, R, F):    ", resultFormat.format(*bestRow))
        worstRow = reduce((lambda x, y: x if x[2] < y[2] else y), results)[:-1]
        print("worst (P, R, F):   ", resultFormat.format(*worstRow))
        # calculate the average result.
        summedRows = results[0][:-1] if len(results) == 1 else reduce((lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2])), results)
        averageRow = list(map((lambda x: x / NUMBER_OF_RUNS), summedRows))
        print("average (P, R, F): ", resultFormat.format(*averageRow))


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
    precisionLine, = plt.plot(xAxis, precision, label='Precision')
    recallLine,    = plt.plot(xAxis, recall,    label='Recall')
    f1ScoreLine,   = plt.plot(xAxis, f1Score,   label='F1 Score')
    plt.legend(handles=[precisionLine, recallLine, f1ScoreLine])
    plt.ylabel('performance')
    plt.xlabel(X_LABEL)
    #plt.show()
    plt.savefig(PLOT_FILE_NAME)
    plt.close(fig)


def printlog(message):
    if (LOGGING):
        print(message)


if __name__ == '__main__':
    exit = main()
    sys.exit(exit)
