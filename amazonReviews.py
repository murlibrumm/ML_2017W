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
MISSING_VALUES   = 'mean' # how to deal with missing values (delete, mean, median, most_frequent)
SIGNIFICANT_COLS = False    # significant columns only (like APM, PACs, #Hotkeys)
NUMBER_OF_RUNS   = 4
RESULT_FORMAT    = '({:0.2f}, {:0.2f}, {:0.2f})'
CROSS_VALIDATION = False
EXPORT_PLOT      = True
X_LABEL          = 'number of nodes in the second hidden layer'
PLOT_FILE_NAME   = 'figures/amazon_neural.png'

CLASS_ENC = preprocessing.LabelEncoder()
SCALER = None
bestModel = None

# change the classifier values here!
ALGORITHMS = ['neural'] #algorithms to use ['forest', 'knn', 'bayes', 'neural']
algorithmParameter = (200, 1000+1, 50) # set a parameter in range(start, end, jump)

# forest params (algorithmParameter controls n_estimators)
forestCriterion = 'gini' # "gini" (default) for the Gini impurity 2) "entropy" for the information gain.
forestMaxDepth  = None      # how deep can a tree be max; default: none

# knn params (algorithmParameter control n_neighbors)
knnWeights   = 'distance'   # weights: 1) 'uniform' (default): weighted equally. 2) 'distance': closer neighbors => more influence
knnAlgorithm = 'auto'      # algorithm to compute the NN: {'ball_tree', 'kd_tree', 'brute', 'auto}

# bayes params TODO

# neural MLP params (algorithmParameter controls hidden_layer_sizes, default: (100,))
neuralActivation = 'relu' # (activation function for the hidden layer) : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
neuralSolver = 'adam' # (for the weight optimization): {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
neuralLearningRate = 'constant'# (Learning rate schedule for weight updates).: {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
neuralMaxIter = 200 # max_iter : int, optional, default 200

# filter warnings of the type UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def main():
    global DATAFRAME
    DATAFRAME = readDataset('datasets/amazonReviews.800.train.csv')
    DATAFRAME = encodeDataset(DATAFRAME)
    classifiers = getClassifiers()
    trainAndPredict(classifiers)
    # predictTestData()

def predictTestData():
    global bestModel
    testDf = readDataset('datasets/amazonReviews.700.test.csv')
    testDf = encodeDataset(testDf)

    testDf = prepareDataSet(testDf)

    # get training & test samples/targets
    test_samples,     actual_class = getSamplesAndTargets(testDf)


    # prediction of the provided test data
    y_predicted_test = bestModel.predict(test_samples)
    print(bestModel)
    y_provided_pred_trans = CLASS_ENC.inverse_transform(y_predicted_test)
    testdata_id = testDf['ID']
    prediction = np.c_[testdata_id, y_provided_pred_trans]
    np.savetxt("amazon_prediction.csv", prediction, delimiter=",", fmt='%s', header='ID,"class"', comments='')



def readDataset(path):
    # csv => DataFrame
    df = pd.read_csv(path)

    printlog('dataset size:' + str(df.shape))
    print(df)
    return df

def encodeDataset(df):
    return df.apply(lambda x: encodeLabels(x))

# call this method first with the trainingsdata
def prepareDataSet(data):
    global SCALER

    if 'Class' in data.columns:
        featureVal = data.drop(columns=['ID','Class'])
    else:
        featureVal = data.drop(columns=['ID'])

    if SCALER is None:
        SCALER = preprocessing.StandardScaler().fit(featureVal)
    scaled_DF = pd.DataFrame(SCALER.transform(featureVal))
    scaled_DF.columns = featureVal.columns
    scaled_DF.index = featureVal.index
    featureVal = scaled_DF

#    norm_DF = pd.DataFrame(preprocessing.normalize(featureVal, norm='l2'))
#    norm_DF.columns = featureVal.columns
#    norm_DF.index = featureVal.index
#    featureVal = norm_DF

    if 'Class' in data.columns:
        featuresWithClass = data.loc[:, 'Class':'Class'].join(featureVal)
        data = data.loc[:, 'ID':'ID'].join(featuresWithClass)
    else:
        data = data.loc[:, 'ID':'ID'].join(featureVal)

    return data

def trainAndPredict(classifiers):
    global bestModel

    resultsPerClassifier = LastUpdatedOrderedDict()
    for (model, name) in classifiers:
        resultsPerClassifier[name] = []

    # perform cross validation
    if CROSS_VALIDATION:
        X, y = getSamplesAndTargets(DATAFRAME)
        cross_scores = cross_val_score(model, X, y, cv=10)
        cross_y_pred = cross_val_predict(model, X, y, cv=10)
    else:
        y = None
        cross_scores = None
        cross_y_pred = None

    for i in range (0, NUMBER_OF_RUNS):
        # split into 80% training data, 20% test data
        train, test = train_test_split(DATAFRAME, test_size=SPLIT_DATA)

        # prepare training and test data separately since the scaling should not be affected by the test data
        train = prepareDataSet(train)
        test = prepareDataSet(test)

        # get training & test samples/targets
        training_samples, training_target = getSamplesAndTargets(train)
        test_samples,     actual_class  = getSamplesAndTargets(test)

        for (model, name) in classifiers:
            # for each classifier, do the training and evaluation
            model.fit(training_samples, training_target)

            bestModel = model

            # predict the samples
            predicted_class = model.predict(test_samples)

            # summarize the fit of the model
            printResults(y, cross_scores, cross_y_pred, actual_class, predicted_class, name)
            resultsPerClassifier[name].append(
                metrics.precision_recall_fscore_support(actual_class, predicted_class, average='weighted'))

    printClassifierReport(resultsPerClassifier)
    if EXPORT_PLOT:
        printPlot(resultsPerClassifier)

def encodeLabels(column):
    if column.name == 'Class':
        CLASS_ENC.fit(column)
        return CLASS_ENC.transform(column)
    # elif column.name == 'ID':
    #    return column
    else:
        return column

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
                (MLPClassifier(hidden_layer_sizes=(10*i, i), activation=neuralActivation, solver=neuralSolver,
                               learning_rate=neuralLearningRate, max_iter=neuralMaxIter), name))
    return classifiers

def getSamplesAndTargets(data):
    # get training samples (without class and ID)
    if 'Class' in data.columns:
        samples = data.drop(['ID', 'Class'], axis=1)

        # get training target (class)
        targets = data['Class'].values
    else:
        samples = data.drop(['ID'], axis=1)
        targets = None
    return samples, targets


def printResults(cross_actual_class, cross_score, cross_pred_class, actual_class, predicted_class, classifier):
    print("\n", "=" * 80, "\n")
    print("=== Classifier:", classifier, "===\n")
    print("=== Classification Report: ===\n"
          "precision (How many selected elements are relevant?): TP / (TP + FP)\n"
          "recall (How many relevant elements are selected?): TP / (TP + FN)\n"
          "f1 score to measure a test's accuracy (considers both precision and recall): 2*((PR * RC)/(PR + RC))\n"
          "support: #elements in this class\n", metrics.classification_report(actual_class, predicted_class))
    if CROSS_VALIDATION:
        print("=== Cross Validation Results: ===\n"
              "Accuracy: %0.2f (+/- %0.2f)" % (cross_score.mean(), cross_score.std() * 2), "\n"
              "Confusion matrix:\n",
              metrics.confusion_matrix(cross_actual_class, cross_pred_class))
    print("=== Confusion Matrix: ===\n"
          "top: predicted values, left: actual values\n",
          metrics.confusion_matrix(actual_class, predicted_class))
    print()
    # here we can use 'weighted' or 'macro' => weighted adjusts for the number of instances per label
    print("f1-score:        %0.2f" % metrics.f1_score(actual_class, predicted_class, average='weighted'))
    print("recall-score:    %0.2f" % metrics.recall_score(actual_class, predicted_class, average='weighted'))
    print("precision-score: %0.2f" % metrics.precision_score(actual_class, predicted_class, average='weighted'))
    print("accuracy-score:  %0.2f" % metrics.accuracy_score(actual_class, predicted_class))


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
