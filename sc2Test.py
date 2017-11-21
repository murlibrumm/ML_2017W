import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neural_network import BernoulliRBM, MLPClassifier, MLPRegressor
from sklearn import metrics
from sklearn import preprocessing
import sys
from sklearn.model_selection import train_test_split
import numpy as np


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
MISSING_VALUES   = 'delete' # how to deal with missing values (delete, mean, median, most_frequent)
SIGNIFICATN_COLS = False    # significant columns only (like APM, PACs, #Hotkeys)

# change the classifier values here!
ALGORITHMS = ['forest', 'knn', 'bayes', 'neural'] #algorithms to use
rf_n_estimators = 10 # default: 10
knn_n_neighbors = 5 # default: 5
nb_priors = None # default: None
mlp_layers = (100) # default: (100,)


def main():
    readDataset()
    handleMissingValues()
    trainAndPredict()


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


def trainAndPredict():
    # split into 80% training data, 20% test data
    train, test = train_test_split(DATAFRAME, test_size=SPLIT_DATA)

    # get training & test samples/targets
    training_samples, training_target = getSamplesAndTargets(train)
    test_samples,     actual_leagues  = getSamplesAndTargets(test)

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

    for (model, name) in classifiers:
        # for each classifier, do the training and evaluation
        model.fit(training_samples, training_target)

        # predict the samples
        predicted_leagues = model.predict(test_samples)

        # summarize the fit of the model
        printResults(actual_leagues, predicted_leagues, name)


def getSamplesAndTargets(data):
    # get training samples (without LeagueIndex and GameID
    samples = data.drop(['GameID', 'LeagueIndex'], axis=1)
    # Test: only select some of the most significant columns
    # samples = data[['NumberOfPACs','ActionLatency','SelectByHotkeys','AssignToHotkeys', 'APM']]

    # get training target (LeagueIndex)
    targets  = data['LeagueIndex'].values
    return samples, targets


def printResults(actual_leagues, predicted_leagues, classifier):
    print("\n", "=" * 80, "\n")
    print("=== Classifier:", classifier, "===\n")
    print("=== Classification Report: ===\n"
          "precision (How many selected elements are relevant?): TP / (TP + FP)\n"
          "recall (How many relevant elements are selected?): TP / (TP + FN)\n"
          "f1 score to measure a test's accuracy (considers both precision and recall): 2*((PR + RC)/(PR + RC))\n"
          "support: #elements in this class\n", metrics.classification_report(actual_leagues, predicted_leagues))
    print("=== Confusion Matrix: ===\n"
          "top: predicted values, left: actual values\n",
          metrics.confusion_matrix(actual_leagues, predicted_leagues))
    print()
    # here we can use 'weighted' or 'macro' => weighted adjusts for the number of instances per label
    print("f1-score:        %0.2f" % metrics.f1_score(actual_leagues, predicted_leagues, average='weighted'))
    print("recall-score:    %0.2f" % metrics.recall_score(actual_leagues, predicted_leagues, average='weighted'))
    print("precision-score: %0.2f" % metrics.precision_score(actual_leagues, predicted_leagues, average='weighted'))
    print("accuracy-score:  %0.2f" % metrics.accuracy_score(actual_leagues, predicted_leagues))


def printlog(message):
    if (LOGGING):
        print(message)


if __name__ == '__main__':
    exit = main()
    sys.exit(exit)
