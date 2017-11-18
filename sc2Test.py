import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#sc2 Dataset
from sklearn.model_selection import train_test_split
# csv => DataFrame
sc2data = pd.read_csv('datasets/SkillCraft1_Dataset.csv')
print('dataset size:', sc2data.shape)

# filter out missing values
# https://stackoverflow.com/questions/27428954/drop-row-if-any-column-value-does-not-a-obey-a-condition-in-pandas
sc2data = sc2data[~(sc2data == '?').any(1)]
print('dataset size (filtered):', sc2data.shape)

# split into 80% training data, 20% test data
train, test = train_test_split(sc2data, test_size=0.2)

# get training samples (without LeagueIndex and GameID
training_samples = train.drop(['GameID', 'LeagueIndex'], axis=1)
# get training target (LeagueIndex) => this needs to be a normal array => use .values
training_target  = train['LeagueIndex'].values

# random forest model
model = RandomForestClassifier(n_estimators=10) # 10 trees
model.fit(training_samples, training_target)

# get training samples (without LeagueIndex and GameID
test_samples = test.drop(['GameID', 'LeagueIndex'], axis=1)
# get training target (LeagueIndex)
actual_leagues  = test[['LeagueIndex']]

# predict the samples
predicted_leagues = model.predict(test_samples)

# summarize the fit of the model
print(metrics.classification_report(actual_leagues, predicted_leagues))
print(metrics.confusion_matrix(actual_leagues, predicted_leagues))