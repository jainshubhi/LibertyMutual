import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score


# Import the data
train = pd.read_csv('./data/train.csv', sep=',', index_col='Id')
test = pd.read_csv('./data/test.csv', sep=',', index_col='Id')

# Convert one character strings to ascii numerals
def convert_ascii(x):
    if type(x) is str:
        return ord(x)
    else:
        return x

# Factorize data (convert alphanumerics to numerics)
train = train.applymap(convert_ascii)
test = test.applymap(convert_ascii)

# Separate training data into x and y
train_y = train['Hazard']
train_x = train.drop('Hazard', axis=1)

# Creating a model based off of the Training data
random_forest_clf = RandomForestClassifier(n_estimators=2000)
ada_boost_clf = AdaBoostClassifier(n_estimators=500)

# Performing cross validation (3 fold) on Training data
score_rfc = cross_val_score(random_forest_clf, train_x, train_y)
print score_rfc.mean()
score_abc = cross_val_score(ada_boost_clf, train_x, train_y)
print score_abc.mean()
