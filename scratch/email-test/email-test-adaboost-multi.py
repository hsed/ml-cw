from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from numpy.random import RandomState

data = pd.DataFrame(pd.read_csv('spambase.data', header=None, sep=','))
np_data = data.as_matrix()
X_all = np_data[:, :-1]
y_all = np_data[:, -1].astype(int)


clf = AdaBoostClassifier(n_estimators=200)
scores = cross_val_score(clf, X_all, y_all,cv=10)
print("mean cv accuracy: ", scores.mean()*100, "%")



# best result using adaboost classifier!! mean score is 93% accuracy!                 