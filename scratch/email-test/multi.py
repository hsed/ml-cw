# Author: Rob Zinkov <rob at zinkov dot com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from numpy.random import RandomState

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression


seed = 3
np.random.seed(seed)

data = pd.DataFrame(pd.read_csv('spambase.data', header=None, sep=','))

# first row is words
np_data = data.as_matrix()
print("Data Shape: " + str(np_data.shape))

#np.random.shuffle(np_data) # this will be done by k-fold eval!
#features_matx = np_data[1,:]
print("Full data, 15 rec: ", np_data[:15, :], "\n\n")

total_records = np_data.shape[0]

#classes
#labels = data.ix[:,-1].values.astype('int32')

train_rec = int(0.7*total_records)  # approx 70%
#test_rec = total_records - train_rec

X = np_data[:, :-1]
y = np_data[:, -1].astype(int)

heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
rounds = 20

classifiers = [
    ("SGD", SGDClassifier()),
    ("ASGD", SGDClassifier(average=True)),
    ("Perceptron", Perceptron()),
    ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                         C=1.0)),
    ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0)),
    ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0]))
]

xx = 1. - np.array(heldout)

for name, clf in classifiers:
    print("training %s" % name)
    rng = RandomState(42)
    yy = []
    for i in heldout:
        yy_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=i, random_state=rng)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
    plt.plot(xx, yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")
plt.show()