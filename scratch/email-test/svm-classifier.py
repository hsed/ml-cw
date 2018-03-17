import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from dataImporter import dataImporter as dI

# fix random seed for reproducibility
seed = 3

# unnormalised -> 90%
# normalised -> 93.5%

X_all, y_all = dI(normalise=True).getAllData()

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cvscores = []
for train, test in kfold.split(X_all, y_all):
    # create model
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(X_all[train], y_all[train])
    h_test = clf.predict(X_all[test])

    #print(h_test)
    

    #score = model.evaluate(X_test, y_test, batch_size=128)
    #print("Final score:", score)
    # evaluate the model
    score = accuracy_score(y_all[test], h_test)
    print("CV_Iteration: %d, Accuracy: %.2f%%" % (len(cvscores)+1, score*100))
    cvscores.append(score * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))





# accuracy from 
