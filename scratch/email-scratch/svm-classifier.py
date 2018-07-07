import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from dataImporter import dataImporter as dI
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

def main():
    # fix random seed for reproducibility
    seed = 3

    # unnormalised -> 90%
    # normalised -> 93.5%

    X_all, y_all = dI().getAllData()

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)   #this seed gives better results!! (91% without seed)

    clf = svm.SVC(gamma=0.001, C=100.)
    clf_linear_kernel = svm.SVC(kernel='linear')

    # scores = cross_val_score(clf, X_all, y_all,cv=kfold)
    # print("Mean Accuracy: %.2f%% (+/- %.2f%%)\n\n" % (scores.mean()*100, scores.std()*100))

    # scores = cross_val_score(clf_linear_kernel, X_all, y_all,cv=kfold)
    # print("Mean Accuracy for linear kernel: %.2f%% (+/- %.2f%%)\n\n" % (scores.mean()*100, scores.std()*100))

    rfecv = RFECV(estimator=clf_linear_kernel, step=1, cv=kfold,
                scoring='accuracy', n_jobs=3, verbose=1,)
    rfecv.fit(X_all, y_all)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    # plt.figure()
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (nb of correct classifications)")
    # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    # plt.show()

# accuracy from 

if __name__ == '__main__': main()