from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from numpy.random import RandomState
from dataImporter import dataImporter as dI
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve

#adaboost
#unnormalised -> 93.35%
#normalised -> 93.35%
#same!! :(
 
#gradient boosting
# 94.04% -- normalised and unnormalised


# on  aside note its 53% for wine
def main():
    X_all, y_all = dI().getAllData()
    clf = GradientBoostingClassifier(n_estimators=200) # AdaBoostClassifier(n_estimators=200)
    scores = cross_val_score(clf, X_all, y_all,cv=10)
    print("Mean Accuracy: %.2f%% (+/- %.2f%%)" % (scores.mean()*100, scores.std()*100))


    #print("mean cv accuracy: ", scores.mean()*100, "%")
    # best result using adaboost classifier!! mean score is 93% accuracy!

    #print("\n\n***trying out grid search!!***\n\n")

    # Gradient boosting tunning # use with caution needs high cpu!!!

    # GBC = GradientBoostingClassifier()
    # gb_param_grid = {'loss' : ["deviance"],
    #             'n_estimators' : [100,200,300],
    #             'learning_rate': [0.1, 0.05, 0.01],
    #             'max_depth': [4, 8],
    #             'min_samples_leaf': [100,150],
    #             'max_features': [0.3, 0.1] 
    #             }

    # gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)

    # gsGBC.fit(X_all,y_all)

    # GBC_best = gsGBC.best_estimator_

    # # Best score
    # print(gsGBC.best_score_)

    # https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling


if __name__ == '__main__': main()