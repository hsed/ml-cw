from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from numpy.random import RandomState
from dataImporter import dataImporter as dI
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.ensemble import *
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
#adaboost
#unnormalised -> 93.35%
#normalised -> 93.35%
#same!! :(

#gradient boosting
# 94.04% -- normalised and unnormalised


# on  aside note its 53% for wine
def main():
    X_all, y_all = dI().getTestData()


    # Gradient boosting tunning # use with caution needs high cpu!!!

    #GBC = ExtraTreesClassifier()#GradientBoostingClassifier()
    #ETC = 
    #pipe = make_pipeline(,ETC)
    pipe = Pipeline([
        ('std', preprocessing.StandardScaler()),
        ('gbc', GradientBoostingClassifier())#ExtraTreesClassifier())
    ])
    gb_param_grid = [{#'gbc__criterion' : ["gini"],
                        'gbc__n_estimators' : [100,200,250],#1000
                        'gbc__learning_rate': [0.1, 0.05, 0.01],
                        #'gbc__max_depth': [2, 4, 8, 16],
                        'gbc__min_samples_leaf': [1, 10],#100 and 200 is bad
                        'gbc__min_samples_split': [10, 100, 400],#
                        'gbc__max_features': ["auto",10, 7,1] #0.5,0.1
                    }]

    gsGBC = GridSearchCV(pipe,param_grid = gb_param_grid, cv=2, scoring="accuracy", n_jobs= -1, verbose = 2)

    gsGBC.fit(X_all,y_all)

    #GBC_best = gsGBC.best_estimator_
    #scores = cross_val_score(pipe, X_all, y_all,cv=10)
    
    print()
    print("Grid scores on development set:")
    print()
    means = gsGBC.cv_results_['mean_test_score']
    stds = gsGBC.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gsGBC.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    print()
    print("Best parameters set found on development set:")
    print()
    print(gsGBC.best_params_)
    #print("Mean Accuracy: %.2f%% (+/- %.2f%%)" % (scores.mean()*100, scores.std()*100))
    # Best score
    print(gsGBC.best_score_)
    #joblib.dump(model_tuner, 'weights/' + sys._getframe().f_code.co_name + '1.pkl', compress = 1)
    # https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling


if __name__ == '__main__': main()
