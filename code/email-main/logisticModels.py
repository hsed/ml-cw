import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
import dataImporter as dI
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import sys
from sklearn.metrics import confusion_matrix
from plotFunctions import plotGridSearchResult
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import os.path
import tempfile

class dataSet:
    data = dI.dataImporter(shuffle=True, stratify=True)
    X_train, y_train = data.getTrainData()
    X_test, y_test = data.getTestData()
    ten_fold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=data.RANDOM_STATE)
    random_state = data.RANDOM_STATE

def report_summary(model):
    print()
    print("Grid scores on development set:")
    print()
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params)) 
    print()
    print("Best parameters set found on development set:")
    print()
    print(model.best_params_)
    print()
    print()
    print("\n*** Best CV Accuracy: %0.2f%% ***\n" % model.best_score_)
    print("*** Test-set accuracy: %0.2f%% ***" % (accuracy_score(dataSet.y_test, model.predict(dataSet.X_test))*100))
    print()
    



def simple_logisticreg(ignore):
    #Set C to a high value to emulate no regularisation
    clf = make_pipeline(preprocessing.StandardScaler(),SGDClassifier(loss="log", penalty=None, max_iter=1000, random_state=dataSet.random_state))
    scores = cross_val_score(clf, dataSet.X_train, dataSet.y_train,cv=dataSet.ten_fold_cv)
    print("Mean Accuracy: %.2f%% (+/- %.2f%%)" % (scores.mean()*100, scores.std()*100))
    #y_pred = scores.predict()
    #print("*** Test-set accuracy: %0.2f%% ***" % (accuracy_score(y_true, y_pred)*100))
    return 0


def tuned_logisticreg_l1_l2(loadWeights):
    #Set C to a high value to emulate no regularisation
    # pp = preprocessing.PolynomialFeatures()
    # X_Fit = pp.fit_transform(dataSet.X_train)
    # clf = make_pipeline(preprocessing.StandardScaler(),
    #         LogisticRegressionCV(penalty='l1',Cs=10, random_state=dataSet.random_state,cv=2, solver='liblinear', n_jobs=-1))
    # clf.fit(X_Fit, dataSet.y_train)
    # scores = clf.score(pp.fit_transform(dataSet.X_test),dataSet.y_test)
    # print("ss", scores)
    # ##print("Mean Accuracy: %.2f%% (+/- %.2f%%)" % (scores.mean()*100, scores.std()*100))
    # return 0

    pipe = Pipeline([
        ('std', preprocessing.StandardScaler()),
        ('lgr', LogisticRegression(random_state=dataSet.random_state))#ExtraTreesClassifier())
    ])
    Cs = [1000, 100, 10, 1.0, 0.01]
    penalty = ['l1', 'l2']
    param_grid = [{ 'lgr__penalty' : penalty,#1000
                    'lgr__C': Cs,
                    #'lgr__max_depth': [2, 4, 8, 16],
                }]

    #gsGBC = GridSearchCV(pipe,param_grid = param_grid, cv=3, scoring="accuracy", n_jobs= -1, verbose = 2)

    #gsGBC.fit(dataSet.X_train, dataSet.y_train)

    model_tuner = None
    if not loadWeights or not os.path.exists('weights/' + sys._getframe().f_code.co_name + '.pkl'):
        loadWeights = False
        model_tuner = GridSearchCV(pipe, param_grid, cv=dataSet.ten_fold_cv, n_jobs=-1, verbose=2, return_train_score=True)
        model_tuner.fit(dataSet.X_train, dataSet.y_train)
    else:
        model_tuner = joblib.load('weights/' + sys._getframe().f_code.co_name + '.pkl')

    report_summary(model_tuner)

    if not loadWeights: joblib.dump(model_tuner, 'weights/' + sys._getframe().f_code.co_name + '.pkl', compress = 1)
    
    
    results = model_tuner.cv_results_
    df = pd.DataFrame(results)
    #print(df)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_lgr__C'].data, dtype=float)
    
    #ax.set_xlim(np.min(X_axis), np.max(X_axis)+10**2)
    #print(X_axis)



    resL1 = dict()
    resL2 = dict()
    params = ['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']
    for param in params:
        resL1[param] = np.array(results[param]).reshape(len(Cs), len(penalty))[:, 0]
        resL2[param] = np.array(results[param]).reshape(len(Cs), len(penalty))[:, 1]
    
    resL1['rank_test_score'] = results['rank_test_score']
    resL2['rank_test_score'] = results['rank_test_score']

    #print("scores", resL1['mean_test_score'], resL2['mean_test_score'])
    # res['mean_test_score'] = results['mean_test_score'].reshape(len(penalty), len(Cs))
    # res['std_test_score'] = results['std_test_score'].reshape(len(penalty), len(Cs))
    # res['mean_train_score'] = results['mean_train_score'].reshape(len(penalty), len(Cs))
    # res['std_train_score'] = results['std_train_score'].reshape(len(penalty), len(Cs))

    plotGridSearchResult("Plot of grid-search for hyperparameter 'C' in L1 Reg", "Epochs", "ACC", Cs, resL1,isLogxScale=True)
    plotGridSearchResult("Plot of grid-search for hyperparameter 'C' in L2 Reg", "Epochs", "ACC", Cs, resL2,isLogxScale=True)
    return 0




def tuned_logisticreg_multi_param(loadWeights):
    #cachedir = mkdtemp()
    pipe = Pipeline([
        #('std', preprocessing.StandardScaler()),
        ('quant', preprocessing.QuantileTransformer()),
        ('lgr', SGDClassifier(loss="log", max_iter=1000, random_state=dataSet.random_state))
        #('lgr', LogisticRegression(random_state=dataSet.random_state))#ExtraTreesClassifier())
    ])
    quant = [200, 500, 800, 1000]
    dist = ['uniform', 'normal']
    l1_rat = [0.15, 0.35, 0.45 ]
    alphas = [0.001, 0.003]
    penalty = ['l1', 'l2', 'elasticnet']
    #deg = [1, 2]
    param_grid = [{ 'lgr__penalty' : penalty,#10005
                    'lgr__alpha': alphas,
                    'quant__output_distribution': dist,
                    'quant__n_quantiles': quant,
                    'lgr__l1_ratio': l1_rat
                }]

    #gsGBC = GridSearchCV(pipe,param_grid = param_grid, cv=3, scoring="accuracy", n_jobs= -1, verbose = 2)

    #gsGBC.fit(dataSet.X_train, dataSet.y_train)

    model_tuner = None
    if not loadWeights or not os.path.exists('weights/' + sys._getframe().f_code.co_name + '.pkl'):
        loadWeights = False
        model_tuner = GridSearchCV(pipe, param_grid, cv=2, n_jobs=1, verbose=2, return_train_score=True)
        model_tuner.fit(dataSet.X_train, dataSet.y_train)
    else:
        model_tuner = joblib.load('weights/' + sys._getframe().f_code.co_name + '.pkl')

    print()
    print("Grid scores on development set:")
    print()
    means = model_tuner.cv_results_['mean_test_score']
    stds = model_tuner.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model_tuner.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    print()
    print("Best parameters set found on development set:")
    print()
    print(model_tuner.best_params_)
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    ## do final test pred
    X_test, y_test = dataSet.data.getTestData()
    y_true, y_pred = y_test, model_tuner.predict(X_test)
    print("*** Test-set accuracy: %0.2f%% ***" % (accuracy_score(y_true, y_pred)*100))
    print()

    if not loadWeights: joblib.dump(model_tuner, 'weights/' + sys._getframe().f_code.co_name + '.pkl', compress = 1)
    
    results = model_tuner.cv_results_
    df = pd.DataFrame(results)
    #print(df)

    return 0
