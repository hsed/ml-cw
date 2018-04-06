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
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import sys
from sklearn.metrics import confusion_matrix
from plotFunctions import plotGridSearchResult

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
    print("\n*** Mean CV Accuracy: %.2f%% (+/- %.2f%%) ***\n" % (means[model.best_index_]*100, stds[model.best_index_]*100))#(model.best_score_*100))
    print("*** Test-set accuracy: %0.2f%% ***" % (accuracy_score(dataSet.y_test, model.predict(dataSet.X_test))*100))
    print()

def simple_perceptron(ignore):
    clf = make_pipeline(preprocessing.StandardScaler(),Perceptron(max_iter=5))
    scores = cross_val_score(clf, dataSet.X_train, dataSet.y_train,cv=dataSet.ten_fold_cv)
    print("Mean Accuracy: %.2f%% (+/- %.2f%%)" % (scores.mean()*100, scores.std()*100))
    return 0

def tuned_perceptron(loadWeights=False):
    clf = make_pipeline(preprocessing.StandardScaler(),Perceptron())
    tuning_parameters = [{'perceptron__max_iter': [1, 10, 100, 1000, 10000]}]

    model_tuner = None
    if not loadWeights:
        model_tuner = GridSearchCV(clf, tuning_parameters, cv=dataSet.ten_fold_cv, n_jobs=-1, verbose=2)
        model_tuner.fit(dataSet.X_train, dataSet.y_train)
    else:
        model_tuner = joblib.load('weights/' + sys._getframe().f_code.co_name + '.pkl')

    print("Best parameters set found on development set:")
    print()
    print(model_tuner.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = model_tuner.cv_results_['mean_test_score']
    stds = model_tuner.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model_tuner.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    ## do final test pred
    X_test, y_test = dataSet.data.getTestData()
    y_true, y_pred = y_test, model_tuner.predict(X_test)
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print()

    if not loadWeights: joblib.dump(model_tuner, 'weights/' + sys._getframe().f_code.co_name + '.pkl', compress = 1)
    
    
    results = model_tuner.cv_results_
    df = pd.DataFrame(results)
    print(df)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_perceptron__max_iter'].data, dtype=float)
    #ax.set_xlim(np.min(X_axis), np.max(X_axis)+10**2)
    print(X_axis)

    plotGridSearchResult("Plot of grid-search for hyperparameter 'Epoch'", "Epochs", "ACC", X_axis, results,isLogxScale=True)

    return 0


def fine_tuned_perceptron(loadWeights=False):
    clf = make_pipeline(preprocessing.StandardScaler(),Perceptron())
    tuning_parameters = [{'perceptron__max_iter': list(range(800, 2000, 100))}]

    model_tuner = None
    if not loadWeights:
        model_tuner = GridSearchCV(clf, tuning_parameters, cv=dataSet.ten_fold_cv, n_jobs=-1, verbose=2)
        model_tuner.fit(dataSet.X_train, dataSet.y_train)
    else:
        model_tuner = joblib.load('weights/' + sys._getframe().f_code.co_name + '.pkl')

    print("Best parameters set found on development set:")
    print()
    print(model_tuner.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = model_tuner.cv_results_['mean_test_score']
    stds = model_tuner.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model_tuner.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    ## do final test pred
    X_test, y_test = dataSet.data.getTestData()
    y_true, y_pred = y_test, model_tuner.predict(X_test)
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print()

    if not loadWeights: joblib.dump(model_tuner, 'weights/' + sys._getframe().f_code.co_name + '.pkl', compress = 1)
    
    
    results = model_tuner.cv_results_
    df = pd.DataFrame(results)
    #print(df)
    report_summary(model_tuner)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_perceptron__max_iter'].data, dtype=float)
    #ax.set_xlim(np.min(X_axis), np.max(X_axis)+10**2)
    #print(X_axis)

    plotGridSearchResult("Plot of narrower grid-search for hyperparameter 'Epoch'", "Epochs", "ACC", X_axis, results,isLogxScale=True)

    if not loadWeights: joblib.dump(model_tuner, 'weights/' + sys._getframe().f_code.co_name + '.pkl', compress = 1)

    
    
    return 0