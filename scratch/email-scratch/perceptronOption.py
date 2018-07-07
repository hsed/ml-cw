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

class dataSet:
    data = dI.dataImporter(shuffle=True, stratify=True)
    X_train, y_train = data.getTrainData()
    ten_fold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=data.RANDOM_STATE)


class plotGridSearchResult:
    def __init__(self, title, xLabel, yLabel, metricVals, results, isLogxScale=False):
        plt.figure(figsize=(13, 13))
        plt.title(title,
          fontsize=16)

        plt.xlabel(xLabel)#("Epochs")
        plt.ylabel(yLabel)#"Score")
        plt.grid()
        ax = plt.axes()
        #ax.set_ylim(0.73, 1) # this will be done later

        # ax.set_ylim(0.73, 1)       
        #print(results.keys)
        
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, "score")]
            sample_score_std = results['std_%s_%s' % (sample, "score")]
            ax.fill_between(metricVals, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color='g')

            if isLogxScale:               
                ax.semilogx(metricVals, sample_score_mean, style, color='g',
                        alpha=1 if sample == 'test' else 0.7,
                        label="%s (%s)" % ("Accuracy", sample))
            else:               
                ax.plot(metricVals, sample_score_mean, style, color='g',
                        alpha=1 if sample == 'test' else 0.7,
                        label="%s (%s)" % ("Accuracy", sample))

        best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
        best_score = results['mean_test_%s' % "score"][best_index]

        ax.set_ylim(ax.get_ylim())

        # # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([metricVals[best_index], ] * 2, [0, best_score],
                  linestyle='-.', color='g', marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("Best CVScore: (%d, %0.2f%%)" % (metricVals[best_index], best_score*100),
                      (metricVals[best_index], best_score + 0.005))

        plt.legend(loc="best")
        plt.grid('off')
        plt.show()


def simple_perceptron(ignore):
    clf = make_pipeline(preprocessing.StandardScaler(),Perceptron(max_iter=5))
    scores = cross_val_score(clf, dataSet.X_train, dataSet.y_train,cv=10)
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
    print(df)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_perceptron__max_iter'].data, dtype=float)
    #ax.set_xlim(np.min(X_axis), np.max(X_axis)+10**2)
    print(X_axis)

    plotGridSearchResult("Plot of narrower grid-search for hyperparameter 'Epoch'", "Epochs", "ACC", X_axis, results,isLogxScale=True)

    if not loadWeights: joblib.dump(model_tuner, 'weights/' + sys._getframe().f_code.co_name + '.pkl', compress = 1)
    
    return 0