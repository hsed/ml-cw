from sklearn.model_selection import StratifiedKFold, cross_val_score,GridSearchCV,learning_curve
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from numpy.random import RandomState
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from plotFunctions import plotGridSearchResult
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.models import load_model
from keras.utils import plot_model
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
import tempfile
import dataImporter as dI

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
    
    



def simple_gradboost(ignore):
    #preprocessing.QuantileTransformer(), only .05% increase maybe
    clf = Pipeline([('gradboost', GradientBoostingClassifier(random_state=dataSet.random_state))])
    scores = cross_val_score(clf, dataSet.X_train, dataSet.y_train,cv=dataSet.ten_fold_cv)
    print("Mean CV Accuracy: %.2f%% (+/- %.2f%%)" % (scores.mean()*100, scores.std()*100))

    #train on entire test set
    clf.fit(dataSet.X_train, dataSet.y_train)
    y_pred = clf.predict(dataSet.X_test)
    print("*** Test-set accuracy: %0.2f%% ***" % (accuracy_score(dataSet.y_test, y_pred)*100))
    return 0


def tuned_gradboost(loadWeights):
    pipe = Pipeline([
        ('std', preprocessing.QuantileTransformer()),
        ('gbc', GradientBoostingClassifier())  # ExtraTreesClassifier())
    ])
    param_grid = [{  # 'gbc__criterion' : ["gini"],#gini is good
                        'gbc__n_estimators': [100, 200, 250],  # 1000
                        'gbc__learning_rate': [0.1, 0.05, 0.01],
                        #'gbc__max_depth': [2, 4, 8, 16],
                        'gbc__min_samples_leaf': [1, 10],  # 100 and 200 is bad
                        'gbc__min_samples_split': [10, 100, 400],
                        'gbc__max_features': ["auto", 10, 7, 1]  # 0.5,0.1
    }]

    #gsGBC = GridSearchCV(pipe,param_grid = param_grid, cv=3, scoring="accuracy", n_jobs= -1, verbose = 2)

    #gsGBC.fit(dataSet.X_train, dataSet.y_train)

    model_tuner = None
    if not loadWeights or not os.path.exists('weights/' + sys._getframe().f_code.co_name + '.pkl'):
        loadWeights = False
        model_tuner = GridSearchCV(pipe, param_grid, cv=2, n_jobs=-1, verbose=2, return_train_score=True) #cv=dataSet.ten_fold_cv
        model_tuner.fit(dataSet.X_train, dataSet.y_train)
    else:
        model_tuner = joblib.load('weights/' + sys._getframe().f_code.co_name + '.pkl')

    report_summary(model_tuner)

    if not loadWeights: joblib.dump(model_tuner, 'weights/' + sys._getframe().f_code.co_name + '.pkl', compress = 1)
    
    
    results = model_tuner.cv_results_

    return 0

def nn_simple(loadWeights):


    # fix random seed for reproducibility, not in use
    seed = 3
    np.random.seed(seed)

    X_train = dataSet.X_train
    y_train = dataSet.y_train

    X_test = dataSet.X_test
    y_test = dataSet.y_test


    isPrevModelLoaded = False
    #input_dim = X_train.shape[1]
    #nb_classes = y_train.shape[1]
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    model = None
    if not loadWeights or not os.path.exists('weights/' + sys._getframe().f_code.co_name + '.h5'):
        loadWeights = False
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=57))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    else:
        model = load_model('weights/' + sys._getframe().f_code.co_name + '.h5')
        isPrevModelLoaded = True

    cvscores = []
    for train, test in kfold.split(X_train, y_train):

        model.fit(X_train[train], y_train[train], epochs=150, batch_size=256, verbose=0)

        scores = model.evaluate(X_train[test], y_train[test], verbose=0)
        print("CV_Iteration: %d, %s: %.2f%%" %
            (len(cvscores)+1, model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    
    print("Mean CV Accuracy %.2f%% (+/- %.2f%%)" %
        (np.mean(cvscores), np.std(cvscores)))

    sc = model.evaluate(X_test, y_test, verbose=0)
    print("Test-set Accuracy:%s: %.2f%%" % (model.metrics_names[1], sc[1]*100))
    if not loadWeights: model.save('weights/' + sys._getframe().f_code.co_name + '.h5')
    return 0


def nn_custom(loadWeights):


    # fix random seed for reproducibility, not in use
    seed = 3
    np.random.seed(seed)

    scaler = preprocessing.QuantileTransformer()

    X_train = dataSet.X_train
    y_train = dataSet.y_train
    scaler.fit(X_train)
    scaler.fit_transform(X_train)

    X_test = dataSet.X_test
    y_test = dataSet.y_test
    scaler.fit_transform(X_test)
    #print("x_train: \n", X_train, "\n\n y_train: ", y_train)

    #raise SystemExit
    #print("Labels: ", y_one_hot_train)
    # convert list of labels to binary class matrix
    #y_train = np_utils.to_categorical(labels)

    isPrevModelLoaded = False
    #input_dim = X_train.shape[1]
    #nb_classes = y_train.shape[1]
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    model = None
    if not loadWeights or not os.path.exists('weights/' + sys._getframe().f_code.co_name + '.h5'):
        loadWeights = False
        model = Sequential()
        model.add(Dense(32, activation='selu', input_dim=57,
                    kernel_initializer='lecun_normal'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    else:
        model = load_model('weights/' + sys._getframe().f_code.co_name + '.h5')
        isPrevModelLoaded = True


    cvscores = []
    for train, test in kfold.split(X_train, y_train):

        # model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
        if not isPrevModelLoaded:
            model.fit(X_train[train], y_train[train], epochs=150, batch_size=256, verbose=0)

            # summarize history for accuracy
            # hist[:,i_hist] = history.history['acc']
            # i_hist +=1

        scores = model.evaluate(X_train[test], y_train[test], verbose=0)
        print("CV_Iteration: %d, %s: %.2f%%" %
            (len(cvscores)+1, model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    
    print("Mean CV Accuracy %.2f%% (+/- %.2f%%)" %
        (np.mean(cvscores), np.std(cvscores)))

    sc = model.evaluate(X_test, y_test, verbose=0)
    print("Test-set Accuracy:%s: %.2f%%" % (model.metrics_names[1], sc[1]*100))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    #plot_model(model, to_file='model.png', show_shapes=True)
    if not loadWeights: model.save('weights/' + sys._getframe().f_code.co_name + '.h5')
    return 0
