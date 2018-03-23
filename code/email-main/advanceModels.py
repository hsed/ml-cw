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
    print("\n*** Best CV Accuracy: %0.2f%% ***\n" % model.best_score_)
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
    #df = pd.DataFrame(results)
    #print(df)

    # resL1 = dict()
    # resL2 = dict()
    # params = ['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']
    # for param in params:
    #     resL1[param] = np.array(results[param]).reshape(len(Cs), len(penalty))[:, 0]
    #     resL2[param] = np.array(results[param]).reshape(len(Cs), len(penalty))[:, 1]
    
    # resL1['rank_test_score'] = results['rank_test_score']
    # resL2['rank_test_score'] = results['rank_test_score']

    # plotGridSearchResult("Plot of grid-search for hyperparameter 'C' in L1 Reg", "Epochs", "ACC", Cs, resL1,isLogxScale=True)
    # plotGridSearchResult("Plot of grid-search for hyperparameter 'C' in L2 Reg", "Epochs", "ACC", Cs, resL2,isLogxScale=True)
    return 0


def nn_custom(loadWeights):


    # fix random seed for reproducibility, not in use
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

    #X_all = np_data[:, :-1]
    #y_all = np_data[:, -1].astype(int)
    scaler = preprocessing.QuantileTransformer()

    X_train = np_data[:train_rec, :-1]
    y_train = np_data[:train_rec, -1].astype(int)
    scaler.fit(X_train)

    X_test = np_data[train_rec:, :-1]
    y_test = np_data[train_rec:, -1].astype(int)
    scaler.fit_transform(X_train)
    #print("x_train: \n", X_train, "\n\n y_train: ", y_train)

    #raise SystemExit
    #print("Labels: ", y_one_hot_train)
    # convert list of labels to binary class matrix
    #y_train = np_utils.to_categorical(labels)


    #input_dim = X_train.shape[1]
    #nb_classes = y_train.shape[1]
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # create model
    model = Sequential()
    model.add(Dense(32, activation='selu', input_dim=57,
                    kernel_initializer='lecun_normal'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    cvscores = []
    for train, test in kfold.split(X_train, y_train):

        # model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
        model.fit(X_train[train], y_train[train], epochs=150, batch_size=256)

        #score = model.evaluate(X_test, y_test, batch_size=128)
        #print("Final score:", score)
        # evaluate the model
        scores = model.evaluate(X_train[test], y_train[test], verbose=0)
        print("CV_Iteration: %d, %s: %.2f%%" %
            (len(cvscores)+1, model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    print("Average Accuracy %.2f%% (+/- %.2f%%)" %
        (np.mean(cvscores), np.std(cvscores)))


    sc = model.evaluate(X_test, y_test, verbose=0)
    print("test score:%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    return 0
