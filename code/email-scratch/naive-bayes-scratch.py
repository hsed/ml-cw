import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import dataImporter as dI
from sklearn.metrics import accuracy_score

data = dI.dataImporter()
X_all, y_all = data.getAllData()

X_all = X_all[:, :-3] # drop last three cols as we only need probabilities

#print("simplified test data:\n\n", X_all[:10,:], "\nx_shape: ", np.shape(X_all))

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)

def getProbSAndProbNotS(y_vect):
    return (np.mean(y_vect), 1 - np.mean(y_vect)) # as its 0 -> not spam, 1-> spam, mean = tot/tot.no

def getProbAGivenSorNotSLst (X, y):
    probLstS = np.zeros((np.shape(X)[1],))
    probLstNotS = np.zeros((np.shape(X)[1],))
    for j in range(0, np.shape(X)[1]):
        # for each col
        probsOfAForS = []
        probsOfAForNotS = []
        for i in range(0, np.shape(X)[0]):
            if y[i] == 1:
                #spam
                probsOfAForS.append(X[i, j])
            else:
                probsOfAForNotS.append(X[i, j])
        if (len(probsOfAForS)) == 0: probsOfAForS.append(0)
        if (len(probsOfAForNotS)) == 0: probsOfAForNotS.append(0)

        probLstS[j] = np.mean(probsOfAForS)
        probLstNotS[j] = np.mean(probsOfAForNotS)
    return (probLstS, probLstNotS)

def classify(x_vect, probLstS, probLstNotS, probS, probNotS):
    # x_col_vect
    multiplierProbGivenS = 1
    multiplierProbGivenNotS = 1
    for j in range(0, len(x_vect)):
        
        if x_vect[j] > 0:
            # it contains atleast one occurence of the 'jth' word or char
            multiplierProbGivenS *= (probLstS[j])
            multiplierProbGivenNotS *= probLstNotS[j]
    if (probS*multiplierProbGivenS > probNotS*multiplierProbGivenNotS):
        return 1
    else: return 0


cvscores = []
for train, test in kfold.split(X_all, y_all):
    
    # split data
    X_train, y_train = (X_all[train], y_all[train])
    X_test, y_test = (X_all[test], y_all[test])
    h_test = np.zeros((np.shape(y_test)))   # use later
    
    # calc P(S) and P(¬S) as max likehood from test data
    probS, probNotS = getProbSAndProbNotS(y_train)

    # calc P(A | S) and P(A | ¬S) as max likehood from test data for every feature A
    probAGivenSLst, probAGivenNotSLst = getProbAGivenSorNotSLst(X_train, y_train)
    
    for i in range(0, np.shape(X_test)[0]):
        #for each row
        h_test[i] = classify(X_test[i,:], probAGivenSLst, probAGivenNotSLst, probS, probNotS)
        #print("expected: ", y_test[i], "actual: ", h_test[i])
    
    score = accuracy_score(y_all[test], h_test)
    print("CV_Iteration: %d, Accuracy: %.2f%%" % (len(cvscores)+1, score*100))
    cvscores.append(score * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
