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

data = dI.dataImporter(shuffle=True, stratify=True)
X_train, y_train = data.getTrainData()
#X_train = X_train[:,:-3]
# define 10-fold cross validation test harness
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)
#multiNB = make_pipeline(MultinomialNB())

#multinomial needs correct probability!!!

clf = make_pipeline(MultinomialNB())
clf2 = make_pipeline(preprocessing.StandardScaler(), BernoulliNB())
clf3 = make_pipeline(preprocessing.StandardScaler(), GaussianNB())
clf4 = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=4.0, kernel='linear', class_weight='balanced', probability=True))
scores = cross_val_score(clf, X_train[:,:-3], y_train,cv=10)
scores2 = cross_val_score(clf2, X_train, y_train,cv=10)
scores3 = cross_val_score(clf3, X_train, y_train,cv=10)
scores4 = cross_val_score(clf4, X_train, y_train,cv=10)

print("Mean ROC_AUC Multi: %.2f%% (+/- %.2f%%)" % (scores.mean()*100, scores.std()*100))
print("Mean ROC_AUC Bernu: %.2f%% (+/- %.2f%%)" % (scores2.mean()*100, scores2.std()*100))
print("Mean ROC_AUC Gauss: %.2f%% (+/- %.2f%%)" % (scores3.mean()*100, scores3.std()*100))
print("Mean ROC_AUC SVC: %.2f%% (+/- %.2f%%)" % (scores4.mean()*100, scores4.std()*100))