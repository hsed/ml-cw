import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import dataImporter as dI

data = dI.dataImporter()
X_all, y_all = data.getAllData()

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)

cvscores = []
for train, test in kfold.split(X_all, y_all):
    # create model
    gnb = GaussianNB()
    
    gnb.fit(X_all[train], y_all[train])
    h_test = gnb.predict(X_all[test])

    #print(h_test)
    
    # this is not good given the distribution!!!
    #score = model.evaluate(X_test, y_test, batch_size=128)
    #print("Final score:", score)
    # evaluate the model
    score = accuracy_score(y_all[test], h_test)
    print("CV_Iteration: %d, Accuracy: %.2f%%" % (len(cvscores)+1, score*100))
    cvscores.append(score * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


# plt.plot(xx, yy, label=name)

# plt.legend(loc="upper right")
# plt.xlabel("Proportion train")
# plt.ylabel("Test Error Rate")
# plt.show()




# accuracy from 
