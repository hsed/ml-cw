import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

data = pd.DataFrame(pd.read_csv('winequality-red.csv', header=0, sep=';'))

# first row is words
np_data = data.as_matrix()
#print("Data Shape: " + str(np_data.shape))

pass
X_all, y_all = np_data[:, :-1], np_data[:, -1]

for j in range(0, X_all.shape[1]):
    mean = np.mean(X_all[:, j])
    std = np.std(X_all[:, j])

    for i in range(0, X_all.shape[0]):
        X_all[i, j] = (X_all[i, j] - mean)/std

print(X_all[0], "\n", y_all[0])
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)

cvscores = []
for train, test in kfold.split(X_all, y_all):
    # create model
    logreg = LogisticRegression()
    
    logreg.fit(X_all[train], y_all[train])
    h_test = logreg.predict(X_all[test])

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
