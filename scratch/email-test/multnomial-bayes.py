# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

# fix random seed for reproducibility
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

X_all = np_data[:, :-1]
y_all = np_data[:, -1].astype(int)
X_all = X_all[:, :-3] # drop last three cols as we only need probabilities

# X_train = np_data[:train_rec,:-1]
# y_train = np_data[:train_rec,-1].astype(int)

# X_test = np_data[train_rec:,:-1]
# y_test = np_data[train_rec:,-1].astype(int)

#print("x_train: \n", X_train, "\n\n y_train: ", y_train)

#raise SystemExit
#print("Labels: ", y_one_hot_train)
# convert list of labels to binary class matrix
#y_train = np_utils.to_categorical(labels)


#input_dim = X_train.shape[1]
#nb_classes = y_train.shape[1]
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cvscores = []
for train, test in kfold.split(X_all, y_all):
    # create model
    mnb = MultinomialNB()
    mnb.fit(X_all[train], y_all[train])
    h_test = mnb.predict(X_all[test])

    #print(h_test)
    

    #score = model.evaluate(X_test, y_test, batch_size=128)
    #print("Final score:", score)
    # evaluate the model
    score = accuracy_score(y_all[test], h_test)
    print("CV_Iteration: %d, Accuracy: %.2f%%" % (len(cvscores)+1, score*100))
    cvscores.append(score * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))





# accuracy from 
