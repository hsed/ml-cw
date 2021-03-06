from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer

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

#X_all = np_data[:, :-1]
#y_all = np_data[:, -1].astype(int)
scaler = QuantileTransformer()

X_train = np_data[:train_rec,:-1]
y_train = np_data[:train_rec,-1].astype(int)
scaler.fit(X_train)

X_test = np_data[train_rec:,:-1]
y_test = np_data[train_rec:,-1].astype(int)
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
model.add(Dense(32, activation='selu', input_dim=57, kernel_initializer='lecun_normal'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

cvscores = []
for train, test in kfold.split(X_train, y_train):


    model.fit(X_train[train], y_train[train], epochs=150, batch_size=256) # model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)

    #score = model.evaluate(X_test, y_test, batch_size=128)
    #print("Final score:", score)
    # evaluate the model
    scores = model.evaluate(X_train[test], y_train[test], verbose=0)
    print("CV_Iteration: %d, %s: %.2f%%" % (len(cvscores)+1,model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("Average Accuracy %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


sc = model.evaluate(X_test, y_test, verbose=0)
print("test score:%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# accuracy from 
