from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
import numpy as np
import pandas as pd




data = pd.DataFrame(pd.read_csv('winequality-red.csv', header=0, sep=';'))

# first row is words
np_data = data.as_matrix()
print("Data Shape: " + str(np_data.shape))

pass

np.random.shuffle(np_data)
#features_matx = np_data[1,:]
print("Full data, 15 rec: ", np_data[:15,:], "\n\n")

total_records = np_data.shape[0]

#classes
#labels = data.ix[:,-1].values.astype('int32')

train_rec = int(0.7*total_records) #approx 70%
#test_rec = total_records - train_rec
X_train = np_data[:train_rec,:-1]
y_train = np_data[:train_rec,-1]

X_test = np_data[train_rec:,:-1]
y_test = np_data[train_rec:,-1]

print("x_train: \n", X_train, "\n\n y_train: ", y_train)

y_one_hot_train = to_categorical(y_train, num_classes=11)
y_one_hot_test = to_categorical(y_test, num_classes=11)

#print("Labels: ", y_one_hot_train)
# convert list of labels to binary class matrix
#y_train = np_utils.to_categorical(labels)


input_dim = X_train.shape[1]
nb_classes = y_one_hot_train.shape[1]



model = Sequential()
model.add(Dense(64, activation='relu', input_dim=11))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(11, activation='softmax'))
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_one_hot_train, epochs=100, batch_size=256)

score = model.evaluate(X_test, y_one_hot_test, batch_size=128)
print("Final score:", score)
