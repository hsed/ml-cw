# module data-importer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.random import RandomState

class dataImporter(object):
   

    def __init__(self, filePath='spambase.data', shuffle=True, splitRatio=0.75, stratify=True, sep=','):
        self.RANDOM_STATE = 8
        self.filePath = filePath
        data = pd.DataFrame(pd.read_csv(filePath, header=None, sep=sep))
        self.np_data_array = data.as_matrix()

        #np.random.seed(RANDOM_STATE)
        #if shuffle: np.random.shuffle(self.np_data_array) # this will be done by k-fold eval!
        #features_matx = np_data_array[1,:]
        #total_records = self.np_data_array.shape[0]

        self.X_all = self.np_data_array[:, :-1]
        self.y_all = self.np_data_array[:, -1].astype(int)

        stratifier = self.y_all if stratify else None
        self.X_train, self.X_test, self.y_train, self.y_test =      \
                        train_test_split(self.X_all, self.y_all,    \
                        test_size=(1-splitRatio), shuffle=shuffle,    \
                        stratify=stratifier, random_state=self.RANDOM_STATE)
        print("[DATAIMPORTER]\nData sizes:", "\n X_train: ", self.X_train.shape, "\ty_train: ", self.y_train.shape, \
              "\n X_test: ", self.X_test.shape, "\ty_test: ", self.y_test.shape)
        spam_pct_y_all = np.count_nonzero(self.y_all)/self.y_all.shape[0]*100
        spam_pct_y_train = np.count_nonzero(self.y_train)/self.y_train.shape[0]*100
        spam_pct_y_test = np.count_nonzero(self.y_test)/self.y_test.shape[0]*100
        print("\nSpam Ratios:", "\n y_all: %.1f%%" % (spam_pct_y_all),\
                "\ty_train: %.1f%%" % (spam_pct_y_train),\
                "\ty_test: %.1f%%" % (spam_pct_y_test), "\n\n")


        # import as default

    def getAllData(self):
        return (self.X_all, self.y_all)

    def getTrainData(self):
        return (self.X_train, self.y_train)
    
    def getTestData(self):
        return (self.X_test, self.y_test)
    
    def getRandomState(self):
        return self.RANDOM_STATE