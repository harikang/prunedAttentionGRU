import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import pandas as pd
import glob, os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin

class Standard_Scaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X
    
def har1():
    # Load the data
    X_train = np.load('./HAR/X_train.npy')
    X_test = np.load('./HAR/X_test.npy')
    y_train = np.load('./HAR/y_train.npy')
    y_test = np.load('./HAR/y_test.npy')

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    # y_valid_encoded = label_encoder.fit_transform(y_valid)
    y_test_encoded = label_encoder.fit_transform(y_test)
    label_binizer = LabelBinarizer()
    y_train = label_binizer.fit_transform(y_train_encoded)
    # y_valid = label_binizer.fit_transform(y_valid_encoded)
    y_test = label_binizer.fit_transform(y_test_encoded)
    print("HAR EXPERIMENT-1 train X,y shape is ",X_train.shape, y_train.shape)
    return X_train, y_train, X_test, y_test


# function for reading CSV files 
def reading_file(activity_csv):     
    results = []
    for i in range(len(activity_csv)):
        df = pd.read_csv(activity_csv[i])
        results.append(df.values)  
    return results
#function for labeling the samples 
def label(activity, label):
    list_y = []
    for i in range(len(activity)):
        list_y.append(label)
    return np.array(list_y).reshape(-1, 1) 

def har3():
    #Read Dataset
    path = "./data" #set path
    os.chdir(path) 
    results = pd.DataFrame([])
    list_file = glob.glob("*.csv") #lisiting all the csv file samples
    #print(list_file)
 

    empty_csv = [i for i in list_file if i.startswith('Empty')] #list for empty csv files 
    lying_csv = [i for i in list_file if i.startswith('Lying')] #list for lying csv files 
    sitting_csv = [i for i in list_file if i.startswith('Sitting')] #list for sitting csv files 
    standing_csv = [i for i in list_file if i.startswith('Standing')] #list for satnding csv files 
    walking_csv = [i for i in list_file if i.startswith('Walking')] #list for walking csv files 

    #calling reading_file function  
    empty = reading_file(empty_csv) 
    lying = reading_file(lying_csv)
    sitting = reading_file(sitting_csv)
    standing = reading_file(standing_csv)
    walking = reading_file(walking_csv)

    walking_label = label(walking, 'walking') 
    empty_label = label(empty, 'empty') 
    lying_label = label(lying, 'lying') 
    sitting_label = label(sitting, 'sitting') 
    standing_label = label(standing, 'standing') 

    #concatenate all the samples into one np array 
    array_tuple = (empty, lying, sitting,standing, walking)
    data_X = np.vstack(array_tuple)

    #concatenate all the label into one array 
    label_tuple = (empty_label, lying_label, sitting_label,standing_label,  walking_label)
    data_y = np.vstack(label_tuple)

    #randomize the sample 

    data, label = shuffle(data_X, data_y)

    label_binarizer = LabelBinarizer()
    label = label_binarizer.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state=42)

    sc = Standard_Scaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform (X_test)

    print("HAR EXPERIMENT-3 train X, y shape is ",X_train.shape, y_train.shape)
    return X_train, y_train, X_test, y_test