import scipy.io as sio
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split

def signfi():
    csi_lab =sio.loadmat('dataset_lab_276_dl.mat')
    csi_home =sio.loadmat('dataset_home_276.mat')

    data_lab = csi_lab['csid_lab']
    label_lab = csi_lab['label_lab']

    data_home = csi_home['csid_home']
    label_home = csi_home['label_home']

    #only home
    csi_abs_home =torch.FloatTensor(np.abs(data_home)) #amp값
    csi_abs_lab =torch.FloatTensor(np.abs(data_lab)) #amp값

    csi_abs = csi_abs_home #you can change it as home to lab

    data = csi_abs.permute(3,0,1,2)

    input_2d = data.reshape(data.shape[0], -1)

    # Standardize the data (mean=0, std=1) using training data
    scaler = StandardScaler().fit(input_2d)
    input_2d = scaler.transform(input_2d)


    input = input_2d.reshape(data.shape) #3116x192x52
    input = torch.tensor(input)

    data = input.reshape(input.shape[0],input.shape[1],-1)  
    encoder = LabelBinarizer()
    label  = encoder.fit_transform(label_home)

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42) # 0.25 x 0.8 = 0.2

    print("data load done with shape",X_train.shape, y_train.shape)
    return X_train, y_train, X_test, y_test