import scipy.io as sio
import torch
from sklearn.preprocessing import StandardScaler, LabelBinarizer
def aril():
    #load train data
    data_amp = sio.loadmat('./ARIL/train_data_split_amp.mat')
    train_data_amp = data_amp['train_data']
    train_data = train_data_amp
    train_label = data_amp['train_activity_label']

    train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label).type(torch.LongTensor)

    #load test data
    data_amp = sio.loadmat('./ARIL/test_data_split_amp.mat')
    test_data_amp = data_amp['test_data']
    test_data = test_data_amp
    test_label = data_amp['test_activity_label']

    test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)    

    train_data = train_data.transpose(1,2)
    test_data = test_data.transpose(1,2)

    X_train_2d = train_data.reshape(train_data.shape[0], -1) #1116x(192*52)
    X_test_2d = test_data.reshape(test_data.shape[0], -1)

    # Standardize the data (mean=0, std=1) using training data
    scaler = StandardScaler().fit(X_train_2d)
    X_train_2d = scaler.transform(X_train_2d)
    # Apply same transformation to test data
    X_test_2d = scaler.transform(X_test_2d)

    X_train = X_train_2d.reshape(train_data.shape) #3116x192x52
    X_test = X_test_2d.reshape(test_data.shape)

    encoder = LabelBinarizer() #labelencoder함수를 가져온다.
    y_train=encoder.fit_transform(train_label)
    y_test=encoder.fit_transform(test_label)

    print("data load done with shape",X_train.shape, y_train.shape)
    return X_train, y_train, X_test, y_test