import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

def stanfi():
    file_path = './data_labels.npz'  

    with np.load(file_path) as data_file:
        data = data_file['data']
        labels = data_file['labels']

    X_train, X_test, y_train, y_test = train_test_split(data, labels,test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42) # 0.25 x 0.8 = 0.2

    X_train_2d = X_train.reshape(X_train.shape[0], -1) 
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    # Standardize the data (mean=0, std=1) using training data
    scaler = StandardScaler().fit(X_train_2d)
    X_train_2d = scaler.transform(X_train_2d)
    # Apply same transformation to test data
    X_test_2d = scaler.transform(X_test_2d)

    X_train = X_train_2d.reshape(X_train.shape) 
    X_test = X_test_2d.reshape(X_test.shape)

    print("data load done with shape",X_train.shape, y_train.shape)
    
    return X_train, y_train, X_test, y_test