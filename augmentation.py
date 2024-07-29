import random
import numpy as np
import torch

def add_gaussian_noise(X_train,y_train):

    _,sequence_length,hid_dim = X_train.shape
    # #fix the random seed
    seed = 1
    deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # #Gaussian noise add
    noise_mu, noise_sigma = 0, 1

    noisetrain1 = np.random.normal(noise_mu, noise_sigma, X_train.reshape(-1,hid_dim).shape)
    noisetrain2 = np.random.normal(noise_mu, noise_sigma, X_train.reshape(-1,hid_dim).shape)
    noisetrain3 = np.random.normal(noise_mu, noise_sigma, X_train.reshape(-1,hid_dim).shape)
    noisy_data1 = X_train.reshape(-1,hid_dim) + X_train.reshape(-1,hid_dim)*noisetrain1
    noisy_data2 = X_train.reshape(-1,hid_dim) + X_train.reshape(-1,hid_dim)*noisetrain2
    noisy_data3 = X_train.reshape(-1,hid_dim) + X_train.reshape(-1,hid_dim)*noisetrain3
    noisy_data_tensor1 = torch.tensor(noisy_data1.reshape(-1,sequence_length,hid_dim), dtype=torch.float32)
    noisy_data_tensor2 = torch.tensor(noisy_data2.reshape(-1,sequence_length,hid_dim), dtype=torch.float32)
    noisy_data_tensor3 = torch.tensor(noisy_data3.reshape(-1,sequence_length,hid_dim), dtype=torch.float32)
    X_train_tensor = torch.tensor(X_train)
    X_train_gn = torch.cat([X_train_tensor,noisy_data_tensor1,noisy_data_tensor2,noisy_data_tensor3],0)
    y_train = torch.Tensor(y_train)
    y_train_gn = torch.cat([y_train,y_train,y_train,y_train],0)
    return X_train_gn, y_train_gn

def shifting(data, shift_steps, axis=1):
    # 3차원 데이터를 shift_steps만큼 이동시킵니다.
    shifted_data = np.roll(data, shift=shift_steps, axis=axis)
    return shifted_data

def augment_labels(labels, num_shifts):
    return np.tile(labels, (num_shifts, 1))

def shift(X_train,y_train):
    augmented_data_fill = []
    shifts = range(-10,10)

    for shift in shifts:
        shifted_data_fill = shifting(X_train, shift)
        augmented_data_fill.append(shifted_data_fill)

    # Combining all augmented data
    X_train_sh = np.concatenate(augmented_data_fill, axis=0)

    # Number of shifts is 6
    num_shifts = 20

    # Augmenting the labels
    y_train_sh =augment_labels(y_train, num_shifts)
    return X_train_sh, y_train_sh

def augmentation(X,y):
    #Data augmentation
    X_train_gn, y_train_gn = add_gaussian_noise(X,y) 
    X_train_sh, y_train_sh = shift(X,y)

    X_train_sh = torch.FloatTensor(X_train_sh)
    y_train_sh = torch.FloatTensor(y_train_sh)

    X_train = torch.cat([X_train_sh,X_train_gn],0)
    y_train = torch.cat([y_train_sh,y_train_gn],0)
    print("augmentation is done and X_train and y_train's shapes are",X_train.shape, y_train.shape)
    return X_train_gn, X_train_sh, X_train, y_train_gn, y_train_sh, y_train


    