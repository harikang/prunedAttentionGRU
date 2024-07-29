import torch
import random
import numpy as np
import scipy.io as sio

from ARIL.aril import aril
from HAR.har import har1, har3
from SignFi.signfi import signfi
from StanFi.stanfi import stanfi

from augmentation import augmentation
from DataLoader.dataloader import *

def arilsetting(batch_size):
    print("You chose ARIL DATASET!")
    #Data Load
    X_train, y_train, X_test, y_test = aril()

    #Data Augmentation
    _,_,X_train_aug, _,_,y_train_aug = augmentation(X_train, y_train)

    #Data Loader
    train_loader = dataloader(X_train_aug,y_train_aug,batch_size)
    test_loader = dataloader(X_test,y_test,batch_size)

    print("Data Loader setting is done!")
    
    return train_loader, test_loader

def harsetting1(batch_size):
    print("You chose HAR Experiment-1 DATASET!")
    #Data Load
    X_train, y_train, X_test, y_test = har1()

    #Data Augmentation
    _,_,X_train_aug, _,_,y_train_aug = augmentation(X_train, y_train)

    #Data Loader
    train_loader = dataloader(X_train_aug,y_train_aug,batch_size)
    test_loader = dataloader(X_test,y_test,batch_size)

    print("Data Loader setting is done!")
    
    return train_loader, test_loader

def harsetting3(batch_size):
    print("You chose HAR Experiment-3 DATASET!")
    #Data Load
    X_train, y_train, X_test, y_test = har3()

    #Data Augmentation
    _,_,X_train_aug, _,_,y_train_aug = augmentation(X_train, y_train)

    #Data Loader
    train_loader = dataloader(X_train_aug,y_train_aug,batch_size)
    test_loader = dataloader(X_test,y_test,batch_size)

    print("Data Loader setting is done!")
    
    return train_loader, test_loader

def signfisetting(batch_size):
    print("You chose SIGNFI DATASET!")
    #Data Load
    X_train, y_train, X_test, y_test = signfi()

    #Data Augmentation
    _,_,X_train_aug, _,_,y_train_aug = augmentation(X_train, y_train)

    #Data Loader
    train_loader = dataloader(X_train_aug,y_train_aug,batch_size)
    test_loader = dataloader(X_test,y_test,batch_size)

    print("Data Loader setting is done!")
    
    return train_loader, test_loader

def stanfisetting(batch_size):
    print("You chose STANFI DATASET!")
    #Data Load
    X_train, y_train, X_test, y_test = stanfi()

    #Data Augmentation
    _,_,X_train_aug, _,_,y_train_aug = augmentation(X_train, y_train)

    #Data Loader
    train_loader = dataloader(X_train_aug,y_train_aug,batch_size)
    test_loader = dataloader(X_test,y_test,batch_size)

    print("Data Loader setting is done!")
    
    return train_loader, test_loader
