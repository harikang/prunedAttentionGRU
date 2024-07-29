from DataLoader.tensordata import TensorData
from torch.utils.data import DataLoader
def dataloader(X,y,batch_size):
    train_dataset = TensorData(X,y)
    loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)
    return loader
