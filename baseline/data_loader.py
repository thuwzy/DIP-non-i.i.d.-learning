import numpy as np
import utils
import torch
import torch.utils.data as Data

final = 1
data = np.load("../course_train.npy")[:utils.train_size]
data_test = np.load("../test_data.npy")[:1200]

def process_data():
    X = torch.Tensor(data[:, :-2])
    Y = torch.LongTensor(data[:, -1])   
    C = torch.LongTensor(data[:, -2])                                          # Resnet features
    if final == 0:
        _X=X[((C+Y)%2).nonzero()].squeeze(1)
        _Y=Y[((C+Y)%2).nonzero()]
        _C=C[((C+Y)%2).nonzero()]
    else:
        _len = len(C)
        _X=X[(((C+Y)%1)+1).nonzero()].squeeze(1)
        _Y=Y[(((C+Y)%1)+1).nonzero()]
        _C=C[(((C+Y)%1)+1).nonzero()]


    _len = len(_Y)
    _Y = torch.zeros(_len, utils.class_size).scatter_(1, _Y, 1)
    _C = torch.zeros(_len, utils.context_size).scatter_(1, _C, 1)
    return (_X, _Y, _C)

def train_data():
    X = torch.Tensor(data[:, :-2])
    Y = torch.LongTensor(data[:, -1])   
    C = torch.LongTensor(data[:, -2])                                          # Resnet features
    if final == 0:
        _X=X[((C+Y)%2).nonzero()].squeeze(1)
        _Y=Y[((C+Y)%2).nonzero()].squeeze(1)
        _C=C[((C+Y)%2).nonzero()].squeeze(1)
    else:
        _len = len(C)
        _X=X[(((C+Y)%1)+1).nonzero()].squeeze(1)
        _Y=Y[(((C+Y)%1)+1).nonzero()].squeeze(1)
        _C=C[(((C+Y)%1)+1).nonzero()].squeeze(1)
    return (_X, _Y, _C)

def test_data():
    X = torch.Tensor(data[:, :-2])
    Y = torch.LongTensor(data[:, -1])   
    C = torch.LongTensor(data[:, -2])                                          # Resnet features
    if final == 0:
        _X=X[((C+Y+1)%2).nonzero()].squeeze(1)
        _Y=Y[((C+Y+1)%2).nonzero()].squeeze(1)
        _C=C[((C+Y+1)%2).nonzero()].squeeze(1)
    else:
        _len = len(C)
        _X=X[(((C+Y)%1)+1).nonzero()].squeeze(1)
        _Y=Y[(((C+Y)%1)+1).nonzero()].squeeze(1)
        _C=C[(((C+Y)%1)+1).nonzero()].squeeze(1)
    return (_X, _Y, _C)

def final_test_data():
    X = torch.Tensor(data_test[:, :])
    return X

def load_data():
    X, Y, C = process_data()
    torch_dataset = Data.TensorDataset(X, Y, C)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=utils.batch_size,
        shuffle=True,
    )
    return loader


if __name__ == "__main__":
    data = np.load("../course_train.npy")
    print("----type----")
    print(type(data))
    print("----shape----")
    print(data.shape)
    print("----data----")
    print(data)

    l = load_data()
    print(l)