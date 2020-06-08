import numpy as np
import utils
import torch
import torch.utils.data as Data

data = np.load("../course_train.npy")[:utils.train_size]

def process_data(context_label=None, test=0):
    X = torch.Tensor(data[:, :-2])
    Y = torch.LongTensor(data[:, -1])   
    C = torch.LongTensor(data[:, -2])                                          # Resnet features
    if test == 0:
        if context_label is None:
            X=X[(C<utils.N_train_context).nonzero()].squeeze(1)
            Y=Y[(C<utils.N_train_context).nonzero()]
            C=C[(C<utils.N_train_context).nonzero()]
        else:
            assert(context_label < utils.context_size)
            # X = X[(C == context_label).nonzero()].squeeze(1)
            # Y = Y[(C == context_label).nonzero()]
            # C = C[(C == context_label).nonzero()]
            X = X[((C != context_label) & (C < utils.N_train_context)).nonzero()].squeeze(1)
            Y = Y[((C != context_label) & (C < utils.N_train_context)).nonzero()]
            C = C[((C != context_label) & (C < utils.N_train_context)).nonzero()]
            
    else:
        _len = len(C)
        X = torch.Tensor(data[::2, :-2])
        Y = torch.LongTensor(data[::2, -1]).unsqueeze(1) 
        C = torch.LongTensor(data[::2, -2]).unsqueeze(1) 

    print(Y.shape)
    _len = len(Y)
    Y = torch.zeros(_len, utils.class_size).scatter_(1, Y, 1)
    C = torch.zeros(_len, utils.context_size).scatter_(1, C, 1)
    # Y = torch.Tensor(np.eye(utils.class_size)[Y])     # class labels
    # C = torch.Tensor(np.eye(utils.context_size)[C])   # context labels
    return (X, Y, C)

def train_data(test=0):
    X = torch.Tensor(data[:, :-2])
    Y = torch.LongTensor(data[:, -1])   
    C = torch.LongTensor(data[:, -2])
    if test == 0:
        X=X[(C<utils.N_train_context).nonzero()].squeeze(1)
        Y=Y[(C<utils.N_train_context).nonzero()].squeeze(1)
        C=C[(C<utils.N_train_context).nonzero()].squeeze(1)
    else:
        _len = len(C)
        X = torch.Tensor(data[::2, :-2])
        Y = torch.LongTensor(data[::2, -1])   
        C = torch.LongTensor(data[::2, -2])       
    return (X, Y, C)

def test_data(test=0):
    X = torch.Tensor(data[:, :-2])
    Y = torch.LongTensor(data[:, -1])   
    C = torch.LongTensor(data[:, -2])
    if test == 0:
        X=X[(C>=utils.N_train_context).nonzero()].squeeze(1)
        Y=Y[(C>=utils.N_train_context).nonzero()].squeeze(1)
        C=C[(C>=utils.N_train_context).nonzero()].squeeze(1)
    else:
        _len = len(C)
        X = torch.Tensor(data[1::2, :-2])
        Y = torch.LongTensor(data[1::2, -1])   
        C = torch.LongTensor(data[1::2, -2]) 
    return (X, Y, C)


def load_data(test=0):
    X, Y, C = process_data(test=test)
    torch_dataset = Data.TensorDataset(X, Y, C)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=utils.batch_size,
        shuffle=True,
    )
    return loader

def load_data_by_context():
    loaders = []
    for i in range(utils.N_train_context):
        X, Y, C = process_data(i, test=0)
        torch_dataset = Data.TensorDataset(X, Y, C)
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=utils.batch_size,
            shuffle=True,
        )
        loaders.append(loader)

    return loaders

def split_validation():
    X = torch.Tensor(data[:, :-2])
    Y = torch.LongTensor(data[:, -1])   
    C = torch.LongTensor(data[:, -2])

    X_train = torch.Tensor()
    X_val = torch.Tensor()
    Y_train = torch.Tensor().long()
    Y_val = torch.Tensor().long()
    C_train = torch.Tensor().long()
    C_val = torch.Tensor().long()

    for i_label in range(utils.class_size):
        appear_contexts = C[Y == i_label].unique()
        train_contexts = appear_contexts[:5]
        val_contexts = appear_contexts[5:]

        for c in train_contexts:
            X_train = torch.cat((X_train, X[(Y == i_label) & (C == c)]))
            Y_train = torch.cat((Y_train, Y[(Y == i_label) & (C == c)]))
            C_train = torch.cat((C_train, C[(Y == i_label) & (C == c)]))
        
        for c in val_contexts:
            X_val = torch.cat((X_val, X[(Y == i_label) & (C == c)]))
            Y_val = torch.cat((Y_val, Y[(Y == i_label) & (C == c)]))
            C_val = torch.cat((C_val, C[(Y == i_label) & (C == c)]))
    
    _len = len(Y_train)
    Y_scatter = torch.zeros(_len, utils.class_size).scatter_(1, Y_train.unsqueeze(1), 1)
    C_scatter = torch.zeros(_len, utils.class_size).scatter_(1, C_train.unsqueeze(1), 1)
    torch_dataset = Data.TensorDataset(X_train, Y_scatter, C_scatter)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=utils.batch_size,
        shuffle=True,
    )

    return loader, (X_train, Y_train, C_train), (X_val, Y_val, C_val)
    

if __name__ == "__main__":
    # data = np.load("../course_train.npy")
    # print("----type----")
    # print(type(data))
    # print("----shape----")
    # print(data.shape)
    # print("----data----")
    # print(data)

    # l = load_data()
    # print(l)
    split_validation()
