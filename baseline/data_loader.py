import numpy as np
import utils
import torch
import torch.utils.data as Data

data = np.load("../course_train.npy")[:utils.train_size]

def split(X, Y, C):
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
    
    return X_train, X_val, Y_train, Y_val, C_train, C_val


def split_validation():
    X = torch.Tensor(data[:, :-2])
    Y = torch.LongTensor(data[:, -1])   
    C = torch.LongTensor(data[:, -2])

    X_train, X_val, Y_train, Y_val, C_train, C_val = split(X, Y, C)
    
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
