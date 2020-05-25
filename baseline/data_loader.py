import numpy as np
import utils
import torch
import torch.utils.data as Data

def process_data():
    data = np.load("../course_train.npy")
    X = torch.Tensor(data[:, :-2])
    Y = torch.Tensor(np.eye(utils.class_size)[data[:, -2].astype("int32")])
    C = torch.Tensor(np.eye(utils.context_size)[data[:, -1].astype("int32")])
    print(X.shape, Y.shape, C.shape)
    return (X, Y, C)

def load_data():
    data = np.load("../course_train.npy")
    X = data[:, :-2]
    Y = data[:, -2:]
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