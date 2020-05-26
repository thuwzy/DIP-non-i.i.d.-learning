import numpy as np
import utils
import torch
import torch.utils.data as Data

data = np.load("../course_train.npy")[:utils.train_size]

def process_data():
    X = torch.Tensor(data[:, :-2])                                              # Resnet features
    Y = torch.Tensor(np.eye(utils.class_size)[data[:, -1].astype("int32")])     # class labels
    C = torch.Tensor(np.eye(utils.context_size)[data[:, -2].astype("int32")])   # context labels
    print(X.shape, Y.shape, C.shape)
    return (X, Y, C)

def test_data():
    X = torch.Tensor(data[:, :-2])
    Y = torch.Tensor(data[:, -1])
    return (X, Y)

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