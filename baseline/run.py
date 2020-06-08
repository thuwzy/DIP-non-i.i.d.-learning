import torch
from torch.autograd import Variable
import utils
import niid
from data_loader import split_validation

EPOCH = 100

if __name__ == "__main__":
    net = niid.FCNet()
    print(net.parameters())
    optimizer_net = torch.optim.Adam(net.parameters(), lr=utils.lr)

    loader, train_tuple, val_tuple = split_validation()
    for epoch in range(EPOCH):
        for step, (x, y, _) in enumerate(loader):
            output = net(x)
            CELoss = torch.nn.CrossEntropyLoss()
            loss = CELoss(output, torch.topk(y, 1)[1].squeeze(1))
            optimizer_net.zero_grad()
            loss.backward()
            optimizer_net.step()
        
        print("-----epoch[{}]-----".format(epoch))
        (X, Y, _) = val_tuple
        output = net(X)
        prediction = torch.argmax(output, dim=1).float()
        correct = (prediction == Y.float()).sum().float()

        print("test acc =", correct / len(Y))

        (X, Y, _) = train_tuple
        output = net(X)
        prediction = torch.argmax(output, dim=1).float()
        correct = (prediction == Y.float()).sum().float()

        print("train acc =", correct / len(Y))