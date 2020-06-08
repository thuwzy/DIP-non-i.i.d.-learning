import torch

import niid
import utils
from data_loader import split_validation

if __name__ == "__main__":
    loaders, train_tuple, val_tuple = split_validation(ensemble=True)
    X_test, Y_test, _ = val_tuple
    X_train, Y_train, _ = train_tuple

    nets = []
    optimizers = []

    for i in range(utils.N_train_context):
        nets.append(niid.FCNet())
        optimizers.append(torch.optim.Adam(nets[i].parameters(), lr=utils.lr))
    
    for epoch in range(utils.epoch):

        preds_test = torch.LongTensor([])
        preds_train = torch.LongTensor([])

        for i in range(utils.N_train_context):
            loader = loaders[i]
            net = nets[i]
            optimizer = optimizers[i]

            print('context', i)
            for step, (x, y, _) in enumerate(loader):
                output = net(x)
                CELoss = torch.nn.CrossEntropyLoss()
                loss = CELoss(output, torch.topk(y, 1)[1].squeeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print("epoch {} step {}, loss = {}".format(epoch, step, loss))
            
            output = net(X_test)
            pred = torch.argmax(output, dim=1)
            preds_test = torch.cat((preds_test, pred.unsqueeze(0)))

            output = net(X_train)
            pred = torch.argmax(output, dim=1)
            preds_train = torch.cat((preds_train, pred.unsqueeze(0)))
        
        print("-----epoch[{}]-----".format(epoch))

        for i, pred in enumerate(preds_test):
            correct = (pred == Y_test).sum().float()
            print("class", i, "test acc =", correct / len(Y_test))

        pred = torch.mode(preds_test, 0)[0]
        correct = (pred == Y_test).sum().float()

        print("test acc =", correct / len(Y_test))

        for i, pred in enumerate(preds_train):
            correct = (pred == Y_train).sum().float()
            print("class", i, "train acc =", correct / len(Y_train))

        pred = torch.mode(preds_train, 0)[0]
        correct = (pred == Y_train).sum().float()

        print("train acc =", correct / len(Y_train))