import torch
from torch.autograd import Variable
import utils
import niid
from data_loader import split_validation

if __name__ == "__main__":
    net = niid.NIIDNet()
    net.freeze2()
    optimizer_net1 = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=utils.lr)
    net.freeze1()
    optimizer_net2 = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=utils.lr)
    net.warm()
    CELoss = torch.nn.CrossEntropyLoss()    
    softmax = torch.nn.Softmax(dim=1)
    loader, train_tuple, val_tuple = split_validation()

    for epoch in range(utils.epoch):
        if epoch >50:
            lamb=0.01
        else:
            lamb=0
        
        for step, (x, y, c) in enumerate(loader):
            net.freeze2()
            output1 = net.forward1(x)
            output2 = net.forward2(x)
            loss = CELoss(output1, torch.topk(y, 1)[1].squeeze(1)) + lamb * torch.norm(softmax(output2),p=2)
            optimizer_net1.zero_grad()
            loss.backward()
            optimizer_net1.step()

            net.freeze1()
            output2 = net.forward2(x)
            loss = CELoss(output2, torch.topk(c, 1)[1].squeeze(1))
            optimizer_net2.zero_grad()
            loss.backward()
            optimizer_net2.step()

            #print("epoch {} step {}, loss = {}".format(epoch, step, loss)) 
        
        print("-----epoch[{}]-----".format(epoch))

        # (X, Y, C) = test_data(test=1)
        X, Y, C = val_tuple
        correct1 = (torch.argmax(net.forward1(X), dim=1).float() == Y.float()).sum().float()
        correct2 = (torch.argmax(net.forward2(X), dim=1).float() == C.float()).sum().float()
        print("test acc =", correct1 / len(Y), "test C acc =", correct2 / len(C))

        # (X, Y, C) = train_data(test=1)
        X, Y, C = train_tuple
        correct1 = (torch.argmax(net.forward1(X), dim=1).float() == Y.float()).sum().float()
        correct2 = (torch.argmax(net.forward2(X), dim=1).float() == C.float()).sum().float()

        print("train acc =", correct1 / len(Y), "train C acc =", correct2 / len(C))