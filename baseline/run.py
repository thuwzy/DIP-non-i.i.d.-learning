import torch
from torch.autograd import Variable
import utils
import niid
from data_loader import load_data, process_data, test_data, train_data

if __name__ == "__main__":
    net = niid.FCNet()
    print(net.parameters())
    optimizer_net = torch.optim.Adam(net.parameters(), lr=utils.lr)
    lossp = niid.Lossp(utils.lam)

    loader = load_data()
    for epoch in range(utils.epoch):
        for step, (x, y, _) in enumerate(loader):
            output = net(x)
            CELoss = torch.nn.CrossEntropyLoss()
            loss = CELoss(output, torch.topk(y, 1)[1].squeeze(1))
            optimizer_net.zero_grad()
            loss.backward()
            optimizer_net.step()
            # output = net(x)
            # lossb = niid.Lossb(utils.batch_size, utils.alpha)
            # optimizer_w = torch.optim.RMSprop([lossb.wn], lr=utils.lr_w)
            # wfi = Variable(net.wfi, requires_grad=False)
            
            # for i in range(utils.rounds_w):
            #     loss = lossb(wfi, x)
            #     optimizer_w.zero_grad()
            #     loss.backward()
            #     optimizer_w.step()
            #     d = lossb.update()
            #     if d < utils.threshold:
            #         break
            
            # # print(lossb.W)
            # w = Variable(lossb.W, requires_grad=False)
            # loss = lossp(wfi, output, y, w)
            # optimizer_net.zero_grad()
            # loss.backward()
            # optimizer_net.step()

            #print("epoch {} step {}, loss = {}".format(epoch, step, loss)) 
        print("-----epoch[{}]-----".format(epoch))
        (X, Y) = test_data()
        output = net(X)
        prediction = torch.argmax(output, dim=1).float()
        correct = (prediction == Y).sum().float()

        print("test acc =", correct / len(Y))

        (X, Y) = train_data()
        output = net(X)
        prediction = torch.argmax(output, dim=1).float()
        correct = (prediction == Y).sum().float()

        print("train acc =", correct / len(Y))