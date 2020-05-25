import torch
from torch.autograd import Variable
import utils
import niid
from data_loader import load_data

if __name__ == "__main__":
    net = niid.FCNet()
    print(net.parameters())
    optimizer_net = torch.optim.Adam(net.parameters(), lr=utils.lr)
    lossp = niid.Lossp(utils.lam)

    loader = load_data()
    for epoch in range(utils.epoch):
        for step, (x, y, I) in enumerate(loader):
            output = net(x)
            lossb = niid.Lossb(utils.batch_size, utils.alpha)
            optimizer_w = torch.optim.Adam(lossb.parameters(), lr=utils.lr)
            wfi = Variable(net.wfi, requires_grad=False)
            
            for i in range(utils.rounds_w):
                loss = lossb(wfi, I)
                optimizer_w.zero_grad()
                loss.backward()
                optimizer_w.step() 
            
            w = Variable(lossb.W, requires_grad= False)
            loss = lossp(wfi, output, w)
            optimizer_net.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer_net.step()  