import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
import numpy as np

class FCNet(nn.Module):

    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(50, 10)
            #nn.Tanh()
        )
        self.out = nn.Softmax(dim=1)
        self.wfi = None

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        self.wfi = x
        #print(x.shape)
        return self.out(x)

class Lossb(nn.Module):

    def __init__(self, n, alpha):
        super(Lossb, self).__init__()
        w = 1.0 / n
        self.n = n
        self.alpha = alpha
        self.wn = Variable(w * torch.ones((n - 1, 1)), requires_grad=True)
        self.W = torch.cat((self.wn, torch.ones((1,1)) - torch.sum(self.wn)))
        self.diff = torch.Tensor(self.W)

    def forward(self, x_out, x):
        loss = 0.0
        for i in range(utils.feature_size):
            ij = (x[:, i] > 0.5).float()

            gx = torch.transpose(x_out, 1, 0)
            gx = gx[torch.arange(gx.size(0))!=i]
            wt = torch.transpose(self.W, 1, 0)
            ten1 = torch.matmul(gx, (self.W * ij)) / torch.matmul(wt, ij)
            ten2 = torch.matmul(gx, (self.W * (1 - ij))) / torch.matmul(wt, 1 - ij)
            loss += torch.norm(ten1 - ten2) ** 2
        loss += self.alpha * (torch.norm(self.W) ** 2)
        return loss

    def update(self):
        new_w = torch.cat((self.wn, torch.ones((1,1)) - torch.sum(self.wn)))
        self.diff = new_w - self.W
        self.W = new_w
        diff = torch.norm(self.diff, p=1)
        return diff

class Lossp(nn.Module):

    def __init__(self, lam):
        super(Lossp, self).__init__()
        self.lam = lam

    def forward(self, gout, fout, y, W):
        lossq = -utils.lam * (torch.norm(gout)**2)
        loss = torch.sum(W * torch.log(torch.sum((fout * y), dim=1))) + lossq
        return loss

