import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
import numpy as np

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(), 
            nn.MaxPool2d(2), 
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 50),
            nn.Tanh(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(50, 10),
        )
        self.out = nn.Softmax()
        self.g_fi = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc3(self.fc2(self.fc1(x)))
        self.g_fi = x
        return self.out(x)

class FCNet(nn.Module):

    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(50, 10),
            nn.Tanh()
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

    def forward(self, x_out, I):
        loss = 0.0
        for i in range(utils.context_size):
            gx = torch.transpose(x_out, 1, 0)
            gx = gx[torch.arange(gx.size(0))!=i]
            wt = torch.transpose(self.W, 1, 0)
            ij = I[:, i]
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

