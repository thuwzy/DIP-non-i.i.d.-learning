import torch
import torch.nn as nn

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
        self.out = nn.Sequential(
            nn.Linear(50, 10),
            nn.Softmax(),
        )

    def forward(x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc2(self.fc1(x))
        return self.out(x)

convnet = ConvNet()
print(convnet)