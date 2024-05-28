import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize


class Cifar_CNN(nn.Module):

    def __init__(self, in_planes=3, num_classes=10):
        super(Cifar_CNN, self).__init__()
        # convolution layer(32x32x3)
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=16, kernel_size=3, padding=1)
        # convolution layer(16x16x16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # convolution layer(8x8x32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # maxplooing layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # linear layer (64 * 4 * 4 -> 512)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        # linear layer (512 -> 10)
        self.fc2 = nn.Linear(512, num_classes)
        # dropout layer (p=0.3)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        # 必须输入 32 x 32， 代码里面有resize参数可以调整
        # print('x = ', x.shape)
        # x = x.view(-1, 3, 32, 32)
        # x = Resize([32, 32])(x)
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        # print(x.shape)
        x = x.view(-1,  64 * 4 * 4)
        # print(x.shape)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x


class CIFAR10Model(nn.Module):
    def __init__(self, in_planes=3, num_classes=10):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(*[
            nn.Conv2d(in_planes, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        ])
        self.cnn_block_2 = nn.Sequential(*[
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        ])
        self.flatten = lambda inp: torch.flatten(inp, 1)
        self.head = nn.Sequential(*[
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        ])

    def forward(self, X):
        X = X.view(-1, 3, 32, 32)
        X = self.cnn_block_1(X)
        X = self.cnn_block_2(X)
        X = self.flatten(X)
        X = self.head(X)
        return X