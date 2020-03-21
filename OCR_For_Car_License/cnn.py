import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ConvNet_Number']


class ConvNet_Number(nn.Module):
    def __init__(self, num_class):
        super(ConvNet_Number, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 10, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(10, 16, kernel_size=5, stride=1, padding=2, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_augmentation(self):
        return


class ConvNet_English(nn.Module):
    def __init__(self, num_class):
        super(ConvNet_English, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.fc1 = nn.Linear(3136, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_augmentation(self):
        return


class ConvNet_Car_License(nn.Module):
    def __init__(self, num_class, dropout=0.5):
        super(ConvNet_Car_License, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.fc1 = nn.Linear(6272, 256)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_augmentation(self):
        return