import torch.nn as nn
import torch.nn.functional as F

# the convolutional network, receives the pararmeters vector and builds the network accordingly.
class Net(nn.Module):
    def __init__(self, net_param):
        super(Net, self).__init__()
        k1, f1, k2, f2, k3, f3 = net_param
        self.f3 = f3
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, f1, k1, padding=k1 // 2)
        self.conv2 = nn.Conv2d(f1, f2, k2, padding=k2 // 2)
        self.conv3 = nn.Conv2d(f2, f3, k3, padding=k3 // 2)
        self.fc1 = nn.Linear(self.f3 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.f3 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x