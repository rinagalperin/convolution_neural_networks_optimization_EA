import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from conv_net import Net
from train import Train

# the experiment setup and run
class Experiment:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # initialization of the dataset we used to train the network
        # if doesn't exist - downloads it from the web automatically.
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.test_err = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_val = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

    # receives the parameters vector and number of epochs for
    # the network and returns the train error and validation error
    # after performing the training.
    def run(self, net_param, epoch):
        net = Net(net_param).cuda()
        t = Train(net, epoch, self.batch_size * 0.0002)
        t.train(self.trainloader)

        test_val = t.get_avg_lost(self.test_val)
        # the defenition of the lambda parameter, as mentioned in the report
        # and used in our formula.
        lam = 1 / 3264 
        test_err = t.get_avg_lost(self.test_err) + lam * (np.linalg.norm(net_param, 2) ** 2)
        print('\t',net_param.tolist().__str__(), 'test_val - ', test_val, 'test_err - ', test_err)
        return test_err, test_val