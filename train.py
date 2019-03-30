import torch
import torch.optim as optim
import torch.nn as nn

# responsible for the training process
class Train:
    def __init__(self, net, epoch, lr):
        self.net = net
        self.epoch = epoch
        self.lr = lr

    # runs the train according to the necessary number of epochs
    def train(self, trainloader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

        for epoch in range(self.epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()

    # calculates the average lost on the test set (either train or validation)
    def get_avg_lost(self, test_set):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_set:
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 1 - correct / total