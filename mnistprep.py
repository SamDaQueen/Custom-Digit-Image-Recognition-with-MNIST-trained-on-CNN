# Samreen Reyaz
# MNIST data preparation

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt


class Net(nn.Module):
    # create the convulutional neural network class

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # A convolution layer with 10 5x5 filters
        x = self.conv1(x)
        # A max pooling layer with a 2x2 window and a ReLU function applied.
        x = F.relu(F.max_pool2d(x, 2))
        # A convolution layer with 20 5x5 filters
        x = self.conv2(x)
        # A dropout layer with a 0.5 dropout rate (50%)
        x = F.dropout(x, 0.5)
        # A max pooling layer with a 2x2 window and a ReLU function applied
        x = F.relu(F.max_pool2d(x, 2))
        # A flattening operation followed by a fully connected Linear layer
        # with 50 nodes and a ReLU function on the output
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        # A final fully connected Linear layer with 10 nodes and the
        # log_softmax function applied to the output.
        x = F.log_softmax(self.fc2(x), dim=1)

        return x


def getData():
    # get the training and testing data from the MNIST data set

    batch_size_train = 64
    batch_size_test = 1000

    # load MNIST data set
    train_set = torch.utils.data.DataLoader(datasets.MNIST(
        'mnist',
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size_train,
        shuffle=False)

    test_set = torch.utils.data.DataLoader(datasets.MNIST(
        'mnist',
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size_test,
        shuffle=True)

    return train_set, test_set


def plotSet(data_set, numberOfExamples, plotRow, plotCol):
    # plot the first numberOfExamples images in the data set
    print("Examining the test set")
    examples = enumerate(data_set)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data[0][0].shape)

    # plot the first N images
    plt.figure()
    for i in range(numberOfExamples):
        plt.subplot(plotRow, plotCol, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])

    plt.show()


def run_epochs(train_set, test_set, net):
    # train and test the network
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

    epochs = 5
    log_interval = 10
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_set.dataset) for i in range(1, epochs+1)]

    def train(epoch):
        net.train()
        for batch_idx, (data, target) in enumerate(train_set):
            optimizer.zero_grad()
            output = net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, batch_idx * len(data), len(train_set.dataset),
                #     100. * batch_idx / len(train_set), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(train_set.dataset)))
                # save the state of the network and optimizer after every epoch
                torch.save(net.state_dict(), './results/mnist_cnn.pth')
                torch.save(optimizer.state_dict(), './results/optimizer.pth')

    def test():
        net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_set:
                output = net(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_set.dataset)
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(test_set.dataset),
        #     100. * correct / len(test_set.dataset)))
        test_losses.append(test_loss)

    for epoch in range(1, epochs + 1):
        train(epoch)
        test()

    # plot the loss
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


def main(argv):
    torch.manual_seed(42)  # seed for reproducibility
    torch.backends.cudnn.enabled = False  # disable CUDA

    # get the training and testing data from the MNIST data set
    train_set, test_set = getData()
    # plotSet(train_set, 6, 2, 3)

    # create the network
    net = Net()
    print(net)

    # train the network
    run_epochs(train_set, test_set, net)

    return


if __name__ == "__main__":
    main(sys.argv)
