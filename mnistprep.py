# Samreen Reyaz
# MNIST data preparation

import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

from net import Net


def getData(batch_size=64):
    """ Get the training and testing data from the MNIST data set

    Args:
        batch_size (int, optional): The batch size for training the model.
        Defaults to 64.

    Returns:
        _type_: _description_
    """

    batch_size_train = batch_size
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
    """ Plot the first numberOfExamples images in the data set

    Args:
        data_set (torch.utils.data.DataLoader): The data set to plot
        numberOfExamples (int): The number of images to plot
        plotRow (int): The number of rows in the plot
        plotCol (int): The number of columns in the plot
    """
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
        plt.axis('off')

    plt.show()


def run_epochs(train_set, test_set, net, epochs=5):
    """ Train the network for a number of epochs

    Args:
        train_set (torch.utils.data.DataLoader): The training data set
        test_set (torch.utils.data.DataLoader): The testing data set
        net (torch.nn.Module): The neural network to train
        epochs (int, optional): The number of epochs to train for.

    Returns:
        float: The accuracy on the test set
    """
    # train and test the network
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

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

        return 100. * correct / len(test_set.dataset)

    for epoch in range(1, epochs + 1):
        train(epoch)
        accuracy = test()

    # plot the loss
    # plt.figure()
    # plt.plot(train_counter, train_losses, color='blue')
    # plt.scatter(test_counter, test_losses, color='red')
    # plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    # plt.xlabel('number of training examples seen')
    # plt.ylabel('negative log likelihood loss')
    # plt.show()
    print(accuracy)
    return accuracy


def main():
    """The entry point of the program"""
    torch.manual_seed(42)  # seed for reproducibility
    torch.backends.cudnn.enabled = False  # disable CUDA

    # get the training and testing data from the MNIST data set
    train_set, test_set = getData()
    # plotSet(train_set, 6, 2, 3)

    # create the network
    net = Net()
    print(net)

    # train the network
    accuracy = run_epochs(train_set, test_set, net)
    print("Accuracy on test set: {:.4f}".format(accuracy))

    return


if __name__ == "__main__":
    main(sys.argv)
