# Samreen Reyaz
# MNIST model reading and testing

import os
from socketserver import DatagramRequestHandler
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import cv2
import glob
import numpy as np
from mnistprep import Net
from customData import CustomData

from torch.utils.data import Dataset, DataLoader


def test(model):

    # load the test data
    test_set = DataLoader(datasets.MNIST(
        'mnist',
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=10,
        shuffle=True)

    model.eval()
    test_loss = 0
    correct = 0

    examples = enumerate(test_set)
    batch_idx, (example_data, example_targets) = next(examples)
    first_10_data = example_data[:10]

    with torch.no_grad():
        for data, target in test_set:
            output = model(first_10_data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_set.dataset)
    print(example_data[0][0].shape)
    # plot the first N images
    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction:  {}".format(output.argmax(dim=1)[i]))
        plt.xticks([])
        plt.yticks([])
        values = [round(x, 2) for x in output.detach().numpy()[i]]
        # print the output value, the index, and the correct label, and the prediction percentage
        print("Prediction:  {}; Ground Truth: {}; Output: {}".format(output.argmax(dim=1)[
              i], example_targets[i], values))

    plt.show()

    # plot the examples
    plt.figure()


def testCustomImages(model):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    # load the test data
    test_path = './self_digits/'
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    test_image_paths = []
    for path in os.listdir(test_path):
        test_image_paths.append(test_path + path)

    print(test_image_paths)

    test_image_paths = list(np.ndarray.flatten(np.array(test_image_paths)))

    print("Test size: {}".format(len(test_image_paths)))
    # show the images

    test_set = CustomData(test_image_paths, transform)

    test_loader = DataLoader(test_set, batch_size=10, shuffle=False)

    print(test_loader)

    model.eval()
    test_loss = 0
    correct = 0

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    with torch.no_grad():
        for data, target in test_loader:
            output = model(example_data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(example_data[0][0].shape)
    # plot the first N images
    plt.figure()
    for i in range(10):
        plt.subplot(3, 5, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction:  {}".format(output.argmax(dim=1)[i]))
        plt.xticks([])
        plt.yticks([])
        values = [round(x, 2) for x in output.detach().numpy()[i]]
        # print the output value, the index, and the correct label, and the prediction percentage
        print("Prediction:  {}; Ground Truth: {}; Output: {}".format(output.argmax(dim=1)[
              i], example_targets[i], values))
    plt.show()


def main(argv):
    torch.manual_seed(42)  # seed for reproducibility
    torch.backends.cudnn.enabled = False  # disable CUDA
    model = Net()
    model.load_state_dict(torch.load('./results/mnist_cnn.pth'))

    # test(model)

    testCustomImages(model)


if __name__ == "__main__":
    main(sys.argv)
