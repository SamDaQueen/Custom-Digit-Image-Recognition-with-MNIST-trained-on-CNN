import sys
from statistics import mean

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from net import GreekSubmodel


def loadData(path, batch_size=32):
    """Load the data from the given path

    Args:
        path (str): The path to the directory containing the data
        batch_size (int, optional): The batch size for the data. Defaults to 32.

    Returns:
        DataLoader: Used to loop through the data
    """

    # get the images from the folder and transform them
    image_data = datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Lambda(lambda x: 255-x),
        ]))

    # create the data loader
    data_loader = DataLoader(
        image_data,
        batch_size=batch_size,
        shuffle=True)

    return data_loader


def ssd(a, b):
    """Calculate the sum of squared differences between two vectors

    Args:
        a (list): The first vector
        b (list): The second vector

    Returns:
        float: The ssd between the two vectors
    """
    return round(sum([((float(x) - float(y))**2)/len(a) for x, y in zip(a, b)]), 3)


def testEmbedding(model, loader):
    """Test the model on the given data

    Args:
        model (Net): The model to test
        loader (DataLoader): The data to test the model on

    Returns:
        Tensor: The output of the model
    """

    model.eval()  # set model to evaluation mode
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)

    # compute the output
    with torch.no_grad():
        for data, target in loader:
            output = model(example_data)
    print(output.shape)
    return output


def match(output, example_targets):
    """Match one alpha, one beta, one gamma to all the other samples

    Args:
        output (Tensor): The outputs calculated by the model
        example_targets (list): The ground truth for the model
    """

    # compute SSD for one alpha, one beta, and one gamma
    for i in range(output.shape[0]):
        if example_targets[i] == 0:
            alpha = i
        elif example_targets[i] == 1:
            beta = i
        elif example_targets[i] == 2:
            gamma = i
        if alpha and beta and gamma:
            break

    print("Alpha index: {}, Beta index: {}, Gamma index: {}".format(alpha, beta, gamma))

    # compute the SSD for alpha
    ssd_alpha = []
    ssd_beta = []
    ssd_gamma = []
    for i in range(output.shape[0]):
        ssd_alpha.append(ssd(output[alpha], output[i]))
        ssd_beta.append(ssd(output[beta], output[i]))
        ssd_gamma.append(ssd(output[gamma], output[i]))

    print("Ground Truth: \t", [int(x) for x in example_targets])
    print("SSD for Alpha: \t", ssd_alpha)
    print("SSD for Beta: \t", ssd_beta)
    print("SSD for Gamma: \t", ssd_gamma)


def display(example_data, example_targets):
    """ Display the images and their labels

    Args:   
        example_data (Tensor): The data to display
        example_targets (list): The ground truth for the data
    """
    for i in range(10):
        plt.subplot(3, 5, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.axis('off')
    plt.show()


def knn(sample, data, test_targets):
    """Find the k nearest neighbors for the given sample

    Args:
        sample (Tensor): The sample to find the k nearest neighbors for
        data (Tensor): The data to find the k nearest neighbors for
        test_targets (list): The ground truth for the data

    Returns:
        int: The label of the k nearest neighbors
    """
    # compute the distances between the sample and all the data
    ssd_knn = []
    for i in range(data.shape[0]):
        ssd_knn.append((ssd(sample, data[i]), int(test_targets[i])))
    ssd_knn.sort()
    # use k = 3
    return max(ssd_knn[:3], key=lambda x: x[1])[1]


def main():
    """The entry point for the program"""
    torch.manual_seed(42)  # seed for reproducibility
    torch.backends.cudnn.enabled = False  # disable CUDA

    # load the model
    greekmodel = GreekSubmodel()
    greekmodel.load_state_dict(torch.load('./results/mnist_cnn.pth'))

    # load the green data
    greek_loader = loadData('./greek', 27)

    # show the first 10 images
    examples = enumerate(greek_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    # test the model on the greek data
    test_output = testEmbedding(greekmodel, greek_loader)
    # match(test_output, example_targets)
    # display(example_data, example_targets)

    customLoader = loadData('./self_greek_digits', 18)
    custom_output = testEmbedding(greekmodel, customLoader)
    custom_examples = enumerate(customLoader)
    custom_batch_idx, (custom_example_data,
                       custom_example_targets) = next(custom_examples)
    # test all the 18 images
    for i in range(18):
        pred = knn(custom_output[i], test_output, example_targets)
        print("Prediction: ", pred, " Ground Truth: ",
              int(custom_example_targets[i]))

    return


if __name__ == "__main__":
    main(sys.argv)
