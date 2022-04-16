import sys

import cv2
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from mnistprep import getData
from net import Net, Submodel


def plot_grid(layer):
    """Plot the weights of a layer as a grid of images

    Args:
        layer (CNN layer): A layer of a CNN model
    """

    for w in range(layer.weight.shape[0]):
        plt.subplot(4, 4, w+1)
        df = pd.DataFrame(layer.weight[w].detach().numpy().reshape(5, 5))
        plt.tight_layout()
        sns.heatmap(df, cmap='YlGnBu_r', cbar=False)
        plt.title("Filter {}".format(w+1))
        plt.axis('off')
    plt.show()


def applyFilters(layer, image):
    """Apply the filters of a layer to an image and plot the results

    Args:
        image (numpy array): The image to apply the filters to.
    """

    with torch.no_grad():
        plt.figure()
        plt.tight_layout()
        # apply the 10 filters to the image
        filter = [layer.weight[i].detach().numpy().reshape(5, 5)
                  for i in range(10)]
        for i in range(10):
            plt.subplot(5, 4, i*2+1)
            sns.heatmap(filter[i], cmap='gray', cbar=False)
            plt.title("Filter {}".format(i+1))
            plt.axis('off')
            plt.subplot(5, 4, i*2+2)
            image = cv2.filter2D(image, -1, filter[i])
            plt.imshow(image, cmap='gray', interpolation='none')
            plt.title("Example {}".format(i+1))
            plt.axis('off')
    plt.show()


def applySubmodel(submodel, image):
    """Apply the submodel to an image and plot the results

    Args:
        image (numpy array): The image to apply the submodel to.
    """
    image = image[None, None]
    image = image.type(torch.FloatTensor)
    with torch.no_grad():
        output = submodel(image)
        print("Output shape: ", output.shape)
        # plot 10 channels
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.tight_layout()
            sns.heatmap(output[0][i], cmap='gray', cbar=False)
            plt.title("Channel {}".format(i+1))
            plt.axis('off')
    plt.show()


def main():
    """The entry point of the program"""
    torch.manual_seed(42)  # seed for reproducibility
    torch.backends.cudnn.enabled = False  # disable CUDA

    # load the model and print it
    model = Net()
    model.load_state_dict(torch.load('./results/mnist_cnn.pth'))
    print(model)

    # print the shape of the first layer
    # print(model.conv1.weight.shape)
    # print the filter weights
    # for i in range(model.conv1.weight.shape[0]):
    #     print("Filter {}; Shape: {}; Weights {}".format(
    #         i+1, model.conv1.weight[i].shape, model.conv1.weight[i]))

    # plot the weights using pyplot
    # plot_grid(model.conv1)

    # apply the 10 filters to the first training image
    train_set, test_set = getData()
    # examples = enumerate(train_set)
    # batch_idx, (example_data, example_targets) = next(examples)
    # first_training_image = example_data[1][0].numpy()
    # applyFilters(model.conv1, first_training_image)

    # build truncated model
    submodel = Submodel()
    submodel.load_state_dict(torch.load('./results/mnist_cnn.pth'))
    print(submodel)
    first_training_example = train_set.dataset.data[0]
    # apply the submodel to the first training image
    applySubmodel(submodel, first_training_example)

    return


if __name__ == "__main__":
    main(sys.argv)
