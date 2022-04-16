import sys

import torch

from mnistprep import getData, run_epochs
from net import Net


def main():
    """The entry point of the program"""
    torch.manual_seed(42)  # seed for reproducibility
    torch.backends.cudnn.enabled = False  # disable CUDA

    output = "epochs, pool, batch_size, dropout"
    # loop the number of epochs from 3 to 6 with a step of 1
    for epoch in range(3, 6):
        # switch between max and average pool
        for pool in range(0, 1):
            # loop the batch size between 2^5 and 2^8 with a step of 2^1
            for batch_size in range(5, 9):
                # loop the dropout rate between 0.2 to 0.6 with a step of 0.2
                for dropout in range(2, 6):
                    # get the training and testing data from the MNIST data set
                    train_set, test_set = getData(batch_size)
                    # create the network
                    net = Net(pool, dropout/10)
                    accuracy = run_epochs(train_set, test_set, net, epoch)

                    output += "\n{}, {}, {}, {}, {}".format(
                        epoch, pool, batch_size, dropout/10, accuracy)

        print(output)
    return


if __name__ == "__main__":
    main(sys.argv)
