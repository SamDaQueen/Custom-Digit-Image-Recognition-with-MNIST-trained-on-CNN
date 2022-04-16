import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Create the convolutional neural network architecture."""

    def __init__(self, pool=0, dropout=0.5) -> None:
        """The constructor for defining the layers of the network.

        Args:
            pool (int): The pooling method to use. 0 for max pooling and 1 for average pooling.
            dropout (float): The dropout rate to use.
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.pool = pool
        self.dropout = dropout

    def forward(self, x):
        """The training function for the network.

        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output of the network.
        """
        # A convolution layer with 10 5x5 filters
        x = self.conv1(x)
        # A max pooling layer with a 2x2 window and a ReLU function applied.
        if self.pool == 0:
            x = F.relu(F.max_pool2d(x, 2))
        else:
            x = F.relu(F.avg_pool2d(x, 2))
        # A convolution layer with 20 5x5 filters
        x = self.conv2(x)
        # A dropout layer with a 0.5 dropout rate (50%)
        x = F.dropout(x, self.dropout)
        # A max pooling layer with a 2x2 window and a ReLU function applied
        if self.pool == 0:
            x = F.relu(F.max_pool2d(x, 2))
        else:
            x = F.relu(F.avg_pool2d(x, 2))
        # A flattening operation followed by a fully connected Linear layer
        # with 50 nodes and a ReLU function on the output
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        # A final fully connected Linear layer with 10 nodes and the
        # log_softmax function applied to the output.
        x = F.log_softmax(self.fc2(x), dim=1)

        return x


class Submodel(Net):
    """A sub model inheriting from the Net class."""

    def __init__(self, *args, **kwargs):
        """The constructor for defining the layers of the network."""

        super().__init__(*args, **kwargs)

    # override the forward method
    def forward(self, x):
        # relu on max pooled results of conv1
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # return x
        # relu on max pooled results of dropout of conv2
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        return x


class GreekSubmodel(Net):
    """A sub model inheriting from the Net class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(x, dim=1)

        return x
