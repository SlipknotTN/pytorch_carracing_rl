import torch.nn as nn
import torch.nn.functional as F


class ModelBaseline(nn.Module):
    """
    Baseline architecture, n grayscale frames as input
    """
    def __init__(self, input_size, input_frames, output_size):
        super(ModelBaseline, self).__init__()

        # Define convolution, poolings and fully connected

        self.conv1 = nn.Conv2d(in_channels=input_frames, out_channels=16, kernel_size=(3, 3), padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Num features depends on input size, we resize two times with 2 max poolings and conv with stride 1
        num_features_fc1 = int(input_size / 4) * int(input_size / 4) * self.conv2.out_channels
        self.fc1 = nn.Linear(num_features_fc1, 1024)
        self.fc2 = nn.Linear(1024, output_size)

    def forward(self, x):
        # Apply convs, relu, poolings
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))

        # flatten
        x = x.view(x.size(0), -1)

        # Apply fully connected fc1 and fc2
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
