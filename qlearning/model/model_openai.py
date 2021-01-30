import torch.nn as nn
import torch.nn.functional as F


class ModelOpenAI(nn.Module):
    """
    OpenAI architecture to solve Atari games, n grayscale frames as input
    """
    def __init__(self, input_size, input_frames, output_size):
        super(ModelOpenAI, self).__init__()

        # Define convolution and fully connected
        self.conv1 = nn.Conv2d(in_channels=input_frames, out_channels=32, kernel_size=(8, 8), padding=2, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)

        # Num features depends on input size, we resize 3 times
        num_features_fc1 = int(input_size / 8) * int(input_size / 8) * self.conv3.out_channels
        self.fc1 = nn.Linear(num_features_fc1, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        # Apply convs and relus
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # flatten
        x = x.view(x.size(0), -1)

        # Apply fully connected fc1 and fc2
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
