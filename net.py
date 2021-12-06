import torch.nn as nn
import hparams


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        """
            in_channels=1: Our images are grayscale. For an RGB image, this would be 3
            stride=1: The number of pixels to move over at each convolution
            padding=2: Ensure the convolutions cover pixels on the edge the same number of times
            as pixels not on the edge. Adds 2px of padding all around each image.
            kernel_size=5: In this case, with a single number, width = height. 5x5 matrix
            that scans over 5x5 worth of pixels (so 25 in total)
            out_channels=16: arbitrary number that tells us we want 16 channels of output features
        """

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(hparams.DROPOUT_CHANCE),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(hparams.DROPOUT_CHANCE),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=85,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(hparams.DROPOUT_CHANCE)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=85, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(hparams.DROPOUT_CHANCE)
        )

        self.fc1 = nn.Linear(in_features=(32 * 7 * 7), out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x
