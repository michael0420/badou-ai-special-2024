import torch.nn as nn


def AlexNet(output_shape=2):
    # input_shape=(3, 224, 224)
    net = nn.Sequential(
        nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(48),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(48, 128, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(128, 192, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(192, 192, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(192, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Flatten(),
        nn.Linear(128 * 5 * 5, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, output_shape)
    )
    return net
