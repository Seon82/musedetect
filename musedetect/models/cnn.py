from torch import nn


class CnnAudioNet(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num
        layers = []

        layers.append(nn.Conv2d(1, 64, kernel_size=(3, 3), padding="same"))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(64, 64, kernel_size=(3, 3), padding="same"))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.LeakyReLU())
        layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
        layers.append(nn.LeakyReLU())

        layers.append(nn.Conv2d(64, 128, kernel_size=(3, 3), padding="same"))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(128, 128, kernel_size=(3, 3), padding="same"))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.LeakyReLU())
        layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
        layers.append(nn.LeakyReLU())
        #
        layers.append(nn.Conv2d(128, 256, kernel_size=(3, 3), padding="same"))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(256, 256, kernel_size=(3, 3), padding="same"))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU())
        layers.append(nn.MaxPool2d(kernel_size=(3, 3)))
        layers.append(nn.LeakyReLU())
        #
        layers.append(nn.Conv2d(256, 256, kernel_size=(6, 1)))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(256))
        #
        self.seq = nn.Sequential(*layers)

        head = []
        head.append(nn.Dropout(0.5))
        head.append(nn.Linear(256, 256))
        head.append(nn.LeakyReLU())
        head.append(nn.Dropout(0.3))
        head.append(nn.Linear(256, 128))
        head.append(nn.LeakyReLU())
        head.append(nn.Dropout(0.3))
        head.append(nn.Linear(128, class_num))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.seq(x)
        return self.head(x.squeeze())
