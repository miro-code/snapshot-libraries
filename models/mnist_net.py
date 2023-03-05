import math

from torch import nn

class MNISTNet(nn.Module):

    def __init__(self, num_classes):
        super(MNISTNet, self).__init__()
        self.layers_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2))
        self.layers_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layers_3 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1600, num_classes),
        )

        # Initialize weights
        for layer in [self.layers_1, self.layers_2, self.layers_3]:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.layers_1(x)
        x = self.layers_2(x)
        x = self.layers_3(x)
        return x

    def forward_per_layer(self, x):
        """
        Operates like forward but additionally returns a list with elements containing the output of each layer
        """
        layer_outputs = []
        for layer in [self.layers_1, self.layers_2, self.layers_3]:
            x = layer(x)
            layer_outputs.append(x)
        return layer_outputs