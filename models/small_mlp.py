from torch import nn

class SmallMLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SmallMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(True),
            nn.Linear(16, 16),
            nn.ReLU(True),
            nn.Linear(16, num_classes),
        )
            
    def forward(self, x):
        return self.layers(x)

    def forward_per_layer(self,x):
        #not implemented properly!
        return self.forward(x)