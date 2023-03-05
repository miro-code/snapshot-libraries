from torch import nn

class TabMLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(TabMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512,num_classes)
        )
            
    def forward(self, x):
        return self.layers(x)

    def forward_per_layer(self,x):
        #not implemented properly!
        return self.forward(x)