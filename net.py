import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDiscrim(nn.Module):

    def __init__(self, x_dim):
        super(ConvDiscrim, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(1,-1),
            nn.Linear(512*4*4,1)
        )
    
    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):

    def __init__(self, x_dim):
        super(Discriminator, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(x_dim, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        return self.layers(x)
