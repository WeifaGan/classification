import torch 
import torch.nn as nn

class Vgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU(inplace=True),
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU(inplace=True),
        )


        self.maxpool = nn.MaxPool2d(2,stride=2)

