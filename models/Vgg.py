import torch 
import torch.nn as nn

class Vgg(nn.Module):
    def __init__(self,layers=16):
        super().__init__()
        self.conv1 = conv_2layer(3,64)
        self.conv2 = conv_2layer(64,128)
        self. = conv_3layer(128,256)
        self.conv4 = conv_3layer(256,512)
        self.conv5 = conv_3layer(512,512)
        self.conv3_19 = conv_3layer(128,256)
        self.conv4_19 = conv_3layer(256,512)
        self.conv5_19 = conv_3layer(512,512)
        self.maxpool = nn.MaxPool2d(2,stride=2)
        self.fc1 = nn.Linear(512,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
        self.softmax = nn.Softmax(10)

    def conv_2layer(self,in_channels,out_channels):
        Conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1),
            nn.BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU(inplace=True),
            )
        return Conv
    
    def conv_3layer(self,in_channels,out_channels):
        Conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1),
            nn.BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            )
        return Conv
    def conv_3layer_19(self,in_channels,out_channels):
        Conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1),
            nn.BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            )
        return Conv
    
    def farward(self,x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.conv2(out)
        out1 = self.maxpool(out)
        if layers==16:
            out = self.conv3(out1)
            out = self.maxpool(out)
            out = self.conv4(out)
            out = self.maxpool(out)
            out = self.conv5(out)
            out = self.maxpool(out)
        else:
            out = self.conv3_19(out1)
            out = self.maxpool(out)
            out = self.conv4_19(out)
            out = self.maxpool(out)
            out = self.conv5_19(out)
            out = self.maxpool(out)
        out = self.fc1(out) 
        out = self.fc2(out) 
        out = self.fc3(out) 
        out = self.maxpool(out)
        return out
        
        
def vgg16():
    return Vgg(16)
        
def vgg19():
    return Vgg(19)




