import torch
import torch.nn as nn

class Mobilev1(nn.Module):
    def __init__(self,class_num=10):
        super().__init__()
        self.conv0 = self.conv(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.conv1_dw = self.conv_dw(in_channels=32)
        self.conv1 = self.conv(in_channels=32,out_channels=64)

        self.conv2_dw = self.conv_dw(in_channels=64,stride=2)
        self.conv2 = self.conv(in_channels=64,out_channels=128)

        self.conv3_dw = self.conv_dw(in_channels=128)
        self.conv3 = self.conv(in_channels=128,out_channels=128)

        self.conv4_dw = self.conv_dw(in_channels=128,stride=2)
        self.conv4 = self.conv(in_channels=128,out_channels=256)

        self.conv5_dw = self.conv_dw(in_channels=256)
        self.conv5 = self.conv(in_channels=256,out_channels=256)

        self.con6_dw = self.conv_dw(in_channels=256,stride=2)
        self.con6 = self.conv(in_channels=256,out_channels=512)

        self.con7 = self.conv_5x()

        self.con8_dw = self.conv_dw(in_channels=512,stride=2)
        self.conv8 = self.conv(in_channels=512,out_channels=1024)

        self.con9_dw = self.conv_dw(in_channels=1024,stride=2)
        self.conv9 = self.conv(in_channels=512,out_channels=1024)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1000,class_num)
        self.softmax = nn.Softmax(1)


    def conv_dw(self,in_channels,stride=1,num=1):
        """make a conv_dw func
        Args:
            in_channels:the number of input channels
        Return:
            return nn.Sequential object
        """

        return nn.Sequential(convDW(stride))

    def conv(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0):
        Conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        return Conv

    
    def conv_5x(self):
        """make layers
        Return:
            return nn.Sequential object
        """
        self.conv_dw(512) 

    def forward(self,x):
        print(x.size())
        out = self.conv0(x)
        print(out.size())
        out = self.conv1_dw(out)
        print(out.size())
        out = self.conv(out)
        print(out.size())
        out = self.conv2_dw(out)
        out = self.conv(out)
        out = self.conv3_dw(out)
        out = self.conv(out)
        out = self.conv5_dw(out)
        out = self.conv(out)
        out = self.conv6_dw(out)
        out = self.conv(out)
        out = self.conv7(out)
        out = self.conv8_dw(out)
        out = self.conv(out)
        out = self.conv9_dw(out)
        out = self.conv(out)
        out = self.avg_pool(out)
        out = self.fc(out)
        out = self.softmax(out)

        return out

class convDW(nn.Module):
    def __init__(self,stride):
        super().__init__()
        self.con_dw =nn.Sequential( 
        nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=stride),
        nn.BatchNorm2d(1),
        nn.ReLU(inplace=True),
        )

    
    def forward(self,x):
        channels = x.size(1)
        out = torch.empty()
        for i in range(channels): 
            out = torch.cat(self.con_dw([out,x[:,i,:,:]],1))
            
        return out 

def mobilenetv1():
    return Mobilev1() 