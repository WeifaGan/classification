"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class Resnet(nn.Module):
  def __init__(self,block,block_num,num_classes=10):
    super().__init__()
    self.in_channels = 64
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,padding=4,stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3,stride=2))

    self.conv2_x = self.make_layer(BasicBlock,64,block_num[0],1)
    self.conv3_x = self.make_layer(BasicBlock,128,block_num[1],2)
    self.conv4_x = self.make_layer(BasicBlock,256,block_num[2],2)
    self.conv5_x = self.make_layer(BasicBlock,512,block_num[3],2)

    self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512*block.expansion,num_classes)

  def make_layer(self,block,out_channels,block_num,stride): 
    # out_channels is not the same for the same conv_x in res18-res152,but in_channels do.
    """make a layer for resnet

    Args:
      block:the class object of BasicBlock
      out_channels:the num channels of the output of this conv_x.
      block_num:the number of block for the conv_x
      stride:the stride of the first this conv_x.It maybe 1 or 2.

    Return:
      return a layer
    """
    strides = [stride] + [1]*(block_num-1) #ex.res34,in conv_3,the stirde of first layer of each block is 2,1,1,1,1
    layers = []
    for stirde in strides:
      layers.append(block(self.in_channels,out_channels,stride))
      self.in_channels = out_channels*block.expansion #the previous out_channels is the in_channels of next block
    return nn.Sequential(*layers)
  
  def forward(self,x):
    out = self.conv1(x)
    out = self.conv2_x(out)
    out = self.conv3_x(out)
    out = self.conv4_x(out)
    out = self.conv5_x(out)
    out = self.avg_pool(out)
    out = out.view(out.size(0),-1)
    out = self.fc(out)

    return out

        
class BasicBlock(nn.Module):
  """residual block for resnet18 and resnet34
  """
  expansion = 1 #类属性，control output channels.In res50-152,out_channels is 4 times in_channels,but res18-34,its 1 times 

  def __init__(self,in_channels,out_channels,stride):
      super().__init__()
      #out_channels of the first and second layers is the same with that of the BasicBlock
      self.resblock = nn.Sequential(
          nn.Conv2d(in_channels=in_channels,out_channels=out_channels,padding=1,kernel_size=3,stride=stride),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(in_channels=out_channels,out_channels=out_channels*BasicBlock.expansion,padding=1,kernel_size=3,stride=1),
          nn.BatchNorm2d(out_channels),
      )

      self.shortcut = nn.Sequential()

      if stride!=1 or in_channels!=out_channels*BasicBlock.expansion:
        self.shortcut = nn.Sequential(
          nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride),
          nn.BatchNorm2d(out_channels*BasicBlock.expansion),
        )




  def forward(self,x):
    """Forward propagation

    Args:
      x:input

    Return:
      return a layer
    """
    result = self.resblock(x)

    return nn.ReLU(inplace=True)(result+self.shortcut(x))


class BlockNeck(nn.Module):
  """residual block for resnet50 to resnet152 
  """
  expansion = 4 #类属性，control output channels.In res50-152,out_channels is 4 times in_channels,but res18-34,its 1 times 

  def __init__(self,in_channels,out_channels,stride):
      super().__init__()
      #out_channels of the first and second layers is the same with that of the BasicBlock
      self.resblock = nn.Sequential(
          nn.Conv2d(in_channels=in_channels,out_channels=out_channels,padding=1,kernel_size=1,stride=stride),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(in_channels=in_channels,out_channels=out_channels,padding=1,kernel_size=3,stride=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(in_channels=out_channels,out_channels=out_channels*BasicBlock.expansion,padding=1,kernel_size=3,stride=1),
          nn.BatchNorm2d(out_channels),
      )

      self.shortcut = nn.Sequential()

      if stride!=1 or in_channels!=out_channels*BasicBlock.expansion:
        self.shortcut = nn.Sequential(
          nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride),
          nn.BatchNorm2d(out_channels*BasicBlock.expansion),
        )




  def forward(self,x):
    """Forward propagation

    Args:
      x:input

    Return:
      return a layer
    """
    result = self.resblock(x)

    return nn.ReLU(inplace=True)(result+self.shortcut(x))



def resnet18():
  return Resnet(BasicBlock,[2,2,2,2])

def resnet34():
  return Resnet(BasicBlock,[2,3,6,4])

def resnet54():
  return Resnet(BlockNeck,[2,3,6,3])

def resnet101():
  return Resnet(BlockNeck,[3,4,23,3])

def resnet152():
  return Resnet(BlockNeck,[3,8,36,3])


  