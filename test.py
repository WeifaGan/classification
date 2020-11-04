import torch 
import torch.nn as nn
import torchvision
from models import *
import torchvision.transforms as transforms
from torch.utils.data import sampler

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, 
  num_workers=2,sampler=sampler.SubsetRandomSampler(range(1000,len(testset))) )

net = resnet18()
net = net.to(device)

checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
print(checkpoint['next_epoch'])
correct,total = 0, 0
for idx,(inputs,targets) in enumerate(testloader):
    inputs,targets = inputs.to(device),targets.to(device)
    output         = net(inputs)
    _,predicted    = output.max(1)
    correct       += predicted.eq(targets).sum().item()
    total         += targets.size(0)  

print("test_acc:%.3f"%(100.*correct/total))





