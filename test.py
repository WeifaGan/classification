import torch 
import torch.nn as nn
from models import resnet


device = 'cuda' if torch.cuda.is_avaliable() else 'cpu'

transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,\ 
  num_workers=2, sampler=sampler.SubsetRandomSampler(1000,)) 

net = Resnet.resnet18()
net = net.to(device)

correct,total = 0, 0
for idx,(inputs,targets) in enumerate(valloader):
    inputs,targets = inputs.to(device),targets.to(device)
    output         = net(inputs)
    _,predicted    = output.max(1)
    correct       += predicted.eq(targets).sum().item()
    total         += targets.size(0)  

print("test_acc:%.3f"%(100.*correct/total))





