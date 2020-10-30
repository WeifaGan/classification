import torch
import torchvision
import torchvision.transforms as transforms
import os 
import argparse
from models import Resnet
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import sampler
from utils.draw import result_visual
parser = argparse.ArgumentParser(description="argument of training")
parser.add_argument('--lr',default=0.001,type=float,help="learning rate")
parser.add_argument("--resume",'-r',action='store_true',help="resume from cheakpoint")
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
print(device)

print("==>Preparing data")

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_val)
valloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2,
    sampler=sampler.SubsetRandomSampler(range(0,int(0.2*len(testset))))) 


print("==>Building model")

net = Resnet.resnet18()
net = net.to(device)

best_acc = 0
if args.resume:
    print("==>Rusuming from checkpoint")
    assert os.path.isdir('checkpoint'),'Error:no checkpoint!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-4) 

train_loss,correct,total,val_loss = 0,0,0,0
train_loss_dw,val_loss_dw,train_acc_dw, val_acc_dw = [], [],[],[]

for epoch in range(100):
    for batch_idx,(inputs,targets) in enumerate(trainloader):
        inputs,targets = inputs.to(device),targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _,predicted = outputs.max(1)
<<<<<<< HEAD
        total       = targets.size(0)
=======
        total       = target.size(0)
>>>>>>> afb34a934e7d9eb6ed347ca7a50b63472ed04edc
        correct     = predicted.eq(targets).sum().item()

        if batch_idx %10 ==0:
            for batch_idx_v,(inputs_v,target_v) in enumerate(valloader):
                inputs_v,targets_v = inputs_v.to(device),target_v.to(device)
                outputs_v = net(inputs_v)
                loss_v = criterion(outputs_v,targets_v)

                val_loss      += loss_v.item()
                _,predicted_v = outputs_v.max(1)
<<<<<<< HEAD
                total_v       = targets_v.size(0)
                correct_v     = predicted_v.eq(targets_v).sum().item()
                

                
                train_loss_dw.append(train_loss/(batch_idx+1))
                val_loss_dw.append(val_loss/(batch_idx+1))
=======
                total_v       = target_v.size(0)
                correct_v     = predicted_v.eq(targets_v).sum().item()
>>>>>>> afb34a934e7d9eb6ed347ca7a50b63472ed04edc

            
            train_loss_dw.append(train_loss/((batch_idx*(epoch+1))+1))
            val_loss_dw.append(val_loss/(batch_idx+1))
            train_acc_dw.append((100.*correct/total))
            val_acc_dw.append((100.*correct_v/total_v))

            print("train_loss:%.3f|train_acc:%.3f|val_loss:%.3f|val_acc:%.3f"%(train_loss/(batch_idx*(epoch+1))+1),\
                (100.*correct/total),val_loss/(batch_idx_v*(epoch+1)+1),(100.*correct_v/total_v))
        
        val_acc = 100.*correct_v/total_v
        if val_acc>best_acc:
            print("saving...")
            state = {
                'net': net.state_dict(),
                'acc':val_acc,
                'epoch':epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state,'./checkpoint/ckpt.pth')
            best_acc = val_acc
        
result_visual(train_loss_dw,val_loss_dw,train_acc_dw,val_acc_dw)





            








