import torch
import torchvision
import torchvision.transforms as transforms
import os 
import argparse

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import sampler
from utils.draw import result_visual
import utils.get_network as get_network
parser = argparse.ArgumentParser(description="argument of training")
parser.add_argument('--network',type=str,help="the network to train")
parser.add_argument('--lr',default=0.1,type=float,help="learning rate")
parser.add_argument('--epoch',default=200,type=int,help="total number of epoch")
parser.add_argument('--batch_size',default=64,type=int,help="total number of epoch")
parser.add_argument("--resume",'-r',action='store_true',help="resume from cheakpoint")
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
print(device)

print("==>Preparing data")

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=False, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_val)
valloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    sampler=sampler.SubsetRandomSampler(range(0,int(0.2*len(testset))))) 

torch.backends.cudnn.benchmark = True
print("==>Building model")

net = get_network.get(args.network) 
# net = mobilenetv1()
net = net.to(device)

best_loss = float('inf') 
start_epoch = 0

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=args.lr,momentum=0.95, weight_decay=5e-4) 
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[80,130,170], gamma=0.1)

if args.resume:
    print("==>Rusuming from checkpoint")
    assert os.path.isdir('checkpoint'),'Error:no checkpoint!'
    checkpoint = torch.load('./checkpoint/best_ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    print(checkpoint.keys())
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch']  # 设置开始的epoch
    lr_scheduler.load_state_dict(checkpoint['lr_schedule'])#加载lr_scheduler


train_loss_dw,val_loss_dw,train_acc_dw, val_acc_dw = [], [],[],[]

for epoch in range(start_epoch,args.epoch):
    train_loss,correct,total = 0,0,0
    for batch_idx,(inputs,targets) in enumerate(trainloader):
        inputs,targets = inputs.to(device),targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _,predicted = outputs.max(1)

        total       = targets.size(0)
        correct     = predicted.eq(targets).sum().item()

        if batch_idx %10 ==0:
            val_loss = 0
            for batch_idx_v,(inputs_v,target_v) in enumerate(valloader):
                inputs_v,targets_v = inputs_v.to(device),target_v.to(device)
                outputs_v = net(inputs_v)
                loss_v = criterion(outputs_v,targets_v)

                val_loss      += loss_v.item()
                _,predicted_v = outputs_v.max(1)
                total_v       = targets_v.size(0)
                correct_v     = predicted_v.eq(targets_v).sum().item()
                
                total_v       = target_v.size(0)
                correct_v     = predicted_v.eq(targets_v).sum().item()

            
            train_loss_dw.append(train_loss/((batch_idx*(epoch+1))+1))
            val_loss_dw.append(val_loss/(batch_idx_v+1))
            train_acc_dw.append((100.*correct/total))
            val_acc_dw.append((100.*correct_v/total_v))

            print("total_ep:%d|cur_ep:%d|step:%d|train_lss:%.3f|train_acc:%.3f|val_lss:%.3f|val_acc:%.3f"%(args.epoch,\
            epoch,batch_idx,train_loss/(batch_idx+1),(100.*correct/total),val_loss/(batch_idx_v+1),(100.*correct_v/total_v)))

        if best_loss>val_loss:
            print("save")
            state = {
                'net': net.state_dict(),
                'epoch':epoch+1,
                'optimizer': optimizer.state_dict(),
                'lr_schedule': lr_scheduler.state_dict()
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state,'./checkpoint/best_ckpt.pth')
            # best_acc = val_acc
            best_loss = val_loss
    lr_scheduler.step()

result_visual(train_loss_dw,val_loss_dw,train_acc_dw,val_acc_dw)





            








