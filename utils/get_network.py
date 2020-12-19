from models import * 

def get(network):
    if network =='resnet18':
        return resnet18()
    elif network == 'resnet34':
        return resnet34()
    elif network == 'resnet50':
        return resnet50()
    elif network == 'resnet101':
        return resnet101()
    elif network == 'resnet152':
        return resnet152()
    elif network == 'vgg16':
        return vgg16()
    




