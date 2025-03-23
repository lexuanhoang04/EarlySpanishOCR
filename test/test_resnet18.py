import torchvision.models as models
resnet = models.resnet18(pretrained=True)
print(resnet.layer1)
print(resnet.layer2)
print(resnet.layer3)
print(resnet.layer4)
