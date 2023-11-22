import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

nclasses = 250


resnet34 = models.resnet34(weights='IMAGENET1K_V1')
for param in resnet34.parameters():
     param.requires_grad = False
for param in resnet34.layer4.parameters():
    param.requires_grad = True
for param in resnet34.layer3[-1].parameters():
    param.requires_grad = True
num_ftrs = resnet34.fc.in_features
resnet34.fc = nn.Sequential(
  nn.Dropout(0.1),
  nn.Linear(num_ftrs, nclasses)
)

resnet = models.resnet18(weights='IMAGENET1K_V1')
for param in resnet.parameters():
    param.requires_grad = False
for param in resnet.layer4.parameters():
    param.requires_grad = True
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Sequential(
  nn.Dropout(0.1),
  nn.Linear(num_ftrs, nclasses)

)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
