import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


nclasses = 250


def get_resnet34():
  resnet34 = models.resnet34(weights='IMAGENET1K_V1')
  #Initially, train only the added classification layer for a few 
  #epochs
  for param in resnet34.parameters():
      param.requires_grad = False
  num_ftrs = resnet34.fc.in_features
  resnet34.fc = nn.Sequential(
    # nn.Linear(num_ftrs, num_ftrs),
    # nn.ReLU(),
    nn.Dropout(0.7),
    nn.Linear(num_ftrs, nclasses)
  )
  return resnet34

def get_resnet18():
  resnet = models.resnet18(weights='IMAGENET1K_V1')
  for param in resnet.parameters():
      param.requires_grad = False
  
  num_ftrs = resnet.fc.in_features
  resnet.fc = nn.Sequential(
    # nn.Linear(num_ftrs, num_ftrs),
    # nn.ReLU(),
    nn.Dropout(0.7),
    nn.Linear(num_ftrs, nclasses)
  )
  return resnet


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
