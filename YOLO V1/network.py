import torchvision as tv
import torch.nn as nn
import torch
from config import B, CLASSES

CLS_NUM = len(CLASSES)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        model = tv.models.resnet50(pretrained=True)
        self.fixed_layers = nn.Sequential(*list(model.children())[:5])
        for p in self.fixed_layers.parameters():
            p.requires_grad = False
        self.trained_layers = nn.Sequential(*list(model.children())[5:8])
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, (4*B+B+CLS_NUM)*7*7)

    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to("cuda:0")
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to("cuda:0")
        x = (x - mean) / std
        x = self.fixed_layers(x).detach()
        x = self.trained_layers(x)
        x = self.avgpooling(x)
        x = x.view(-1, 2048)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.view(-1, 7, 7, 4*B+B+CLS_NUM)

