import torch
import torch.nn as nn
import torch.optim as opt
import scipy.io as sio
import numpy as np
import time

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.global_pooling = nn.AvgPool2d(4, 4)
        self.linear = nn.Linear(512, 4)
        self.layer1 = self._make_layer(1, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 3)
        self.layer4 = self._make_layer(256, 512, 3)

    def _make_layer(self, in_ch, out_ch, nums):
        blocks = []
        blocks.append(Block(in_ch, out_ch))
        for i in range(1, nums):
            blocks.append(Block(out_ch, out_ch))
        blocks.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pooling(x)[:, :, 0, 0]
        x = self.linear(x)
        return x

class VGG16_V2(nn.Module):
    def __init__(self):
        super(VGG16_V2, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.linear1 = nn.Linear(2*2*512, 1024)
        self.linear2 = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, 2*2*512)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def train():
    data = sio.loadmat("./data.mat")
    train_data = torch.tensor(data["train"][:, np.newaxis, :, :]/255.0, dtype=torch.float32)
    train_label = torch.tensor(np.argmax(data["train_label"], axis=1), dtype=torch.long)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_data.to(device)
    train_label.to(device)
    nums = train_data.size(0)
    vgg16 = VGG16_V2()
    vgg16.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(vgg16.parameters(), lr=0.0001)
    for i in range(10000):
        s = time.time()
        rand_idx = np.random.randint(0, nums, [64])
        batch = train_data[rand_idx]
        batch_label = train_label[rand_idx]
        logits = vgg16(batch)
        loss = criterion(logits, batch_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        e = time.time()
        if i % 1 == 0:
            print("Iteration: %d, Loss: %f, Time: %f"%(i, loss, e-s))

train()