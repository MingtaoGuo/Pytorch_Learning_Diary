import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio

class Block(nn.Module):
    def __init__(self, in_ch, k):
        super(Block, self).__init__()
        self.H = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, k, 1, 1),
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True),
            nn.Conv2d(k, k, 3, 1, 1)
        )

    def forward(self, x):
        return torch.cat([x, self.H(x)], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_ch, k, nums_block):
        super(DenseBlock, self).__init__()
        blocks = []
        for i in range(nums_block):
            blocks.append(Block(in_ch, k))
            in_ch += k
        self.denseblock = nn.Sequential(*blocks)

    def forward(self, x):
        return self.denseblock(x)

class Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Transition, self).__init__()
        self.trans = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, 1),
            nn.AvgPool2d(2, 2)
        )

    def forward(self, x):
        return self.trans(x)

class DenseNet(nn.Module):
    def __init__(self, k=24, theta=0.5, layers=[6, 12, 24, 16], cls_num=10):
        super(DenseNet, self).__init__()
        out_ch = k*2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(3, 2, 1)
        in_ch = k*2
        out_ch = in_ch + layers[0] * k
        self.denseblock1 = DenseBlock(in_ch=in_ch, k=k, nums_block=layers[0])
        self.transition1 = Transition(in_ch=out_ch, out_ch=int(theta*out_ch))
        in_ch = int(theta*out_ch)
        out_ch = in_ch + layers[1] * k
        self.denseblock2 = DenseBlock(in_ch=in_ch, k=k, nums_block=layers[1])
        self.transition2 = Transition(in_ch=out_ch, out_ch=int(theta*out_ch))
        in_ch = int(theta*out_ch)
        out_ch = in_ch + layers[2] * k
        self.denseblock3 = DenseBlock(in_ch=in_ch, k=k, nums_block=layers[2])
        self.transition3 = Transition(in_ch=out_ch, out_ch=int(theta*out_ch))
        in_ch = int(theta*out_ch)
        self.denseblock4 = DenseBlock(in_ch=in_ch, k=k, nums_block=layers[3])
        out_ch = in_ch + layers[3] * k
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(out_ch, cls_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.denseblock1(x)
        x = self.transition1(x)
        x = self.denseblock2(x)
        x = self.transition2(x)
        x = self.denseblock3(x)
        x = self.transition3(x)
        x = self.denseblock4(x)
        x = self.global_pool(x)[:, :, 0, 0]
        x = self.linear(x)
        return x

def train():
    densenet = DenseNet()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    densenet.to(device)
    optimizer = torch.optim.SGD(densenet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_scheduler_opt = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50000, 75000], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    data1 = sio.loadmat("./cifar-10-batches-mat/data_batch_1.mat")
    data2 = sio.loadmat("./cifar-10-batches-mat/data_batch_2.mat")
    data3 = sio.loadmat("./cifar-10-batches-mat/data_batch_3.mat")
    data4 = sio.loadmat("./cifar-10-batches-mat/data_batch_4.mat")
    data5 = sio.loadmat("./cifar-10-batches-mat/data_batch_5.mat")
    data = np.concatenate((data1["data"], data2["data"], data3["data"], data4["data"], data5["data"]), axis=0)
    labels = np.concatenate((data1["labels"], data2["labels"], data3["labels"], data4["labels"], data5["labels"]), axis=0)
    data = np.reshape(data, [-1, 3, 32, 32])
    labels = labels[:, 0]
    train_nums = 49500
    train_data = data[:train_nums]
    train_labels = labels[:train_nums]
    val_data = data[train_nums:]
    val_labels = labels[train_nums:]
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    val_data = torch.tensor(val_data, dtype=torch.float32).to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.long).to(device)
    for i in range(100000):
        rand_idx = np.random.randint(0, train_nums, [64])
        batch = train_data[rand_idx]
        batch_label = train_labels[rand_idx]
        logits = densenet(batch)
        loss = criterion(logits, batch_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler_opt.step()
        if i % 100 == 0:
            logits = densenet(val_data)
            val_loss = criterion(logits, val_labels)
            val_acc = torch._cast_Float(torch.argmax(logits, dim=1) == val_labels).mean()
            print("Iteration: %d, Val_loss: %f, Val_acc: %f"%(i, val_loss, val_acc))
    pass

if __name__ == "__main__":
    train()