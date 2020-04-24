import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio


class DenseBlock(nn.Module):
    def __init__(self, in_ch, k, l):
        super(DenseBlock, self).__init__()
        self.H = []
        for i in range(l):
            ch = in_ch + i * k
            H = nn.Sequential(
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, k, 1, 1),
                nn.BatchNorm2d(k),
                nn.ReLU(inplace=True),
                nn.Conv2d(k, k, 3, 1, 1)
            )
            self.H.append(H)


    def forward(self, x):
        temp = []
        temp.append(x)
        for H in self.H:
            x = H(x)
            temp.append(x)
            x = torch.cat(temp, dim=1)
        return x

class Transition_Layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Transition_Layer, self).__init__()
        self.trans = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, 1),
            nn.AvgPool2d(2, 2)
        )

    def forward(self, x):
        return self.trans(x)

class DenseNet(nn.Module):
    def __init__(self, k=24, layer=[6, 12, 24, 16], cls_num=10):
        super(DenseNet, self).__init__()
        self.preprocess = Preprocess()
        self.conv1 = nn.Sequential(nn.Conv2d(3, k, 7, 2, 3),
                                   nn.BatchNorm2d(k),
                                   nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.DenseBlock1 = DenseBlock(k, k, l=layer[0])
        self.TransLayer1 = Transition_Layer((layer[0]+1)*k, k)
        self.DenseBlock2 = DenseBlock(k, k, l=layer[1])
        self.TransLayer2 = Transition_Layer((layer[1]+1)*k, k)
        self.DenseBlock3 = DenseBlock(k, k, l=layer[2])
        self.TransLayer3 = Transition_Layer((layer[2]+1)*k, k)
        self.DenseBlock4 = DenseBlock(k, k, l=layer[3])
        self.global_pool = nn.AvgPool2d(1, 1)
        self.linear = nn.Linear((layer[3] + 1)*k, cls_num)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.DenseBlock1(x)
        x = self.TransLayer1(x)
        x = self.DenseBlock2(x)
        x = self.TransLayer2(x)
        x = self.DenseBlock3(x)
        x = self.TransLayer3(x)
        x = self.DenseBlock4(x)
        x = self.global_pool(x)[:, :, 0, 0]
        x = self.linear(x)
        return x

class Preprocess(nn.Module):
    def __init__(self):
        super(Preprocess, self).__init__()

    def forward(self, x):
        var, mean = torch.var_mean(x, dim=[2, 3], keepdim=True)
        x = (x - mean) / torch.sqrt(var)
        return x

def train():
    weight_decay = 1e-4
    densenet = DenseNet()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    densenet.to(device)
    optimizer = torch.optim.SGD(densenet.parameters(), lr=0.1, momentum=0.9)
    lr_scheduler_opt = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50000, 75000], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    L2_reg = nn.MSELoss()
    data1 = sio.loadmat("./cifar-10-batches-mat/data_batch_1.mat")
    data2 = sio.loadmat("./cifar-10-batches-mat/data_batch_2.mat")
    data3 = sio.loadmat("./cifar-10-batches-mat/data_batch_3.mat")
    data4 = sio.loadmat("./cifar-10-batches-mat/data_batch_4.mat")
    data5 = sio.loadmat("./cifar-10-batches-mat/data_batch_5.mat")
    data = np.concatenate((data1["data"], data2["data"], data3["data"], data4["data"], data5["data"]), axis=0)
    labels = np.concatenate((data1["labels"], data2["labels"], data3["labels"], data4["labels"], data5["labels"]), axis=0)
    data = np.reshape(data, [-1, 3, 32, 32])
    labels = labels[:, 0]
    nums = data.shape[0]
    train_nums = 49500
    train_data = data[:train_nums]
    train_labels = labels[:train_nums]
    val_data = data[train_nums:]
    val_labels = labels[train_nums:]
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    val_data = torch.tensor(val_data, dtype=torch.float32).to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.long).to(device)
    paras = list(densenet.parameters())
    for i in range(100000):
        rand_idx = np.random.randint(0, train_nums, [64])
        batch = train_data[rand_idx]
        batch_label = train_labels[rand_idx]
        logits = densenet(batch)
        reg = torch.mean(torch.tensor([L2_reg(p, torch.zeros_like(p)) for p in paras]))
        loss = criterion(logits, batch_label) + reg * weight_decay
        optimizer.zero_grad()
        loss.backward()
        lr_scheduler_opt.step()
        if i % 100 == 0:
            logits = densenet(val_data)
            val_loss = criterion(logits, val_labels)
            val_acc = torch._cast_Float(torch.argmax(logits, dim=1) == val_labels).mean()
            print("Iteration: %d, Val_loss: %f, Val_acc: %f"%(i, val_loss, val_acc))
    pass

train()