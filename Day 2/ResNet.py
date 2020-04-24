import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio


class ResBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, downsample=False):
        super(ResBlock, self).__init__()
        if downsample:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, 2),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, 1, 1),
                nn.BatchNorm2d(out_ch)
            )
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, 2),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, 1),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, 1, 1),
                nn.BatchNorm2d(out_ch)
            )
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, 1),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        temp = self.short_cut(x)
        x = self.block(x)
        x += F.relu(x + temp)
        return x

class ResNet(nn.Module):
    def __init__(self, num_class=10):
        super(ResNet, self).__init__()
        self.preprocess = Preprocess()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 64, 256, 3, False)
        self.layer2 = self._make_layer(256, 128, 512, 4, True)
        self.layer3 = self._make_layer(512, 256, 1024, 6, True)
        self.layer4 = self._make_layer(1024, 512, 2048, 3, True)
        self.avg_pool = nn.AvgPool2d(1, 1)
        self.linear = nn.Linear(2048, num_class)

    def _make_layer(self, in_ch, mid_ch, out_ch, nums_block, downsample=True):
        blocks = []
        blocks.append(ResBlock(in_ch, mid_ch, out_ch, downsample))
        for i in range(1, nums_block):
            blocks.append(ResBlock(out_ch, mid_ch, out_ch, False))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)[:, :, 0, 0]
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
    resnet50 = ResNet(10)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    resnet50.to(device)
    optimizer = torch.optim.Adam(resnet50.parameters(), lr=0.001)
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
    for i in range(100000):
        rand_idx = np.random.randint(0, train_nums, [256])
        batch = train_data[rand_idx]
        batch_labels = train_labels[rand_idx]
        logits = resnet50(batch)
        train_loss = criterion(logits, batch_labels)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_acc = torch._cast_Float(torch.argmax(logits, dim=1) == batch_labels).mean()
        if i % 100 == 0:
            logits = resnet50(val_data)
            val_acc = torch._cast_Float(torch.argmax(logits, dim=1) == val_labels).mean()
            print("Iteration: %d, Loss: %f, Val Accuracy: %f"%(i, train_loss, val_acc))

if __name__ == "__main__":
    train()

