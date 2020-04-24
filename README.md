# Pytorch_Learning_Diary
Learning pytorch on TensorFlow basis, and write down every error I meet.
### Day 1-4: How to construct a CNN [VGG16, ResNet, DenseNet], simple to complex.
###   Day 5: How to use a pre-trained model [Style transfer]
###   Day 6: How to edit the parameters in CNN [GAN, WGAN, WGAN-GPU etc.]
## Day 1 
1. Reading the turorial on the pytorch official website https://pytorch.org/tutorials/
2. According to the tutorial, implementing the basic pytorch code and understanding the back-propagation mechanism of pytorch.
3. Thinking about the differences and similarities between TensorFlow and Pytorch.
4. Coding a simple convolutional neural network: VGG16

## Day 2
1. Pytorch code conventions:

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, downsample=False):
        super(ResBlock, self).__init__()

    def forward(self, x):

        return x

class ResNet(nn.Module):
    def __init__(self, num_class=10):
        super(ResNet, self).__init__()

    def _make_layer(self, x):

        return x

    def forward(self, x):

        return x

```
2. Implementing the CNN: ResNet
## Day 3
1. Implementing the CNN: DenseNet. 

***There are some problems.***
The model cannot use GPU to train. After analysing the code, the reason is that I write some nn.Conv, nn.BatchNorm,etc in function forward
```Python
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, downsample=False):
        super(ResBlock, self).__init__()

    def forward(self, x):
        x = F.conv2d(x) # It cannot allocate the weight to GPU
        return x
```
## Day 4
1. Implementing the CNN: DenseNet in GPU. It's different from Day 3 and successful.

## Day 5
1. In order to know how to use pre-trained model in pytorch, the plan of Day 5 is to implementing the style transfer.
# Some error in learning
1. In order
```Python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```        
2. The weight deday
```Python
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
```
3. Dynamic reduce learning rate
```Python
optimizer = torch.optim.SGD(densenet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
lr_scheduler_opt = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50000, 75000], gamma=0.1)
...
optimizer.step()
lr_scheduler_opt.step()
```
4. Using GPU for training
```Python
net = Net()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
net.to(device)
data.to(device)
```
5. Global pooling
```Python
self.global_pool = nn.AdaptiveAvgPool2d(1)
```
#### To be continued
