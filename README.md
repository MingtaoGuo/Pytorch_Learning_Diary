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
1. Implementing the CNN: DenseNet in GPU. It's different from Day 3 and successful. The accurancy of validation set is 0.88 (cifar10).

## Day 5
1. In order to know how to use pre-trained model in pytorch, the plan of Day 5 is to implementing the style transfer.

## Day 6
1. Inplementing the spectral normalization GAN, it can help us to understand the module and parameters.
## Day 7
1. Inplementing the YOLO V1
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
6. Pre-trained model DIY
```Python
vgg19 = models.vgg19(pretrained=True)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net_1_1 = nn.Sequential(*list(list(vgg19.children())[0])[:1])
        self.net_2_1 = nn.Sequential(*list(list(vgg19.children())[0])[1:6])
        self.net_3_3 = nn.Sequential(*list(list(vgg19.children())[0])[6:15])
        self.net_4_3 = nn.Sequential(*list(list(vgg19.children())[0])[15:24])
        self.net_5_3 = nn.Sequential(*list(list(vgg19.children())[0])[24:33])

    def forward(self, x):
        f_1_1 = self.net_1_1(x)
        f_2_1 = self.net_2_1(f_1_1)
        f_3_3 = self.net_3_3(f_2_1)
        f_4_3 = self.net_4_3(f_3_3)
        f_5_3 = self.net_5_3(f_4_3)
        return [f_1_1, f_2_1, f_3_3, f_4_3, f_5_3]
```
7. Parameters with grad or without grad
```Python
vgg19 = models.vgg19(pretrained=True)
#the model parameters
for p in list(vgg19.parameters()):
    p.requires_grad_(False)
#the single parameters
x = torch.randn_like(p)#tensor(content, dtype=torch.float32)#fusion
x.requires_grad = True
```
8. Model initialization
```Python
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant(m.weight, 1)
        nn.init.constant(m.bias, 0)
```
9. Weights cliping
```Python
for para in discriminator.parameters():
    para.data.clamp_(-0.01, 0.01)
```
10. Pre-trained model data preprocess
```Python
import torchvision.transforms as transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
norm = normalize(img)#img: 3x224x224, range[0, 1], dtype: torch tensor
```
11. Runtime out of memery, use detach() to handle this problem
Because in loop, pytorch will accumulate graph to use memory again and again
```Python
class SpectralNorm(nn.Module):
    def __init__(self, out_ch, iter=1):
        super(SpectralNorm, self).__init__()
        self.iter = iter
        self.u = nn.Parameter(torch.randn(out_ch, 1), requires_grad=False).to("cuda:0")
        self.out_ch = out_ch

    def forward(self, W):
        #W: out x in, u: out x 1
        for i in range(self.iter):
            v = l2_norm(torch.matmul(W.transpose(1, 0), self.u))
            self.u = l2_norm(torch.matmul(W, v)).detach() # If remove the detach() operation, in training , show "out of memory" 
        sigma = torch.matmul(torch.matmul(self.u.transpose(1, 0), W), v).detach()
        W = W / sigma.clamp_min(1e-10)
        # del sigma, v
        return W
```
12. Save model
```Python
#Load model
checkpoint = torch.load("./model.pth")
model.load_state_dict(checkpoint["model"])
Opt.load_state_dict(checkpoint["Opt"])
#Save model
state = {'model':model.state_dict(), 'Opt':Opt.state_dict(), 'itr':i}
torch.save(state, "./model.pth")
```
13. Pretrained model
```Python
model = torchvision.models.resnet50(pretrained=True)
model.eval()#This is important!!!!Because of the batch normalization
```
#### To be continued
