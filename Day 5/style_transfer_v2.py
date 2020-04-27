import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


vgg19 = models.vgg19(pretrained=True)
for p in list(vgg19.parameters()):
    p.requires_grad_(False)

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

def projection(x):
    min_ = np.min(x)
    max_ = np.max(x)
    return 255 * x / (max_ - min_) - 255 * min_ / (max_ - min_)

def preprocess(image_name, image_size):
    image = Image.open(image_name).convert('RGB')
    if type(image_size) is not tuple:
        image_size = tuple([int((float(image_size) / max(image.size))*x) for x in (image.height, image.width)])
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1])])
    tensor = Normalize(rgb2bgr(Loader(image) * 256)).unsqueeze(0)
    return tensor


def train():
    vgg = Net()
    vgg.to("cuda:0")
    w_l = [0.2, 0.2, 0.2, 0.2, 0.2]
    alpha = 1e-3
    beta = 1.0
    # content = torch.tensor(np.transpose(np.array(Image.open("./1.jpg")), axes=[2, 0, 1])[np.newaxis], dtype=torch.float32)
    # style = torch.tensor(np.transpose(np.array(Image.open("./2.jpg")), axes=[2, 0, 1])[np.newaxis], dtype=torch.float32)
    content = preprocess("./1.jpg", 300)
    style = preprocess("./2.jpg", 300)
    a = style.to("cuda:0")#style
    p = content.to("cuda:0")#content
    x = torch.randn_like(p)#fusion
    x = x.to("cuda:0")
    x.requires_grad = True
    optimizer = torch.optim.LBFGS([x])
    for i in range(301):
        def closure():
            feat_a = vgg(a)
            feat_p = vgg(p)
            feat_x = vgg(x)
            L_content = 0.5 * torch.sum((feat_p[3] - feat_x[3]) ** 2)
            L_style = 0
            for w, f_a, f_x in zip(w_l, feat_a, feat_x):
                shape = f_a.size()
                C, H, W = shape[1], shape[2], shape[3]
                f_a = f_a.view(1, C, H * W)
                a_T = f_a.permute(0, 2, 1)
                a_Gram_mat = torch.matmul(f_a, a_T)
                shape = f_x.size()
                C, H, W = shape[1], shape[2], shape[3]
                f_x = f_x.view(1, C, H * W)
                x_T = f_x.permute(0, 2, 1)
                f_Gram_mat = torch.matmul(f_x, x_T)
                L_style += w * torch.sum((a_Gram_mat - f_Gram_mat) ** 2) / (4 * C**2 * H**2 * W**2)
            loss = alpha * L_content + beta * L_style
            optimizer.zero_grad()
            loss.backward()
            return loss
        optimizer.step(closure)
        loss = closure()
        if i % 100 == 0:
            print("Iteration: %d, Loss: %e"%(i, loss))
            img = np.transpose(x[0].cpu().detach().numpy(), [1, 2, 0])
            new_img = np.zeros_like(img)
            new_img[:, :, 0] = img[:, :, 2]
            new_img[:, :, 1] = img[:, :, 1]
            new_img[:, :, 2] = img[:, :, 0]
            img = projection(new_img)
            Image.fromarray(np.uint8(img)).save("./"+str(i)+".jpg")

if __name__ == "__main__":
    train()
