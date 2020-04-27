import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from PIL import Image

batchsize = 128


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(128, 4 * 4 * 1024),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 1024, 4, 4)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )
        self.linear = nn.Linear(4 * 4 * 512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 4 * 4 * 512)
        x = self.linear(x)
        return x

def weight_clip(model, min_, max_):
    params = list(model.parameters())

def DCGAN():
    generator = Generator()
    discriminator = Discriminator()
    generator.to("cuda:0")
    discriminator.to("cuda:0")
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    data = sio.loadmat("./facedata.mat")["data"]
    data = np.transpose(data, axes=[0, 3, 1, 2])
    nums = data.shape[0]
    for i in range(100000):
        z = torch.randn(batchsize, 128).to("cuda:0")
        fake_img = generator(z)
        fake_logits = discriminator(fake_img)
        g_loss = -torch.mean(torch.log(torch.sigmoid(fake_logits) + 1e-10))
        for j in range(1):
            rand_idx = np.random.randint(0, nums, [batchsize])
            batch = data[rand_idx] / 127.5 - 1.0
            batch = torch.tensor(batch, dtype=torch.float32).to("cuda:0")
            real_logits = discriminator(batch)
            d_loss = -torch.mean(torch.log(torch.sigmoid(real_logits)+1e-10) + torch.log(1 - torch.sigmoid(fake_logits)+1e-10))
            opt_d.zero_grad()
            d_loss.backward(retain_graph=True)
            opt_d.step()
        opt_g.zero_grad()
        g_loss.backward(retain_graph=True)
        opt_g.step()
        if i % 100 == 0:
          img = np.uint8(np.transpose((fake_img.cpu().detach().numpy()[0]+1)*127.5, axes=[1, 2, 0]))
          Image.fromarray(img).save("./results/"+str(i)+".jpg")
          print("Iteration: %d, D_loss: %f, G_loss: %f"%(i, d_loss, g_loss))

def WGAN():
    generator = Generator()
    discriminator = Discriminator()
    generator.to("cuda:0")
    discriminator.to("cuda:0")
    opt_g = torch.optim.RMSprop(generator.parameters(), lr=5e-5)
    opt_d = torch.optim.RMSprop(discriminator.parameters(), lr=5e-5)
    data = sio.loadmat("./facedata.mat")["data"]
    data = np.transpose(data, axes=[0, 3, 1, 2])
    nums = data.shape[0]
    for i in range(100000):
        z = torch.randn(batchsize, 128).to("cuda:0")
        fake_img = generator(z)
        fake_logits = discriminator(fake_img)
        g_loss = -torch.mean(fake_logits)
        for j in range(1):
            rand_idx = np.random.randint(0, nums, [batchsize])
            batch = data[rand_idx] / 127.5 - 1.0
            batch = torch.tensor(batch, dtype=torch.float32).to("cuda:0")
            real_logits = discriminator(batch)
            d_loss = -torch.mean(real_logits) + torch.mean(fake_logits)
            for para in discriminator.parameters():
                para.data.clamp_(-0.01, 0.01)
            opt_d.zero_grad()
            d_loss.backward(retain_graph=True)
            opt_d.step()
        opt_g.zero_grad()
        g_loss.backward(retain_graph=True)
        opt_g.step()
        if i % 100 == 0:
          img = np.uint8(np.transpose((fake_img.cpu().detach().numpy()[0]+1)*127.5, axes=[1, 2, 0]))
          Image.fromarray(img).save("./results/"+str(i)+".jpg")
          print("Iteration: %d, D_loss: %f, G_loss: %f"%(i, d_loss, g_loss))

if __name__ == "__main__":
    WGAN()