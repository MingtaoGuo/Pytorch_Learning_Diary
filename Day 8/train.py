from networks import Generator, Discriminator
import torch
import torch.optim as optim
import scipy.io as sio
import numpy as np
from PIL import Image

batchsize = 32
n_cri = 1
lr = 2e-4

def train():
    generator = Generator()
    discriminator = Discriminator()
    generator.to("cuda:0")
    discriminator.to("cuda:0")
    Opt_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    Opt_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    data = np.concatenate((sio.loadmat("D:/cifar10/data_batch_1.mat")["data"], sio.loadmat("D:/cifar10/data_batch_2.mat")["data"],
                           sio.loadmat("D:/cifar10/data_batch_3.mat")["data"], sio.loadmat("D:/cifar10/data_batch_4.mat")["data"],
                           sio.loadmat("D:/cifar10/data_batch_5.mat")["data"]))
    nums = data.shape[0]

    for i in range(100000):
        rand_idx = np.random.choice(range(nums), batchsize)
        batch = np.reshape(data[rand_idx], [batchsize, 3, 32, 32])
        batch = torch.tensor(batch/127.5 - 1, dtype=torch.float32).to("cuda:0")
        for j in range(n_cri):
            z = torch.randn(batchsize, 128).to("cuda:0")
            fake_img = generator(z).detach()
            fake_logits = discriminator(fake_img)
            real_logits = discriminator(batch)
            D_loss = torch.mean(torch.max(torch.zeros_like(real_logits), 1. - real_logits)) + torch.mean(torch.max(torch.zeros_like(fake_logits), 1. + fake_logits))
            Opt_D.zero_grad()
            D_loss.backward()
            Opt_D.step()
        z = torch.randn(batchsize, 128).to("cuda:0")
        fake_img = generator(z)
        fake_logits = discriminator(fake_img)
        G_loss = -torch.mean(fake_logits)
        Opt_G.zero_grad()
        G_loss.backward()
        Opt_G.step()
        if i % 100 == 0:
            img = (fake_img[0] + 1) * 127.5
            Image.fromarray(np.uint8(np.transpose(img.cpu().detach().numpy(), axes=[1, 2, 0]))).save("./results/"+str(i)+".jpg")
            print("Iteration: %d, D_loss: %f, G_loss: %f"%(i, D_loss, G_loss))
        if i % 1000 == 0:
            torch.save(generator, "generator.pth")
            torch.save(discriminator, "discriminator.pth")



if __name__ == "__main__":
    train()
