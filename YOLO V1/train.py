from network import Model
from ops import make_loss
from utils import read_batch
import torch


def train():
    model = Model()
    model.to("cuda:0")
    Opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    # checkpoint = torch.load("./model.pth")
    # model.load_state_dict(checkpoint["model"])
    # Opt.load_state_dict(checkpoint["Opt"])
    for i in range(10000):
        Opt.zero_grad()
        imgs, targets = read_batch()
        imgs = torch.tensor(imgs, dtype=torch.float32).to("cuda:0")
        targets = torch.tensor(targets, dtype=torch.float32).to("cuda:0")
        preds = model(imgs)
        loss = make_loss(preds, targets)
        loss.backward()
        Opt.step()
        if i % 10 == 0:
            print("Iteration: %d, Loss: %f"%(i, loss))
        if i % 10 == 0:
            state = {'model':model.state_dict(), 'Opt':Opt.state_dict(), 'itr':i}
            torch.save(state, "./model.pth")


if __name__ == "__main__":
    train()

