from network import Model
from config import CLASSES, S, IMG_H, IMG_W, B
from utils import Image, np, draw_bboxes
import torch


def test(path):
    model = Model()
    model.to("cuda:0")
    model.eval()
    checkpoint = torch.load("./model.pth")
    model.load_state_dict(checkpoint["model"])
    img = np.array(Image.open(path).resize([448, 448]))[np.newaxis]
    img = np.transpose(img, axes=[0, 3, 1, 2]) / 255
    img = torch.tensor(img, dtype=torch.float32).to("cuda:0")
    preds = model(img).cpu().detach().numpy()
    cell_h, cell_w = IMG_H / S, IMG_W / S
    x, y = np.meshgrid(range(S), range(S))
    preds_xywhs = []
    for i in range(B):
        preds_x = (preds[0, :, :, i*4] + x) * cell_w
        preds_y = (preds[0, :, :, i*4+1] + y) * cell_h
        preds_w = preds[0, :, :, i*4+2] * IMG_W
        preds_h = preds[0, :, :, i*4+3] * IMG_H
        preds_xywh = np.dstack((preds_x, preds_y, preds_w, preds_h))
        preds_xywhs.append(preds_xywh)
    preds_xywhs = np.dstack(preds_xywhs)
    preds_xywhs = np.reshape(preds_xywhs, [-1, 4])
    preds_class = preds[0, :, :, 10:]
    preds_class = np.reshape(preds_class, [-1, 20])
    preds_c = preds[0, :, :, 8:10]
    preds_c = np.reshape(preds_c, [-1, 1])
    max_arg = np.argmax(preds_c, axis=0)
    print("max confidence: %f"%(preds_c[max_arg]))
    max_arg_ = np.argmax(preds_class[int(max_arg//2)])
    print("class confidence: %f"%(preds_class[max_arg//2, max_arg_]))
    print("class category: %s"%(CLASSES[int(max_arg_)]))
    Image.fromarray(np.uint8(draw_bboxes(np.array(Image.open(path).resize([448, 448])), preds_xywhs[max_arg[0]:max_arg[0]+1]))).show()



if __name__ == "__main__":
    path = "./cat3.jpg"
    test(path)

