import torch
import torch.nn.functional as F
from config import S, IMG_W, IMG_H, B, LAMBDA_COR, LAMBDA_NOOBJ, BATCH_SIZE

def cal_ious(preds, targets):
    #preds: B x 7 x 7 x 30, targets: B x 7 x 7 x 24
    size = preds.size()
    targets_bbox = targets[..., :4].view(size[0], size[1], size[2], 1, 4)
    preds_bboxes = preds[..., :4*B].view(size[0], size[1], size[2], B, 4)
    x, y, w, h = targets_bbox[..., 0], targets_bbox[..., 1], targets_bbox[..., 2], targets_bbox[..., 3]
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    x_, y_, w_, h_ = preds_bboxes[..., 0], preds_bboxes[..., 1], preds_bboxes[..., 2], preds_bboxes[..., 3]
    x1_, y1_, x2_, y2_ = x_ - w_ / 2, y_ - h_ / 2, x_ + w_ / 2, y_ + h_ / 2
    inter_x1 = torch.max(x1, x1_)
    inter_x2 = torch.min(x2, x2_)
    inter_y1 = torch.max(y1, y1_)
    inter_y2 = torch.min(y2, y2_)
    inter_area = F.relu(inter_x2 - inter_x1) * F.relu(inter_y2 - inter_y1)
    union_area = w * h + w_ * h_ - inter_area
    ious = inter_area / union_area
    return ious #B x 7 x 7 x 2

def make_loss(preds, targets):
    ious = cal_ious(preds, targets)
    argmax_idx = torch.argmax(ious, dim=-1) #to one-hot vector as mask cross B
    mask = torch.zeros(BATCH_SIZE, 7, 7, B).scatter(-1, argmax_idx.cpu().view(BATCH_SIZE, 7, 7, 1), 1).to("cuda:0")
    xs = []
    ys = []
    ws = []
    hs = []
    for i in range(B):
        xs.append(preds[..., i*4:i*4+1])
        ys.append(preds[..., i*4+1:i*4+2])
        ws.append(preds[..., i*4+2:i*4+3])
        hs.append(preds[..., i*4+3:i*4+4])
    xs = torch.cat(xs, dim=-1)
    ys = torch.cat(ys, dim=-1)
    ws = torch.cat(ws, dim=-1).clamp_min(1e-10)
    hs = torch.cat(hs, dim=-1).clamp_min(1e-10)
    cs = preds[..., 4*B:4*B+B]
    classes = preds[..., 4*B+B:]
    gt_xs = targets[..., 0:1]
    gt_ys = targets[..., 1:2]
    gt_ws = targets[..., 2:3]
    gt_hs = targets[..., 3:4]
    gt_classes = targets[..., 4:]
    gt_mask = torch.sum(targets[..., 4:], dim=-1, keepdim=True)
    loss1 = ((xs - gt_xs)**2 + (ys - gt_ys)**2) * mask * gt_mask
    loss2 = ((ws**0.5 - gt_ws**0.5)**2 + (hs**0.5 - gt_hs**0.5)**2) * mask * gt_mask
    loss3 = (cs - gt_mask)**2 * mask * gt_mask
    loss4 = (cs - gt_mask)**2 * (1 - gt_mask) * mask
    loss5 = (classes - gt_classes)**2 * gt_mask
    loss = torch.sum(loss1) * LAMBDA_COR / gt_mask.sum() + torch.sum(loss2) * LAMBDA_COR / gt_mask.sum() + torch.sum(loss3) / gt_mask.sum() + \
           torch.sum(loss4) * LAMBDA_NOOBJ / (1 - gt_mask).sum() + torch.sum(loss5) / gt_mask.sum()
    if loss.cpu().detach().numpy() < 0:
        a = 0
    return loss




# preds = torch.randn(1, 7, 7, 4*B+B+20)
# targets = torch.randn(1, 7, 7, 24).clamp(0., 1.0)
# ious =  cal_ious(preds, targets)
# argmax_idx = torch.argmax(ious, dim=-1)
# make_loss(preds, targets)
# one_hot = torch.zeros(1, 7, 7, 2).scatter(-1, argmax_idx.view(1, 7, 7, 1), 1)
# a = 0


