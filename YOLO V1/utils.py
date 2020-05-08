import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import os
from config import CLASSES, IMG_H, IMG_W, XML_PATH, IMG_PATH, BATCH_SIZE, S


def read_data(xml_path, img_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall("object")
    names = []
    gtbboxes = np.zeros([len(objects), 4], dtype=np.int32)
    for idx, obj in enumerate(objects):
        names.append(obj.find("name").text)
        xmin = int(obj.find("bndbox").find("xmin").text)
        xmax = int(obj.find("bndbox").find("xmax").text)
        ymin = int(obj.find("bndbox").find("ymin").text)
        ymax = int(obj.find("bndbox").find("ymax").text)
        gtbboxes[idx, 0] = (xmin + xmax)//2
        gtbboxes[idx, 1] = (ymin + ymax)//2
        gtbboxes[idx, 2] = xmax - xmin
        gtbboxes[idx, 3] = ymax - ymin
    img = np.array(Image.open(img_path))
    labels = np.zeros([len(objects)])
    for idx, name in enumerate(names):
        labels[idx] = CLASSES.index(name)
    return img, gtbboxes, labels


def normalize_img(img, bboxes):
    img_h, img_w = img.shape[0], img.shape[1]
    x, y, w, h = bboxes[:, 0:1], bboxes[:, 1:2], bboxes[:, 2:3], bboxes[:, 3:4]
    y_ = y * IMG_H / img_h
    x_ = x * IMG_W / img_w
    w_ = w * IMG_W / img_w
    h_ = h * IMG_H / img_h
    img = np.array(Image.fromarray(np.uint8(img)).resize([IMG_W, IMG_H]))
    bboxes = np.concatenate((x_, y_, w_, h_), axis=1)
    return img, bboxes


def draw_bboxes(img, bboxes):
    nums = bboxes.shape[0]
    for i in range(nums):
        x, y, w, h = bboxes[i, 0], bboxes[i, 1], bboxes[i, 2], bboxes[i, 3]
        x1, y1, x2, y2 = np.maximum(int(x - w//2), 0), np.maximum(int(y - h//2), 0), np.minimum(int(x + w//2), IMG_W-1), np.minimum(int(y + h//2), IMG_H-1)
        green_line = np.zeros_like(img[y1:y2, x1])
        green_line[:, 1] = np.ones_like(green_line[:, 1]) * 255
        img[y1:y2, x1] = green_line
        green_line = np.zeros_like(img[y1:y2, x1])
        green_line[:, 1] = np.ones_like(green_line[:, 1]) * 255
        img[y1:y2, x2] = green_line

        green_line = np.zeros_like(img[y1, x1:x2])
        green_line[:, 1] = np.ones_like(green_line[:, 1]) * 255
        img[y1, x1:x2] = green_line
        green_line = np.zeros_like(img[y2, x1:x2])
        green_line[:, 1] = np.ones_like(green_line[:, 1]) * 255
        img[y2, x1:x2] = green_line
    return img


def read_batch():
    filenames = os.listdir(XML_PATH)
    imgs = np.zeros([BATCH_SIZE, 3, IMG_H, IMG_W])
    targets = np.zeros([BATCH_SIZE, S, S, 4 + len(CLASSES)])
    for i in range(BATCH_SIZE):
        filename = np.random.choice(filenames, 1)[0]
        img, bbox, label = read_data(XML_PATH+filename, IMG_PATH+filename[:-3]+"jpg")
        img, bbox = normalize_img(img, bbox)
        img = np.transpose(img, axes=[2, 0, 1])
        imgs[i] = img
        for j in range(bbox.shape[0]):#bbox: n x 4
            cell_h, cell_w = IMG_H / S, IMG_W / S
            x_idx, y_idx = int(bbox[j][0] // cell_w), int(bbox[j][1] // cell_h)
            targets[i, y_idx, x_idx, 0] = bbox[j, 0] / cell_w - x_idx
            targets[i, y_idx, x_idx, 1] = bbox[j, 1] / cell_h - y_idx
            targets[i, y_idx, x_idx, 2] = bbox[j, 2] / IMG_W
            targets[i, y_idx, x_idx, 3] = bbox[j, 3] / IMG_H
            label_vec = np.zeros([20])
            label_vec[int(label[j])] = 1
            targets[i, y_idx, x_idx, 4:] = label_vec

    return imgs/255, targets

# import torchvision.transforms as transforms
# import torch
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# read_batch()
# imgs, batch_idxes, targets, clses = read_batch(anchors)
# img = torch.tensor(imgs[0]/255)
# pos_idx = np.where(clses[0]==1)[0]
# bbox = transform_reverse(anchors[np.int32(batch_idxes[0][pos_idx])], targets[0][pos_idx])
# imgs = np.transpose(imgs, axes=[0, 2, 3, 1])
# Image.fromarray(np.uint8(draw_bboxes(imgs[0], bbox))).show()
# img, bbox, label = read_data("G:/迅雷下载/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/000005.xml",
#                              "G:/迅雷下载/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg")
# img, bbox = normalize_img(img, bbox)
# anchors = get_anchors()
# tragets, clses, maskes = read_batch(anchors)
# pass
# target, cls, mask = generate_batch(anchors, bbox)
# bbox = transform_reverse(anchors[posidx], target[np.where(cls==1)[0]])
# iou = cal_iou(anchors, bbox)
# Image.fromarray(np.uint8(draw_bboxes(img, anchors[10000:10010]))).show()
# pass