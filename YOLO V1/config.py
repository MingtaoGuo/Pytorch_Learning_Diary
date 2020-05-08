CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
XML_PATH = "D:/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/"
IMG_PATH = "D:/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/"

BATCH_SIZE = 16
IMG_H = 448
IMG_W = 448

S = 7
B = 2
LAMBDA_COR = 5
LAMBDA_NOOBJ = 0.5