from __future__ import print_function, absolute_import

from . import model_main
from . import utils
import time as time
import torch
import torch.nn as nn
from PIL import Image
import cv2
import numpy as np
from .utils import bbox_iou,non_max_suppression
from .yolo_loss import YOLOLoss

yolo_label_names = (
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'sofa',
    'pottedplant',
    'bed',
    'diningtable',
    'toilet',
    'tvmonitor',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush')



class Detector:

    def __init__(self,filename='./model/official_yolov3_weights_pytorch.pth'):
        seed = 1
        self.cuda_use=False
        self.size=416
        self.device_run = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config = {"model_params": {"backbone_name": "darknet_53",
                                   "backbone_pretrained":""},
                  "yolo": {
                             "anchors": [[[116, 90], [156, 198], [373, 326]],
                                        [[30, 61], [62, 45], [59, 119]],
                                        [[10, 13], [16, 30], [33, 23]]],
                             "classes": 80,
                         },
                "parallels": [0],
                "confidence_threshold":0.5,
                "pretrain_snapshot": filename
                 }
        self.model=model_main.ModelMain(self.config,is_training=False)
        self.model.train(False)

        # Включаем распаралеливание
        self.model = nn.DataParallel(self.model).to(self.device_run)

        state_dict = torch.load(filename,map_location=self.device_run)
        self.model.load_state_dict(state_dict)
        
        self.yolo_losses = []
        for i in range(3):
            self.yolo_losses.append(YOLOLoss(self.config["yolo"]["anchors"][i],
                                    self.config["yolo"]["classes"], (self.size, self.size)))


    def Detect(self,image):
        curTime=time.time() 
        #На вход изобрважение OpenCV
        #config["yolo"]["classes"]

        class_id=[]
        bboxs=[]
        scores=[]
        ori_h, ori_w,_ = image.shape
        pre_h, pre_w = self.size, self.size

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size),
                               interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)

        images = np.asarray([image])
        images = torch.from_numpy(images).to(self.device_run)


        with torch.no_grad():
            outputs = self.model(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, self.config["yolo"]["classes"],
                                                   conf_thres=self.config["confidence_threshold"],
                                                   nms_thres=0.45)

        for idx, detections in enumerate(batch_detections):
            if detections is not None:
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    # Rescale coordinates to original dimensions
                    y1 = (y1 / pre_h) * ori_h
                    x1 = (x1 / pre_w) * ori_w
                    y2 = (y2 / pre_h) * ori_h
                    x2 = (x2 / pre_w) * ori_w
                    
                    bboxs.append((x1,y1,x2,y2))
                    class_id.append(int(cls_pred))
                    scores.append(float(cls_conf))
                    # Add the bbox to the plot
        elapsed=(time.time()-curTime)*1000
        #return class_IDs, scores, bounding_boxs,elapsed
        return class_id,scores,bboxs,elapsed

    # def extract_features(self, model, image):
    #     # Открываем картинку и конвертим в BMP
    #     img = image.convert('RGB')
    #     # Применяем трасформацию и получаем тензор
    #     img_tensor = self.image_transformer(img).float()
    #     # Расширяем тензор для обработки в моделе [128]->[1,128]
    #     img_tensor = img_tensor.unsqueeze_(0)
    #     # Следующая строка не понятно что делает, вероятно для совместимости с предыдущими версиями
    #     #inputs = Variable(img_tensor)
    #     # Получить фичи
    #     outputs = model(img_tensor)
    #     outputs = outputs.data.cpu()
    #     return outputs







