from __future__ import print_function
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import torch

import os
import sys
import numpy as np
import glob
import cv2
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from load_coco import CocoDataset
sys.path.append(os.path.join(os.path.dirname(__file__),'../../utils'))
from detector import RetinanetDetector
from anchors import Anchors
sys.path.append(os.path.join(os.path.dirname(__file__),'../../networks'))
from model import resnet50
sys.path.append(os.path.join(os.path.dirname(__file__),'../../configs'))
from config import cfgs

def parms():
    parser = argparse.ArgumentParser(description='coco test')
    parser.add_argument('--weights', default='',
                        type=str, help='Trained state_dict file path')
    parser.add_argument('--cuda', default=False, type=bool,
                        help='Use cuda in live demo')
    parser.add_argument('--img_path',type=str,default='',help='')
    parser.add_argument('--load_num',type=int,default=0,help='load model num')
    parser.add_argument('--img_dir',type=str,default='',help='')
    parser.add_argument('--save_dir',type=str,default='',help='')
    parser.add_argument('--year',type=str,default='2017',help='')
    return parser.parse_args()

class Retinanet_Eval(object):
    def __init__(self,args):
        self.img_size = cfgs.ImgSize
        self.img_dir = args.img_dir
        self.save_dir = args.save_dir
        self.build_net()
        self.load_model(args.load_num)
        self.laod_anchor()
        self.score_threshold = cfgs.score_threshold
    
    def build_net(self):
        self.Retinanet_model = resnet50(cfgs.ClsNum)
        self.Detector = RetinanetDetector()

    def load_model(self,load_num):
        load_path = "%s/%s_%s.pth" %(cfgs.model_dir,cfgs.ModelPrefix,load_num)
        print('Resuming training, loading {}...'.format(load_path))
        if torch.cuda.is_available():
            self.Retinanet_model.load_state_dict(torch.load(load_path))
            self.Retinanet_model.cuda()
            self.Detector.cuda()
        else: 
            weights = torch.load(load_path,map_location='cpu')
            #weights = self.rename_dict(weights)
            self.Retinanet_model.load_state_dict(weights)
        self.Retinanet_model.eval()

    def rename_dict(self,state_dict):
        state_dict_new = dict()
        for key,value in list(state_dict.items()):
            state_dict_new[key[7:]] = value
        return state_dict_new


    def laod_anchor(self):
        get_anchor = Anchors()
        img_batch = torch.ones((1,3,self.img_size,self.img_size))
        self.anchors = get_anchor(img_batch)
        if torch.cuda.is_available():
            self.anchors = self.anchors.cuda()
    
    def inference(self,input_img):
        '''
        input_img: [batch,c,h,w]
        '''
        pred_cls, pred_bbox,conf_maps = self.Retinanet_model(input_img)
        rectangles = self.Detector(self.anchors, pred_bbox, pred_cls)
        return rectangles,conf_maps

    def re_scale(self,img):
        img_h, img_w = img.shape[:2]
        ratio = max(img_h, img_w) / float(self.img_size)
        new_h = int(img_h / ratio)
        new_w = int(img_w / ratio)
        ox = (self.img_size - new_w) // 2
        oy = (self.img_size - new_h) // 2
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        out = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8) 
        out[oy:oy + new_h, ox:ox + new_w, :] = scaled
        return out.astype(np.float32),[ox,oy,new_w,new_h]

    def de_scale(self,box,window,img_h,img_w):
        ox,oy,new_w,new_h = window
        xmin, ymin, xmax, ymax = box[:,:,:,1],box[:,:,:,2],box[:,:,:,3],box[:,:,:,4]
        box[:,:,:,1] = (xmin - ox) / float(new_w) * img_w
        box[:,:,:,2] = (ymin - oy) / float(new_h) * img_h
        box[:,:,:,3] = (xmax - ox) / float(new_w) * img_w
        box[:,:,:,4] = (ymax - oy) / float(new_h) * img_h
        return box
    def de_scale_org(self,box,window,img_h,img_w):
        ox,oy,new_w,new_h = window
        xmin, ymin, xmax, ymax = box[:,0],box[:,1],box[:,2],box[:,3]
        box[:,0] = (xmin - ox) / float(new_w) * img_w
        box[:,1] = (ymin - oy) / float(new_h) * img_h
        box[:,2] = (xmax - ox) / float(new_w) * img_w
        box[:,3] = (ymax - oy) / float(new_h) * img_h
        return box
    
    def normalize(self,img):
        img = img / 255.0
        img[:,:,0] -= cfgs.PIXEL_MEAN[0]
        img[:,:,0] = img[:,:,0] / cfgs.PIXEL_NORM[0] 
        img[:,:,1] -= cfgs.PIXEL_MEAN[1]
        img[:,:,1] = img[:,:,1] / cfgs.PIXEL_NORM[1]
        img[:,:,2] -= cfgs.PIXEL_MEAN[2]
        img[:,:,2] = img[:,:,2] / cfgs.PIXEL_NORM[2]
        return img.astype(np.float32)

    def decode_bboxes(self,detections):
        '''
        boxes: shape is [batch,cls_num,default_boxes_num,5]
        '''
        bboxes = []
        scores = []
        labels = []
        for i in range(0,detections.shape[1]):
            for j in range(detections.shape[2]):
                if detections[0, i, j, 0] >= self.score_threshold:
                    scores.append(detections[0, i, j, 0])
                    bboxes.append(detections[0, i, j, 1:])
                    labels.append(i)
        return np.array(scores),np.array(labels),np.array(bboxes)

    def test_img(self,frame):
        height, width = frame.shape[:2]
        img_scale, window = self.re_scale(frame.copy())
        img_scale = self.normalize(img_scale)
        img_input = torch.from_numpy(img_scale).permute(2, 0, 1)
        img_input = Variable(img_input.unsqueeze(0))
        if torch.cuda.is_available():
            img_input.cuda()
        t1=time.time()
        rectangles,conf_maps = self.inference(img_input)  # forward pass
        #detections = rectangles.data.cpu().numpy()
        t2=time.time()
        #print('consume:',t2-t1)
        scores,labels,detections = rectangles
        scores = scores.data.numpy()
        labels = labels.data.numpy()
        # scale each detection back up to the image
        #detections = self.de_scale(detections,window,height,width)
        bboxes = self.de_scale_org(detections,window,height,width)
        #scores,labels,bboxes = self.decode_bboxes(detections)
        return scores,labels,bboxes

    def eval_coco(self, year,load_name='test'):
        ObjectNames = cfgs.COCODataNames
        dataset_test = CocoDataset()
        dataset_test.load_coco(self.img_dir, load_name, year=year) #class_names=ObjectNames)
        #make dir for names
        #print("valid names",dataset_test.ValidNames)
        print("labels,",dataset_test.trainId2catId.keys())
        #load image data and bbox
        test_img_ids = dataset_test.img_ids
        test_cat_ids = dataset_test.cat_ids
        print("total imgs",len(test_img_ids))
        results = []
        for tmp_idx in tqdm(range(len(test_img_ids))):
            img,img_id = dataset_test.load_image(tmp_idx)
            #label_show(img,tmp_bboxes,name_list)
            scores, labels, boxes = self.test_img(img)
            # correct boxes for image scale
            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]
                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]
                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : img_id,
                        'category_id' : dataset_test.trainId2catId[label],
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }
                    # append detection to results
                    results.append(image_result)
            # append image to list of processed images
            # print progress
        if not len(results):
            return 0
        # write output
        json.dump(results, open('{}_bbox_results.json'.format(load_name), 'w'), indent=4)
        # load results in COCO evaluation tool
        coco_true = dataset_test.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(load_name))
        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = test_img_ids
        coco_eval.params.catIds = test_cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return 1

if __name__ == '__main__':
    args = parms()
    coco_test = Retinanet_Eval(args)
    fg = coco_test.eval_coco(args.year)