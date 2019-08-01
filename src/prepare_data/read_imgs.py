#**********************************
#author: lxy
#time: 14:30 2019.7.1
#tool: python3
#version: 0.1
#project: pedestrian detection
#modify:
#####################################################
import os
import sys
import torch
import cv2
import numpy as np
import random
import torch.utils.data as u_data
#from convert_to_pickle import label_show
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs


class ReadDataset(u_data.Dataset): #data.Dataset
    """
    VOC Detection Dataset Object
    """
    def __init__(self):
        self.voc_file = cfgs.voc_file
        self.coco_file = cfgs.coco_file
        self.img_size = cfgs.ImgSize
        self.voc_dir = cfgs.voc_dir
        self.coco_dir = cfgs.coco_dir
        self.ids = []
        self.annotations = []
        self.load_txt()
        self.idx = 0
        self.total_num = self.__len__()
        self.shulf_num = list(range(self.total_num))
        randm.shuffle(self.shulf_num)

    def __getitem__(self, index):
        im, gt,h,w,window = self.pull_item(index)
        return im, gt,h,w,window

    def __len__(self):
        return len(self.annotations)

    def load_txt(self):
        self.voc_r = open(self.voc_file,'r')
        #self.coco_r = open(self.coco_file,'r')
        voc_annotations = self.voc_r.readlines()
        #coco_annotations = self.coco_r.readlines()
        for tmp in voc_annotations:
            tmp_splits = tmp.strip().split(',')
            img_path = os.path.join(self.voc_dir,tmp_splits[0])
            self.ids.append((self.voc_dir,tmp_splits[0].split('/')[-1][:-4]))
            bbox = map(float, tmp_splits[1:])
            if not isinstance(bbox,list):
                bbox = list(bbox)
            bbox.insert(0,img_path)
            self.annotations.append(bbox)
        '''
        for tmp in coco_annotations:
            tmp_splits = tmp.strip().split(',')
            img_path = os.path.join(self.coco_dir,tmp_splits[0])
            bbox = map(float, tmp_splits[1:])
            if not isinstance(bbox,list):
                bbox = list(bbox)
            bbox.insert(0,img_path)
            self.annotations.append(bbox)
        '''
    def close_txt(self):
        self.voc_r.close()
        self.coco_r.close()

    def pull_item(self, index):
        '''
        output: img - shape(c,h,w)
                gt_boxes+label: box-(x1,y1,x2,y2)
                label: dataset_class_num 
        '''
        tmp_annotation = self.annotations[index]
        tmp_path = tmp_annotation[0]
        img_data = cv2.imread(tmp_path)
        h,w = img_data.shape[:2]
        img_data = img_data[:,:,::-1]
        gt_box_label = np.array(tmp_annotation[1:],dtype=np.float32).reshape(-1,5)
        #print(gt_box_label) 
        img_data, window = self.re_scale(img_data)
        img_data = self.normalize(img_data)
        return torch.from_numpy(img_data).permute(2, 0, 1),gt_box_label,h,w,window
    
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
        window = [ox,oy,new_w,new_h]
        return out, window
    
    def normalize(self,img):
        img = img / 255.0
        img[:,:,0] -= cfgs.PIXEL_MEAN[0]
        img[:,:,0] = img[:,:,0] / cfgs.PIXEL_NORM[0] 
        img[:,:,1] -= cfgs.PIXEL_MEAN[1]
        img[:,:,1] = img[:,:,1] / cfgs.PIXEL_NORM[1]
        img[:,:,2] -= cfgs.PIXEL_MEAN[2]
        img[:,:,2] = img[:,:,2] / cfgs.PIXEL_NORM[2]
        return img.astype(np.float32)

    def descale(self,box,window,img_w,img_h):
        ox,oy,new_w,new_h = window
        xmin, ymin, xmax, ymax = box[:,:,:,1],box[:,:,:,2],box[:,:,:,3],box[:,:,:,4]
        box[:,:,:,1] = (xmin - ox) / float(new_w) * img_w
        box[:,:,:,2] = (ymin - oy) / float(new_h) * img_h
        box[:,:,:,3] = (xmax - ox) / float(new_w) * img_w
        box[:,:,:,4] = (ymax - oy) / float(new_h) * img_h
        '''
        box[:,:,:,1] = np.minimum(np.maximum(xmin * img_w,0),img_w)
        box[:,:,:,2] = np.minimum(np.maximum(ymin * img_h,0),img_h)
        box[:,:,:,3] = np.minimum(np.maximum(xmax * img_w,0),img_w)
        box[:,:,:,4] = np.minimum(np.maximum(ymax * img_h,0),img_h)
        '''
        return box

if __name__=='__main__':
    test_d = ReadDataset()
    img_dict = dict()
    i=0
    total = 133644
    while 3-i:
        img, gt = test_d.get_batch(2)
        img_dict['img_data'] = img[0].numpy()
        img_dict['gt'] = gt[0]
        label_show(img_dict)
        #print(gt[0][:,-1])
        #sys.stdout.write('\r>> %d /%d' %(i,total))
        #sys.stdout.flush()
        i+=1
    print(i)