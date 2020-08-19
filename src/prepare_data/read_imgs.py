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
    Detection Dataset Object
    """
    def __init__(self,imgdir,filein,mode='train'):
        self.imgdir = imgdir
        self.ids = []
        self.annotations = []
        self.load_txt(filein)
        self.total_num = self.__len__()
        self.shulf_num = list(range(self.total_num))
        random.shuffle(self.shulf_num)
        self.rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        self.rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        self._annopath = os.path.join('%s', 'Seglabels22', '%s.png')
        self._imgpath = os.path.join('%s', 'train2017', '%s.jpg')
        self.mode = mode

    def __getitem__(self, index):
        img,annot = self.pull_item(index)
        return img,annot

    def __len__(self):
        return len(self.annotations)

    def load_txt(self,filein):
        file_rd = open(filein,'r')
        file_txt = file_rd.readlines()
        '''
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
        for tmp in file_txt:
            tmp_splits = tmp.strip().split(',')
            # img_path = os.path.join(self.imgdir,tmp_splits[0])
            self.ids.append((self.imgdir,tmp_splits[0].split('/')[-1][:-4]))
            bbox = map(float, tmp_splits[1:])
            if not isinstance(bbox,list):
                bbox = list(bbox)
            # bbox.insert(0,img_path)
            self.annotations.append(bbox)

    def pull_item(self, index):
        '''
        output: img - shape(c,h,w)
                gt_boxes+label: box-(x1,y1,x2,y2)
                label: dataset_class_num 
        '''
        tmp_path = self._imgpath % (self.ids[index])
        tmp_annotation = self.annotations[index]
        # tmp_path = tmp_annotation[0]
        img_data = cv2.imread(tmp_path)
        img_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
        gt_box_label = np.array(tmp_annotation,dtype=np.float32).reshape(-1,5)
        img,gt = self.processimg(img_data,gt_box_label)
        return torch.from_numpy(img).permute(2, 0, 1),gt

    def pull_image(self,index):
        tmp_path = self._imgpath % (self.ids[index])
        # tmp_annotation = self.annotations[index]
        img_data = cv2.imread(tmp_path)
        img_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
        return img_data
    
    def processimg(self,img,gt):
        if self.mode == 'train':
            img,gt,_ = self.mirror(img,gt)
        img,gt = self.resize_subtract_mean(img,gt)
        return img,gt

    def mirror(self,image, boxes,mask=None):
        height, width, _ = image.shape
        boxes_tmp = boxes.copy()
        if random.randrange(2):
            image = image[:, ::-1,:]
            boxes[:, 0] = width - boxes_tmp[:, 2] -1
            boxes[:,2] = width - boxes_tmp[:,0] -1
        if mask is not None:
            mask = mask[:,::-1]
        return image,boxes,mask
    
    def rescale(self,image,boxes_f,height,width):
        boxes_f[:,0] = boxes_f[:,0] / float(width) #* cfgs.IMGWidth
        boxes_f[:,2] = boxes_f[:,2] / float(width) #* cfgs.IMGWidth
        boxes_f[:,1] = boxes_f[:,1] / float(height) #* cfgs.IMGHeight
        boxes_f[:,3] = boxes_f[:,3] / float(height) #* cfgs.IMGHeight
        image = cv2.resize(image,(cfgs.IMGWidth,cfgs.IMGHeight))
        return image,boxes_f
    
    def resize_subtract_mean(self,image,gt):
        # if self.mode =='train':
        h,w = image.shape[:2]
        if h < cfgs.IMGHeight or w < cfgs.IMGWidth:
            image,gt = self.rescale(image,gt,h,w)
        else:
            image,gt = self.cropimg(image,gt)
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.rgb_mean
        image = image / self.rgb_std
        return image,gt

    def cropimg(self,image,gt_box):
        h,w = image.shape[:2]
        while 1:
            dh,dw = int(random.random()*(h-cfgs.IMGHeight)),int(random.random()*(w-cfgs.IMGWidth))
            nx1 = dw
            nx2 = dw+cfgs.IMGWidth
            ny1 = dh
            ny2 = dh+cfgs.IMGHeight
            img = image[ny1:ny2,nx1:nx2,:]
            # gt = gt[dh:(dh+cfgs.IMGHeight),dw:(dw+cfgs.IMGWidth)]
            gt = gt_box.copy()
            keep_idx = np.where(gt[:,2]>nx1)
            gt = gt[keep_idx]
            keep_idx = np.where(gt[:,0]<nx2)
            gt = gt[keep_idx]
            keep_idx = np.where(gt[:,3]>ny1)
            gt = gt[keep_idx]
            keep_idx = np.where(gt[:,1]<ny2)
            gt = gt[keep_idx]
            gt[:,0] = np.clip(gt[:,0],nx1,nx2)-nx1
            gt[:,2] = np.clip(gt[:,2],nx1,nx2)-nx1
            gt[:,1] = np.clip(gt[:,1],ny1,ny2)-ny1
            gt[:,3] = np.clip(gt[:,3],ny1,ny2)-ny1
            gt[:,0] = gt[:,0] / float(cfgs.IMGWidth)
            gt[:,2] = gt[:,2] / float(cfgs.IMGWidth)
            gt[:,1] = gt[:,1] / float(cfgs.IMGHeight)
            gt[:,3] = gt[:,3] / float(cfgs.IMGHeight)
            if len(gt)>0:
                break
        return img,gt



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