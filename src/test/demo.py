########################################
#author: lxy
#time: 14:30 2019.7.24
#tool: python3
#version: 0.1
#project: pedestrian detection
#modify:
#####################################################
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
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from anchors import Anchors
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from retinamask import RetinaMask
from detector import RetinanetDetector
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def str2bool(rawstr):
    return rawstr.lower() in ['yes','true','t','1']

def parms():
    parser = argparse.ArgumentParser(description='refinedet test')
    parser.add_argument('--weights', default='',type=str, help='Trained state_dict file path')
    parser.add_argument('--cuda', default=False, type=str2bool,help='Use cuda in live demo')
    parser.add_argument('--img_path',type=str,default='',help='')
    parser.add_argument('--modelpath',type=str,default=None,help='load model path')
    parser.add_argument('--img_dir',type=str,default='',help='')
    parser.add_argument('--save_dir',type=str,default='',help='')
    return parser.parse_args()

class Retinanet_Test(object):
    def __init__(self,args):
        self.imgh = cfgs.IMGHeight
        self.imgw  = cfgs.IMGWidth
        self.img_dir = args.img_dir
        self.save_dir = args.save_dir
        self.use_gpu = args.cuda
        self.rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        self.rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        self.build_net(args.modelpath)
        self.laod_anchor()
    
    def build_net(self,load_path):
        self.Retinanet_model = RetinaMask(cfgs.CLSNUM,'test')
        self.Detector = RetinanetDetector()
        print('Resuming training, loading {}...'.format(load_path))
        if self.use_gpu:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            weights = torch.load(load_path,map_location=device)
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
        # img_batch = torch.ones((1,3,self.img_size,self.img_size))
        self.anchors = get_anchor(self.imgh,self.imgw)
        if self.use_gpu:
            self.anchors = self.anchors.cuda()
    
    def inference(self,img):
        '''
        input_img: [batch,c,h,w]
        '''
        t1 = time.time()
        input_img = img.copy()
        batch_imgs,batch_shapes = self.preprocess([input_img])
        if self.use_gpu:
            batch_imgs = batch_imgs.cuda()
        pred_cls, pred_bbox,conf_maps = self.Retinanet_model(batch_imgs)
        rectangles = self.Detector(self.anchors, pred_bbox, pred_cls)
        detections = rectangles.data.cpu().numpy()
        t2=time.time()
        print('consume:',t2-t1)
        # scale each detection back up to the image
        self.label_show(detections,[img],batch_shapes)
        return img,conf_maps  #rectangles,conf_maps
    
    def preprocess(self,imglist):
        '''
        img: bgr --> rgb
        '''
        out_list = []
        shape_list = []
        for image in imglist:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            h,w = image.shape[:2]
            image = cv2.resize(image,(self.imgw,self.imgh))
            image = image.astype(np.float32)
            image = image / 255.0
            image -= self.rgb_mean
            image = image / self.rgb_std
            image = np.transpose(image,(2,0,1))
            out_list.append(image)
            shape_list.append([h,w])
        out_list = np.array(out_list)
        return torch.from_numpy(out_list),shape_list

    def label_show(self,boxes,framelist,shapelist):
        COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        for indx in range(len(framelist)):
            frame = framelist[indx]
            tmph,tmpw = shapelist[indx]
            for i in range(1,boxes.shape[1]):
                j = 0
                while boxes[indx, i, j, 0] >= cfgs.conf_threshold:
                    score = boxes[indx,i,j,0]
                    pt = boxes[indx, i, j, 1:] #* scale
                    #print(pt)
                    pt[0] = pt[0]/cfgs.IMGWidth * tmpw
                    pt[2] = pt[2]/cfgs.IMGWidth * tmpw
                    pt[1] = pt[1]/cfgs.IMGHeight * tmph
                    pt[3] = pt[3]/cfgs.IMGHeight * tmph
                    min_re = min(pt[2]-pt[0],pt[3]-pt[1])
                    txt = str(score) #cfgs.shownames[i]
                    if min_re <16:
                        thres = 0.35
                        font_scale = int(1)
                    else:
                        thres = 0.4
                        font_scale = int((pt[2]-pt[0])*0.01)
                    if score >=thres:
                        cv2.rectangle(frame,(int(pt[0]), int(pt[1])),(int(pt[2]), int(pt[3])),(0,255,0), 2)
                        cv2.putText(frame,txt, (int(pt[0]), int(pt[1])),
                                FONT,0.5, (255, 255, 255), 1, 4)#cv2.LINE_AA)
                    j += 1
    def label_show_org(self,scores, cls_ids, bboxes,img):
        idxs = np.where(scores>cfgs.score_threshold)
        for j in range(idxs[0].shape[0]):
            bbox = bboxes[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = cfgs.shownames[int(cls_ids[idxs[0][j]])]
            #cv2.putText(img, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
            cv2.putText(img, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            #print(label_name)

    def get_hotmaps(self,conf_maps):
        '''
        conf_maps: feature_pyramid maps for classification
        '''
        hotmaps = []
        for tmp_map in conf_maps:
            batch,h,w,c = tmp_map.size()
            tmp_map = tmp_map.view(batch,h,w,-1,cfgs.CLSNUM)
            tmp_map = tmp_map[0]
            tmp_map_soft = torch.nn.functional.softmax(tmp_map,dim=3)
            cls_mask = torch.argmax(tmp_map_soft,dim=3,keepdim=True)
            #score,cls_mask = torch.max(tmp_map_soft,dim=4,keepdim=True)
            #cls_mask = cls_mask.unsqueeze(4).expand_as(tmp_map_soft)
            #print(cls_mask.data.size(),tmp_map_soft.data.size())
            tmp_hotmap = tmp_map_soft.gather(3,cls_mask)
            map_mask = torch.argmax(tmp_hotmap,dim=2,keepdim=True)
            tmp_hotmap = tmp_hotmap.gather(2,map_mask)
            tmp_hotmap.squeeze_(3)
            tmp_hotmap.squeeze_(2)
            print('map max:',tmp_hotmap.data.max())
            hotmaps.append(tmp_hotmap.data.numpy())
        print('hotmap num:',len(hotmaps))
        return hotmaps

    def display_hotmap(self,hotmaps):
        '''
        hotmaps: a list of hot map ,every shape is [1,h,w]
        '''       
        fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True)
        ax1 = axes[0,0]
        im1 = ax1.imshow(hotmaps[0])
        # We want to show all ticks...
        #ax.set_xticks(np.arange(len(farmers)))
        #ax.set_yticks(np.arange(len(vegetables)))
        # ... and label them with the respective list entries
        #ax.set_xticklabels(farmers)
        #ax.set_yticklabels(vegetables)
        # Rotate the tick labels and set their alignment.
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         #       rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        #for i in range(len(vegetables)):
         #   for j in range(len(farmers)):
          #      text = ax.text(j, i, harvest[i, j],
           #                 ha="center", va="center", color="w")
        #cb1 = fig.colorbar(im1)
        ax1.set_title("feature_3")
        #**************************************************************
        ax2 = axes[0,1]
        im2 = ax2.imshow(hotmaps[1])
        #cb2 = fig.colorbar(im2)
        ax2.set_title('feature_4')
        #************************************************
        ax3 = axes[0,2]
        im3 = ax3.imshow(hotmaps[2])
        #cb3 = fig.colorbar(im3)
        ax3.set_title('feature_5')
        #**********************************************
        img = hotmaps[3]
        min_d = np.min(img)
        max_d = np.max(img)
        tick_d = []
        while min_d < max_d:
            tick_d.append(min_d)
            min_d+=0.01
        ax4 = axes[1,0]
        im4 = ax4.imshow(hotmaps[3])
        ax4.set_title('feature_6')
        cb4 = fig.colorbar(im4) #ticks=tick_d)
        #***********************************************
        ax5 = axes[1,1]
        img5 = ax5.imshow(hotmaps[4])
        ax5.set_title('feature_7')
        #fig.tight_layout()
        plt.savefig('hotmap.png')
        plt.show()

    def test_dir(self,imgpath):
        print(imgpath)
        if os.path.isdir(imgpath):
            img_paths = glob.glob(os.path.join(imgpath,'*'))
            # save_dir = os.path.join(imgpath,'test')
            # if not os.path.exists(save_dir):
                # os.makedirs(save_dir)
            for idx,tmp in enumerate(img_paths):
                if not os.path.isfile(tmp):
                    continue
                img = cv2.imread(tmp)
                if img is None:
                    print('None',tmp)
                    continue
                frame,_ = self.inference(img)
                cv2.imshow('result',frame)
                cv2.waitKey(0)
                # savepath = os.path.join(save_dir,'test_%d.jpg' % idx)
                # cv2.imwrite(savepath,frame)
        elif os.path.isfile(imgpath) and imgpath.endswith('txt'):
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            f_r = open(imgpath,'r')
            file_cnts = f_r.readlines()
            for j in tqdm(range(len(file_cnts))):
                tmp_file = file_cnts[j].strip()
                #tmp_file = file_cnts[j].strip()+'.jpg'
                tmp_splits = tmp_file.split(',')
                tmp_file = tmp_splits[0] #.split('/')[-1]
                gt_box = map(float,tmp_splits[1:])
                gt_box = np.array(list(gt_box))
                gt_box = gt_box.reshape([-1,5])
                save_name = tmp_file
                tmp_path = os.path.join(self.img_dir,tmp_file) 
                if not os.path.exists(tmp_path):
                    continue
                img = cv2.imread(tmp_path) 
                if img is None:
                    print('None',tmp)
                    continue
                #frame,_ = self.test_img(img)                
                for idx in range(gt_box.shape[0]):
                    pt = gt_box[idx,:4]
                    i = int(gt_box[idx,4])
                    cv2.rectangle(img,
                                (int(pt[0]), int(pt[1])),
                                (int(pt[2]), int(pt[3])),
                                (0,0,255), 2) 
                    cv2.putText(img, cfgs.VOCDataNames[i], (int(pt[0]), int(pt[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, 4)#cv2.LINE_AA)
                cv2.imshow('result',img)
                cv2.waitKey(0)               
                #savepath = os.path.join(self.save_dir,save_name)
                #cv2.imwrite(savepath,frame)
        elif os.path.isfile(imgpath) and imgpath.endswith(('.mp4','.avi')) :
            #url = "rtsp://admin:dh123456@192.168.2." + ip + "/cam/realmonitor?channel=1&subtype=0"
            url = "rtsp://admin:hk123456@192.168.1.64/h264/1/main/av_stream"
            cap = cv2.VideoCapture(imgpath)
            if not cap.isOpened():
                print("failed open camera")
                return 0
            else: 
                while 1:
                    _,img = cap.read()
                    frame,_ = self.test_img(img)
                    cv2.imshow('result',frame)
                    q=cv2.waitKey(10) & 0xFF
                    if q == 27 or q ==ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()
        elif os.path.isfile(imgpath):
            img = cv2.imread(imgpath)
            if img is not None:
                # grab next frame
                # update FPS counter
                frame,odm_maps = self.inference(img)
                # hotmaps = self.get_hotmaps(odm_maps)
                # self.display_hotmap(hotmaps)
                # keybindings for display
                cv2.imshow('result',img)
                cv2.imwrite('test1.jpg',img)
                key = cv2.waitKey(0) 
        else:
            print('please input the right img-path')

if __name__ == '__main__':
    args = parms()
    model_net = Retinanet_Test(args)
    img_path = args.img_path
    model_net.test_dir(img_path)
