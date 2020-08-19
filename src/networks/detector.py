#######################################################
#author: lxy
#time: 14:30 2019.7.24
#tool: python3
#version: 0.1
#project: pedestrian detection
#modify:
#####################################################
import os
import sys
import cv2
import torch
from torch import nn
from torch.autograd import Function
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from pth_nms import nms_py
from boxes_util import decode
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

class RetinanetDetector(Function):
    def __init__(self):
        #super(RetinanetDetector,self).__init__()
        self.top_k = cfgs.top_k
        self.score_threshold = cfgs.conf_threshold
        self.nms_threshold = cfgs.nms_threshold
        self.num_classes = cfgs.CLSNUM
        self.softmax = nn.Softmax(dim=-1)
        self.variance = cfgs.variance

    def forward(self,anchors,regression,classification):
        num = classification.size(0)
        num_priors = anchors.size(0)
        conf_data = self.softmax(classification)
        # print('anchor,reg,cls',anchors.size(),regression.size(),classification.size())
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        batch_priors = anchors.view(-1, num_priors,4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)
        decoded_boxes = decode(regression.view(-1, 4),batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        #scores = torch.max(classification, dim=2, keepdim=True)[0]
        #scores_over_thresh = (scores>self.score_threshold)[0, :, 0]
        
        # batch = transformed_anchors.size(0)  # batch size
        # num_priors = anchors.size(1)
        # output = torch.zeros(batch, self.num_classes, self.top_k, 5)
        # conf_preds = classification.view(batch, num_priors,self.num_classes).transpose(2, 1)
        for i in range(num):
            bboxes = decoded_boxes[i].clone()
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            for cl in range(1,self.num_classes):
                c_mask = conf_scores[cl].gt(self.score_threshold)
                scores = conf_scores[cl][c_mask]
                # print('mask,box',c_mask.size(),boxes.size())
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(bboxes)
                boxes = bboxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                #print(boxes.size(), scores.size())
                # ids, count = pth_nms(boxes, scores,self.nms_threshold,self.top_k)
                ids,count = nms_py(boxes.detach().cpu().numpy(),scores.detach().cpu().numpy(),self.nms_threshold,self.top_k)
                ids = torch.tensor(ids,dtype=torch.long)
                if count ==0:
                    continue
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].view(-1,1),
                               boxes[ids[:count]].view(-1,4)), 1)
        '''
        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return (torch.zeros(0), torch.zeros(0), torch.zeros(0, 4))
        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]
        anchors_nms_idx,_ = pth_nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :],overlap=cfgs.nms_threshold)
        nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
        return (nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :])
        '''
        return output


class MaskTarget(Function):
    def __init__(self):
        #super(RetinanetDetector,self).__init__()
        self.top_k = cfgs.top_k
        self.score_threshold = cfgs.conf_threshold
        self.nms_threshold = cfgs.nms_threshold
        self.num_classes = cfgs.CLSNUM
        self.softmax = nn.Softmax(dim=-1)
        self.variance = cfgs.variance
        self.mask_h,self.mask_w = cfgs.MaskOutsize

    def forward(self,anchors,regression,classification,masks_t):
        num = classification.size(0)
        num_priors = anchors.size(0)
        conf_data = self.softmax(classification)
        # print('anchor,reg,cls',anchors.size(),regression.size(),classification.size())
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        batch_priors = anchors.view(-1, num_priors,4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)
        decoded_boxes = decode(regression.view(-1, 4),batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)
        output = torch.zeros(num, self.num_classes, self.top_k,self.mask_h,self.mask_w)
        #scores = torch.max(classification, dim=2, keepdim=True)[0]
        #scores_over_thresh = (scores>self.score_threshold)[0, :, 0]
        
        # batch = transformed_anchors.size(0)  # batch size
        # num_priors = anchors.size(1)
        # output = torch.zeros(batch, self.num_classes, self.top_k, 5)
        # conf_preds = classification.view(batch, num_priors,self.num_classes).transpose(2, 1)
        for i in range(num):
            bboxes = decoded_boxes[i].clone()
            masks = masks_t[i]
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            for cl in range(1,self.num_classes):
                c_mask = conf_scores[cl].gt(self.score_threshold)
                scores = conf_scores[cl][c_mask]
                # print('mask,box',c_mask.size(),boxes.size())
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(bboxes)
                m_mask = c_mask.unsqueeze(1).expand_as(masks)
                boxes = bboxes[l_mask].view(-1, 4)
                masks = masks[m_mask]
                # idx of highest scoring and non-overlapping boxes per class
                #print(boxes.size(), scores.size())
                # ids, count = pth_nms(boxes, scores,self.nms_threshold,self.top_k)
                ids,count = nms_py(boxes.detach().cpu().numpy(),scores.detach().cpu().numpy(),self.nms_threshold,self.top_k)
                ids = torch.tensor(ids,dtype=torch.long)
                if count ==0:
                    continue
                # output[i, cl, :count] = torch.cat((scores[ids[:count]].view(-1,1),boxes[ids[:count]].view(-1,4)), 1)
                # output[i,cl,:count] = masks[ids[:count]]
                maskcrop(boxes,masks,ids,count,self.mask_h,self.mask_w,i,cl,output)
        '''
        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return (torch.zeros(0), torch.zeros(0), torch.zeros(0, 4))
        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]
        anchors_nms_idx,_ = pth_nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :],overlap=cfgs.nms_threshold)
        nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
        return (nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :])
        '''
        return output

def maskcrop(boxes,masks,ids,count,h,w,idx,cl,mask_t):
    '''
    boxes: [num,4]
    masks: [num,h,w]
    '''
    boxes_tt = boxes[ids[:count]].view(-1,4)
    masks_tt = masks[ids[:count]]
    for i in range(count):
        x1,y1,x2,y2 = boxes_tt[i]
        masktmp = masks_tt[i]
        xc,yc = (x1+x2)/2.0,(y1+y2)/2.0
        r = max(x2-x1,y2-y1)
        x1 = xc-r/2.0
        x2 = xc+r/2.0
        y1 = yc-r/2.0
        y2 = yc+r/2.0
        masktmp = masktmp[y1:y2,x1:x2]
        masktmp = cv2.resize(masktmp,(h,w))
        mask_t[idx,cl,i] = masktmp