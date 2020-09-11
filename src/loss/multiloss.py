#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils_loss import bbox_overlaps_diou,bbox_overlaps_ciou,bbox_overlaps_giou,bbox_overlaps_iou
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from boxes_util import  match_ssd,match_soft,decode
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    this is equal: -x[class] + log(sum(exp(x))) = -log{exp(x[class]) / sum(exp(X))}
    in order to deal with inf: log(sum(exp)) = log(sum(exp(x-a))) + a
    """
    # x_max = x.data.max()
    x_max, x_max_indices = torch.max(x, dim=1, keepdim=True)
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max

class IouLoss(nn.Module):
    def __init__(self,pred_mode = 'Center',variances=None,losstype='Diou'):
        super(IouLoss, self).__init__()
        self.pred_mode = pred_mode
        self.variances = variances
        self.loss = losstype
    def forward(self, loc_p, loc_t,prior_data):
        if self.pred_mode == 'Center':
            decoded_boxes = decode(loc_p, prior_data, self.variances)
        else:
            decoded_boxes = loc_p
        if self.loss == 'Iou':
            loss = torch.sum(1.0 - bbox_overlaps_iou(decoded_boxes, loc_t))
        else:
            if self.loss == 'Giou':
                loss = torch.sum(1.0 - bbox_overlaps_giou(decoded_boxes,loc_t))
            else:
                if self.loss == 'Diou':
                    loss = torch.sum(1.0 - bbox_overlaps_diou(decoded_boxes,loc_t))
                else:
                    loss = torch.sum(1.0 - bbox_overlaps_ciou(decoded_boxes, loc_t))            
        return loss

class SmoothEntroy(nn.Module):
    def __init__(self,num_classes,weights=1.0,eps=1e-5,reduction='mean'):
        super(SmoothEntroy,self).__init__()
        if isinstance(weights,list):
            self.alpha = torch.Tensor(weights)
        else:
            self.alpha = torch.ones(num_classes)
        self.eps = eps
        self.size_average = reduction

    def forward(self,pred,targets,sparse_cls,ignore_label=-1):
        '''
        num_priors: positive and negtive anchors
        pred: [n*num_priors,cls_num],
        targets: [n*num_priors,cls_num]
        sparse_cls:[n,num_priors]
        '''
        cls_softmax = F.softmax(pred,dim=1)
        self.alpha = self.alpha.to(pred.device)
        keepids = sparse_cls >=0
        sparse_cls = sparse_cls[keepids]
        ids = sparse_cls.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)]
        cls_log = torch.log(cls_softmax+ self.eps)
        cls_log = cls_log[keepids]
        encty_loss = cls_log.gather(1,ids)
        targets = targets[keepids]
        # encty_loss = F.cross_entropy(pred,sparse_cls,reduction='none',ignore_index=-1)
        loss = targets * encty_loss
        loss = alpha * loss
        if self.size_average=='mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    def __init__(self, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = cfgs.CLSNUM
        self.negpos_ratio = cfgs.NEG_POS_RATIOS
        self.variance = cfgs.variance
        self.threshold = cfgs.OVERLAP_THRESH
        self.match = match_soft
        self.alpha=torch.FloatTensor([0.5,0.5,1,1,1,1,1,1,1])
        # self.softloss = SmoothEntroy(self.num_classes,self.alpha)
        # self.iouloss = IouLoss(variances=cfgs.variance)

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        conf_data,loc_data,priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = priors.size(0)
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        # mask_t = torch.Tensor(num,num_priors,cfgs.MaskOutsize[0],cfgs.MaskOutsize[1])
        soft_t = torch.Tensor(num,num_priors)
        defaults = priors.data
        for idx in range(num):
            bbox_label = targets[idx]
            if self.use_gpu:
                bbox_label = bbox_label.cuda()
            truths = bbox_label[:, :-1].data
            labels = bbox_label[:, -1].data
            # self.match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)
            self.match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,soft_t,idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            soft_t = soft_t.cuda()
            self.alpha = self.alpha.cuda()
        # wrap targets
        loc_t.requires_grad = False
        conf_t.requires_grad = False
        soft_t.requires_grad = False
        #get positive
        pos = conf_t > 0
        num_pos = pos.long().sum(dim=1, keepdim=True)
        # pos_neg = conf_t >=0
        # total_num = pos_neg.long().sum()
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # batch_priors = priors.unsqueeze(0).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        # soft_t = soft_t[pos_idx].view(-1,4)
        # batch_priors = batch_priors[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        # loss_iou = self.iouloss(loc_p,soft_t,batch_priors)
        # regression_diff = torch.abs(loc_t - loc_p)
        # loss_l = torch.where(
                    # torch.le(regression_diff, 1.0 / 9.0),
                    # 0.5 * 9.0 * torch.pow(regression_diff, 2),
                    # regression_diff - 0.5 / 9.0
                # )
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # loss_c = self.focal_loss(batch_conf,conf_t.view(-1,1),ignore.view(-1,1))
        conf_sel = conf_t.clone()
        ignore = conf_sel < 0
        conf_sel[ignore] = 0
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_sel.view(-1, 1))
        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # print('loss debug:',loss_idx[0,:10],idx_rank[0,:10])
        # print('loss:',loss_c[0,loss_idx[0,:10]])
        # num_pos = pos.sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio *num_pos,max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # the  pos and neg is bool,so get the true using gt(0)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        # soft_targets = soft_t[(pos_idx + neg_idx).gt(0)].view(-1,self.num_classes)
        soft_targets = soft_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p,targets_weighted,weight=self.alpha,reduction='none',ignore_index=-1)
        loss_c = soft_targets * loss_c
        loss_c = loss_c.sum()
        # loss_c = self.softloss(conf_p,soft_targets,targets_weighted)
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum() if num_pos.data.sum() > 0 else num
        # Ncls = num_neg.data.sum() 
        loss_l /= N
        loss_c /= N
        # loss_iou /= N
        return loss_c,loss_l

