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
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from boxes_util import  match_ssd
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
        self.match = match_ssd
        # self.focal_loss = FocalLoss(num_classes=self.num_classes,size_average=False)
        self.alpha=torch.FloatTensor([0.1,0.1,1,1,1,1,1,1,1])
        # self.focal_loss = FocalLoss2(self.num_classes,alpha=self.alpha,size_average=False)
        # self.focal_loss = FocalLoss3(weight=self.alpha,reduction='sum')

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
        defaults = priors.data
        for idx in range(num):
            bbox_label = targets[idx]
            if self.use_gpu:
                bbox_label = bbox_label.cuda()
            truths = bbox_label[:, :-1].data
            labels = bbox_label[:, -1].data
            self.match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            self.alpha = self.alpha.cuda()
        # wrap targets
        loc_t.requires_grad = False
        conf_t.requires_grad = False
        #get positive
        pos = conf_t > 0
        num_pos = pos.long().sum(dim=1, keepdim=True)
        # pos_neg = conf_t >=0
        # total_num = pos_neg.long().sum()
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
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
        loss_c = F.cross_entropy(conf_p,targets_weighted,weight=self.alpha,reduction='sum',ignore_index=-1)
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum() if num_pos.data.sum() > 0 else num
        Ncls = num_neg.data.sum() 
        loss_l /= N
        loss_c /= (N+Ncls)
        return loss_c,loss_l

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, targets):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        # preds = preds.view(-1,preds.size(-1))
        labels = targets.clone()
        ignore = labels < 0
        labels[ignore] = 0
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        clsmask = targets.view(-1,1)==-1
        # loss = torch.where(torch.ne(targets.view(-1,1), -1.0), loss, torch.zeros(loss.shape).cuda())
        loss[clsmask.t()] = 0.0
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class FocalLoss2(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss2, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1).clone()
        ignore = ids < 0
        ids[ignore] = 0
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        clsmask = targets.view(-1,1)==-1
        batch_loss[clsmask] = 0.0
        #print('-----bacth_loss------')
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class FocalLoss3(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss3, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
 
    def forward(self, output, target,cls_mask):
        # convert output to pseudo probability
        outsoft = F.softmax(output, dim=1)
        probs = torch.stack([outsoft[i, t] for i, t in enumerate(target)])
        focal_weight = torch.pow(1-probs, self.gamma)
        if output.is_cuda and not self.weight.is_cuda:
            self.weight = self.weight.cuda()
        # add focal weight to cross entropy
        ce_loss = F.cross_entropy(output, target, weight=self.weight, reduction='none')
        focal_loss = focal_weight * ce_loss
        focal_loss[cls_mask] = 0
        if self.reduction == 'mean':
            focal_loss = (focal_loss/focal_weight.sum()).sum()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
 
        return focal_loss