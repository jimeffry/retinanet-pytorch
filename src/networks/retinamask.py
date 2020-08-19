import sys
import os
import torch.nn as nn
import torch
import math
import time
import torch.nn.functional as F
from resnet import resnet50
from fpn import PyramidFeatures
from bbox_cls_head import RegressionModel,ClassificationModel
from maskhead import MaskPredictor
# from pooler import Pooler
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

class RetinaMask(nn.Module):
    def __init__(self,num_classes,mode='train'):
        super(RetinaMask,self).__init__()
        if mode=='train':
            loadmodel=True
        else:
            loadmodel=False
        self.backbone = resnet50(num_classes, pretrained=loadmodel)
        fpn_sizes = cfgs.FPNSIZES
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        # self.roipool = Pooler(cfgs.MaskOutsize,cfgs.MaskFPNScales)
        # self.maskpredict = MaskPredictor(1024,256,num_classes)
        if mode=='train':
            prior = 0.01
            self.classificationModel.output.weight.data.fill_(0)
            self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

    def forward(self,x):
        pred_cls = list()
        scores_maps = list()
        outlist = self.backbone(x)
        features = self.fpn(outlist)
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        for feature in features:
            tmp = self.classificationModel(feature)
            scores_maps.append(tmp[1])
            pred_cls.append(tmp[0])
        # classification = torch.cat([tmp for tmp in pred_cls], dim=1)
        classification = torch.cat(pred_cls,dim=1)
        return classification,regression,scores_maps

    def freeze_bn(self):
        self.backbone.freeze_bn()