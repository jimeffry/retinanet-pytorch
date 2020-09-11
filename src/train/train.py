import time
import os
import argparse
import pdb
import collections
import sys
import numpy as np
import logging
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import  DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from retinamask import RetinaMask
from detector import RetinanetDetector
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from anchors import Anchors
from boxes_util import match,match_ssd
from util_match import refine_match as match_ref
sys.path.append(os.path.join(os.path.dirname(__file__),'../loss'))
from multiloss import MultiBoxLoss
sys.path.append(os.path.join(os.path.dirname(__file__),'../prepare_data'))
# from dataloader import  collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from factory import dataset_factory,detection_collate
sys.path.append(os.path.join(os.path.dirname(__file__),'../test'))
from eval_voc import test_net
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# print('CUDA available: {}'.format(torch.cuda.is_available()))

def parms():
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', type=str,default='coco',help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations ')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--log_dir',type=str,default='../logs',help='save log dir')
    parser.add_argument('--model_path',type=str,default=None,help='load model paths')
    parser.add_argument('--start_iter',type=int,default=0,help='start')
    parser.add_argument('--use_cuda',type=str2bool,default=True,help='if use cuda')
    parser.add_argument('--batch_size',type=int,default=2,help='batch size for train')
    parser.add_argument('--gpu_list',type=str,default='0',help='traing gpus')
    parser.add_argument('--lr',type=float,default=0.01,help='the learning rate')
    parser.add_argument('--weight_decay',type=float,default=5e-4,help="")
    return parser.parse_args()

def rename_dict(state_dict):
    state_dict_new = dict()
    for key,value in list(state_dict.items()):
        state_dict_new[key[7:]] = value
    return state_dict_new
        

def main(args):
    use_cuda = args.use_cuda
    if not os.path.exists(cfgs.model_dir):
        os.makedirs(cfgs.model_dir)
    #*******************************************************************************create logg
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    log_name = time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'
    log_path = os.path.join(log_dir,log_name)
    hdlr = logging.FileHandler(log_path)
    logger.addHandler(hdlr)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    #*****************************************************************************Create the data loaders
    dataset_train,dataloader_val = dataset_factory()
    dataloader_train = DataLoader(dataset_train, num_workers=4, collate_fn=detection_collate,batch_size=args.batch_size,shuffle=True)
    # dataloader_val = DataLoader(dataset_val,num_workers=1,batch_size=1)
    #  dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
    #*********************************************************************************load model
    if torch.cuda.is_available() and use_cuda:
        device = torch.device('cuda')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
    else:
        device = torch.device('cpu')

    retinanet = RetinaMask(9,'train').to(device)
    if args.model_path:
        model_weights = torch.load(args.model_path,map_location=device)
        #model_weights = rename_dict(model_weights)
        retinanet.load_state_dict(model_weights,strict=True)
        logger.info("load weightes success: {}".format(args.model_path))
    BoxDetector = RetinanetDetector()
    #****************************************************************** load anchor
    get_anchors = Anchors()
    anchors = get_anchors(cfgs.IMGHeight,cfgs.IMGWidth)
    if use_cuda:
        anchors = anchors.cuda().float()
    print("anchors:",anchors.size())
    #*******************************************************************creat loss
    # focalLoss = FocalLoss()
    criterion = MultiBoxLoss(use_gpu=use_cuda)
    if len(args.gpu_list.split(','))>0:
        retinanet = torch.nn.DataParallel(retinanet)
    # retinanet.train()
    # optimizer = optim.Adam(retinanet.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    optimizer = optim.SGD(retinanet.parameters(), lr=args.lr, momentum=0.9,weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    loss_reg = collections.deque(maxlen=500)
    loss_cls = collections.deque(maxlen=500)
    # retinanet.module.freeze_bn()
    logger.info('Num training images: {}'.format(dataset_train.__len__()))
    rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
    rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
    step = 0
    tmp_max = 0.0
    for epoch_num in range(args.start_iter,args.epochs):
        retinanet.train()
        #retinanet.module.freeze_bn()
        lr = poly_lr_scheduler(optimizer,args.lr,epoch_num,max_iter=args.epochs, power=0.8)
        for idx,(img_batch, gt_batch )in enumerate(dataloader_train):
            save_fg = 0
            step +=1
            if use_cuda:
                img_batch = img_batch.cuda()
            '''
            images = img_batch.numpy()
            targets = gt_batch
            priors = anchors
            conf_t = test_anchor(targets,priors)
            for i in range(args.batch_size):
                tmp_img = np.transpose(images[i],(1,2,0))
                # tmp_img = tmp_img + rgb_mean
                # tmp_img = tmp_img * 255
                tmp_img *= rgb_std
                tmp_img += rgb_mean
                tmp_img *=255
                tmp_img = np.array(tmp_img,dtype=np.uint8)
                tmp_img = cv2.cvtColor(tmp_img,cv2.COLOR_RGB2BGR)
                h,w = tmp_img.shape[:2]
                if len(targets[i])>0:
                    gt = targets[i]
                    for j in range(gt.shape[0]):
                        x1,y1 = int(gt[j,0]),int(gt[j,1])
                        x2,y2 = int(gt[j,2]),int(gt[j,3])
                        # print('pred',x1,y1,x2,y2,gt[j,4],w,h)
                        if x2 >x1 and y2 >y1:
                            cv2.rectangle(tmp_img,(x1,y1),(x2,y2),(0,0,255))
                for j in range(priors.size(0)):
                    if conf_t[i,j] >0:
                        box = priors[j].cpu().numpy()
                        # print(box)
                        x1,y1 = box[0],box[1]
                        x2,y2 = box[2],box[3]
                        x1,y1 = int(x1),int(y1)
                        x2,y2 = int(x2),int(y2)
                        cv2.rectangle(tmp_img,(x1,y1),(x2,y2),(255,0,0))
                cv2.imwrite('train_match4.jpg',tmp_img)
                cv2.imshow('src',tmp_img)
                cv2.waitKey(0)
            '''
            classification, regression,_ = retinanet(img_batch)
            #print("begin to cal loss")
            classification_loss, regression_loss = criterion([classification,regression,anchors],gt_batch)
            # classification_loss = classification_loss.mean()
            # regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            if bool(loss == 0):
                continue
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            optimizer.step()
            loss_hist.append(float(loss.item()))
            loss_cls.append(float(classification_loss.item()))
            loss_reg.append(float(regression_loss.item()))
            if step %500 ==0:
                logger.info('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f} | cls_mean:{:.6f} | reg_mean:{:.6f} | lr: {:.6f}'.format(epoch_num,step,classification_loss.item(), regression_loss.item(), np.mean(loss_hist),np.mean(loss_cls),np.mean(loss_reg),lr))
            if step % 3000 ==0:
            #     mmap = test_net(retinanet,BoxDetector,anchors,dataloader_val,use_cuda,'train',args)
                save_fg = 1
            # if mmap > tmp_max:
            #     tmp_max = mmap
            #     save_fg = 1
            if save_fg:
                sfile = sfile = 'retina_' + args.dataset + '_best.pth'
                spath = os.path.join(cfgs.model_dir,sfile)
                if len(args.gpu_list.split(','))>0:
                    torch.save(retinanet.module.state_dict(),spath)
                else:
                    torch.save(retinanet.state_dict(),spath)
                logger.info("*****************save weightes******,%d" % step)
        # scheduler.step(np.mean(epoch_loss))
        # torch.save(retinanet.module, '{}/{}_retinanet_{}.pt'.format(args.model_dir,args.dataset, epoch_num))

def test_anchor(targets,priors):
    num_priors = priors.size(0)
    num = len(targets)
    loc_t = torch.Tensor(num, num_priors, 4)
    conf_t = torch.LongTensor(num, num_priors)
    defaults = priors.data
    # print(defaults.size())
    for idx in range(num):
        truths = targets[idx][:, :-1].data
        labels = targets[idx][:, -1].data
        # print(truths.size())
        match_ssd([0.3,0.5], truths, defaults,[0.1,0.2], labels,
                       loc_t, conf_t, idx)
    return conf_t

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    #     return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr

if __name__=='__main__':
    args = parms()
    main(args)
