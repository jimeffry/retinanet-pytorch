import time
import os
import argparse
import pdb
import collections
import sys
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
import model
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from anchors import Anchors
sys.path.append(os.path.join(os.path.dirname(__file__),'../loss'))
import losses
sys.path.append(os.path.join(os.path.dirname(__file__),'../prepare_data'))
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer,ReadDataset

assert torch.__version__.split('.')[1] == '4'

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

print('CUDA available: {}'.format(torch.cuda.is_available()))

def parms():
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--model_dir',type=str,default='models',help='save model dir')
    parser.add_argument('--log_dir',type=str,default='loggs',help='save log dir')
    parser.add_argument('--model_path',type=str,default=None,help='load model paths')
    parser.add_argument('--start_iter',type=int,default=0,help='start')
    parser.add_argument('--cuda',type=str2bool,default=True,help='if use cuda')
    parser.add_argument('--batch_size',type=int,default=2,help='batch size for train')
    parser.add_argument('--gpu_list',type=str,default='0',help='')
    return parser.parse_args()

def main(args):
    gpu_list = [int(i) for i in args.gpu_list.split(',')]
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
    if args.dataset == 'coco':
        if args.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')
        #dataset_train = CocoDataset(args.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_train = ReadDataset(transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        #dataset_val = CocoDataset(args.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    elif args.dataset == 'csv':
        if args.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')
        if args.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')
        dataset_train = CSVDataset(train_file=args.csv_train, class_list=args.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        if args.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=args.csv_val, class_list=args.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=args.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=1, collate_fn=collater, batch_sampler=sampler)
    #if dataset_val is not None:
     #   sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
      #  dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
    # Create the model
    if args.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=False)
    elif args.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=False)
    elif args.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=False)
    elif args.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=False)
    elif args.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=False)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')   
    if args.model_path:
        model_weights = torch.load(args.model_path)
        retinanet.load_state_dict(model_weights,strict=False)
        logger.info("load weightes success: {}".format(args.model_path))
    #****************************************************************** load anchor
    get_anchors = Anchors()
    #*******************************************************************creat loss
    focalLoss = losses.FocalLoss()
    use_gpu = True
    if use_gpu:
        retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()
    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    retinanet.train()
    retinanet.module.freeze_bn()
    logger.info('Num training images: {}'.format(len(dataset_train)))
    for epoch_num in range(args.start_iter,args.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                img_batch = data['img']
                img_batch = img_batch.cuda().float()
                gt_batch = data['annot']
                gt_batch = torch.tensor(gt_batch,dtype=torch.float)
                gt_batch = gt_batch.cuda()
                classification, regression = retinanet(img_batch)
                anchors = get_anchors(img_batch)
                anchors = anchors.cuda().float()
                #print("begin to cal loss")
                classification_loss, regression_loss = focalLoss(classification,regression,anchors,gt_batch)
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss
                if bool(loss == 0):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))
                if iter_num %100 ==0:
                    logger.info('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f} | lr: {:.6f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist),optimizer.param_groups[0]['lr']))
                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue
        # if args.dataset == 'coco':
        #     print('Evaluating dataset')
        #     coco_eval.evaluate_coco(dataset_val, retinanet)
        # elif args.dataset == 'csv' and args.csv_val is not None:
        #     print('Evaluating dataset')
        #     mAP = csv_eval.evaluate(dataset_val, retinanet)
        scheduler.step(np.mean(epoch_loss))
        #torch.save(retinanet.module, '{}/{}_retinanet_{}.pt'.format(args.model_dir,args.dataset, epoch_num))
        torch.save(retinanet.module.state_dict(),'{}/{}_retinanet_{}.pth'.format(args.model_dir,args.dataset, epoch_num))
        logger.info("*****************save weightes")
    retinanet.eval()
    torch.save(retinanet, '{}/model_final.pth'.format(args.model_dir))

if __name__ == '__main__':
    args = parms()
    main(args)
