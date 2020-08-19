import os
import sys
from read_imgs import ReadDataset
import torch
# from torchvision import transforms
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def dataset_factory(mode='train'):
    dataset_train = ReadDataset(cfgs.imagedir,cfgs.train_file)
    dataset_val = ReadDataset(cfgs.imagedir,cfgs.val_file,'val')
    return dataset_train,dataset_val

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.from_numpy(sample[1]).float())
    return torch.stack(imgs, 0), targets