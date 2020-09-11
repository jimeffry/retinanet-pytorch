from easydict import EasyDict

cfgs = EasyDict()

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
#************************************************************dataset
cfgs.CLSNUM = 9
cfgs.COCODataNames = ['person','bicycle','motorcycle','car','bus','airplane','train','boat']
cfgs.VOCDataNames = ['person','bicycle','motorbike','car','bus','aeroplane','train','boat']
#cfgs.VOCDataNames = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair',
 #   'cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']
cfgs.PIXEL_MEAN = [0.485,0.456,0.406] # R, G, B
cfgs.PIXEL_NORM = [0.229,0.224,0.225] #rgb
cfgs.variance = [0.1, 0.2]
cfgs.voc_file = '/home/lxy/Develop/git_prj/retinanet-pytorch/data/trainval12.txt'  
cfgs.voc_dir = '/data/detect/VOC/VOCdevkit'
cfgs.coco_file = '/home/lxy/Develop/git_prj/retinanet-pytorch/data/coco2017train.txt'
cfgs.coco_dir = '/data/detect/COCO' # '/wdc/LXY.data/CoCo2017'
cfgs.train_file = '../data/coco2017train.txt'
cfgs.val_file = '../data/coco2017val.txt'
cfgs.imagedir = '/data/detect/COCO' #'/mnt/data/LXY.data/COCO' #'/data/detect/COCO'
cfgs.model_dir = '/data/models/retinanet' #'/mnt/data/LXY.data/models/retinanet'
#**********************************************************************train
cfgs.Show_train_info = 100
cfgs.Smry_iter = 2000
cfgs.Total_Imgs = 133459#133644
cfgs.IMGHeight = 640#540
cfgs.IMGWidth = 1280#960
cfgs.NEG_POS_RATIOS = 3.0
cfgs.OVERLAP_THRESH = [0.4,0.5] # negtive < 0.3, positive >0.5
cfgs.ModelPrefix = 'coco_retinanet' #'coco_retinanet' #'coco_resnet_50_state_dict' #
cfgs.Momentum = 0.9
cfgs.Weight_decay = 5e-4
cfgs.lr_steps = [20000, 40000, 60000]
cfgs.lr_gamma = 0.1
cfgs.epoch_num = 120000
#****************************************************model
cfgs.FPNSIZES=[128*4,256*4,512*4]
cfgs.MaskFPNScales = []
cfgs.MaskOutsize = (28,28)
#*******************************************************test
#*******************************************************test
cfgs.top_k = 300
cfgs.score_threshold = 0.2
cfgs.nms_threshold = 0.4
cfgs.model_dir = '/data/models/retinanet'
cfgs.conf_threshold = 0.01
cfgs.labelmap = ['person','bicycle','motorbike','car','bus','aeroplane','train','boat']
cfgs.save_folder = '/mnt/data/LXY.data/COCO/train_val'
cfgs.use_07 = True
# cfgs.shownames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#                 'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
#                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
#                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                 'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
#                 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                 'teddy bear', 'hair drier', 'toothbrush']
cfgs.shownames = ['bg','person','bicycle','motorbike','car','bus','aeroplane','train','boat']