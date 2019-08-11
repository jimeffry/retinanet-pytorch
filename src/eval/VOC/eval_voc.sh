 #!/usr/bin/bash
 python voc_eval.py --trained_model /data/models/retinanet/coco_retinanet_53.pth  \
   --save_folder /data/VOC/VOCdevkit/VOC2007/retinanet --voc_root /data/VOC/VOCdevkit --cuda Fasle