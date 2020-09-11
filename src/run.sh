#!/usr/bin/bash
# python test/demo.py --img_path /data/detect/t11.png --modelpath /data/models/retinanet/retina_coco_best.pth
# python test/demo.py  --modelpath /data/models/retinanet/retina_coco_best.pth --img_path /data/videos/anshan_crops2/752_5.jpg 
# python test/demo.py --img_path /home/lxy/Desktop/imgsegs --modelpath /data/models/retinanet/retina_coco_bestv2_iou.pth
# python test/demo.py --img_path ../data/coco2017val.txt --img_dir /data/detect/COCO --modelpath /data/models/retinanet/retina_coco_bestv2_iou.pth
python test/demo.py --img_path /home/lxy/Desktop/imgsegs/tf31.png --modelpath /data/models/retinanet/retina_coco_bestv2_iou.pth --save_dir ~/Desktop/imgzip
#python test/demo.py --img_path /data/test/car1.jpeg --load_num 124
# python test/demo.py --img_path  /data/detect/al1.png --load_num 124
#python test/demo.py --img_path /data/pedestrian_2.mp4 --load_num 124
#python test/demo.py --img_path /data/videos/ped2.mp4 --load_num 55
#******************
# python train/train.py --use_cuda false --batch_size 2 --lr 0.001 --gpu_list 0 --epochs 100 --dataset coco #--model_path /mnt/data/LXY.data/models/retinanet/retina_coco_best.pth

#**************process dataset
# python utils/process_voc.py --VOC-dir /data/detect/VOC/VOCdevkit/VOC2007 --anno-file /data/detect/VOC/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt --out-file ../data/trainval07.txt --recordfile ../data/voc07_record.txt --cmd-type readvoc
# python utils/process_voc.py --VOC-dir=/data/detect/VOC/VOCdevkit/VOC2012 --anno-file /data/detect/VOC/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt --out-file ../data/trainval12.txt --recordfile ../data/voc12_record.txt --cmd-type readvoc
#python voc_process.py --VOC-dir /data/VOC/VOCdevkit/VOC2007 --anno-file /data/VOC/VOCdevkit/VOC2007/ImageSets/Main/test.txt --out-file ../../datas/VOC/test_voc07.txt --save-name test_record --cmd-type readvoc

# python utils/process_coco.py --image_dir /data/detect/COCO/train2017 --annotation_path /data/detect/COCO/annotations/instances_train2017.json --recordfile ../data/cocotrain_record.txt --out_file ../data/cocos3train.txt
# python utils/process_coco.py --image_dir /data/detect/COCO/val2017 --annotation_path /data/detect/COCO/annotations/instances_val2017.json --recordfile ../data/cocoval_record.txt --out_file ../data/cocos3val.txt

# python utils/processdataset.py