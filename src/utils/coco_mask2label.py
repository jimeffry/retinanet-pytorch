from pycocotools.coco import COCO
import numpy as np
import cv2
import os

# Add your coco dataset images folder path
data_dir = '/users/lijiayu/Downloads/riverSegmentation/riverMaskOriginal2/'
ann_file = '/users/lijiayu/downloads/riverData/annotations/instances_batch3_water_new.json'
# Add your output mask folder path
seg_output_path = '/users/lijiayu/downloads/riverSegmentation/riverMask2/'
# Store original images into another folder
original_img_path = '/users/lijiayu/downloads/riverSegmentation/riverMaskInput2/'
train = '/users/lijiayu/downloads/riverSegmentation/train2.txt'
val = '/users/lijiayu/downloads/riverSegmentation/val2.txt'
#test = '/users/lijiayu/downloads/riverSegmentation/test2.txt'

coco = COCO(ann_file)
catIds = coco.getCatIds(catNms=['water'])  # Add more categories ['person','dog']
imgIds = coco.getImgIds(catIds=catIds)
print(catIds)
train_file = open(train, "w")
# val_file = open(val, "w")
# test_file = open(test, "w")

problem_list = ["wuxi_2808.png"]

if not os.path.exists(original_img_path):
    os.makedirs(original_img_path)

if not os.path.exists(seg_output_path):
    os.makedirs(seg_output_path)

print("total N of images:", len(imgIds))
for i in range(len(imgIds)):
    img = coco.loadImgs(imgIds[i])[0]
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=0)

    anns = coco.loadAnns(annIds) #list
    # print(anns)

    mask = coco.annToMask(anns[0])
    for j in range(1, len(anns)):
        mask += coco.annToMask(anns[j])

    file_name = data_dir+img['file_name']
    print(i, file_name)
    #image_name = "wuxi_{}.png".format(9667+i)
    image_name = img['file_name'].split("/")[-1]
    #image_name = img['file_name']
    #if image_name in problem_list:


    original_img = cv2.imread(file_name)
    #cv2.imwrite(os.path.join(original_img_path, image_name), original_img)
    #cv2.imwrite(os.path.join(seg_output_path, image_name), mask)

    if i < 9000:
        i = i+1
        train_file.write(image_name.split(".")[0] + "\n")
    # # elif i >=8000 and i < 9000:
    # #     val_file.write(img['file_name'].split(".")[0] + "\n")
    # else:
    #     val_file.write(img['file_name'].split(".")[0] + "\n")

    print("processing...")

print("Done")