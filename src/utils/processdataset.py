import numpy as np 
from matplotlib import pyplot as plt 
import os
import sys
import cv2
import tqdm
import json
import csv

def plothist(datadict,name):
    # xdata = datadict.keys()
    # ydata = []
    # for tmp in xdata:
    #     ydata.append(datadict[tmp])
    print('total plt:',len(datadict))
    fig, ax = plt.subplots(1, 1, figsize=(9, 3), sharey=True)
    # ax.bar(xdata,ydata)
    xn,bd,paths = ax.hist(datadict,bins=20)
    fw = open('../data/%s.txt' % name,'w')
    for idx,tmp in enumerate(xn):
        fw.write("{}:{}\n".format(tmp,bd[idx]))
    fw.close()
    plt.savefig('../data/%s.png' % name,format='png')
    plt.show()

def get_data2(imgdir):
    datas = []
    f1_cnts = os.listdir(imgdir)
    for i in tqdm.tqdm(range(len(f1_cnts))):
        tmp_f = f1_cnts[i].strip()
        tmpdir = os.path.join(imgdir,tmp_f)
        f2_cnts = os.listdir(tmpdir)
        for imgname in f2_cnts:
            imgpath = os.path.join(tmpdir,imgname.strip())
            img = cv2.imread(imgpath)
            h,w = img.shape[:2]
            datas.append(max(h,w))
    plothist(datas)

def getdata1(imgdir):
    datamax = []
    datamin = []
    fcnts = os.listdir(imgdir)
    for i in tqdm.tqdm(range(len(fcnts))):
        imgname = fcnts[i].strip()
        imgpath = os.path.join(imgdir,imgname)
        img = cv2.imread(imgpath)
        h,w = img.shape[:2]
        datamax.append(max(h,w))
        datamin.append(min(w,h))
    # plothist(datamax,'cocotrainmax')
    # plothist(datamin,'cocotrainmin')
    plothist(datamax,'voc12trainvalmax')
    plothist(datamin,'voc12trainvalmin')

def getdatalist(imgdir,outfile1,outfile2):
    fcnts = os.listdir(imgdir)
    fw = open(outfile1,'w')
    fw2 = open(outfile2,'w')
    cnt = 0
    total = len(fcnts)
    valc = total-200
    for tmp in fcnts:
        cnt+=1
        imgname = tmp.strip()[:-4]
        if cnt < valc:
            fw.write(imgname+'\n')
        else:
            fw2.write(imgname+'\n')
    fw.close()
    fw2.close()

if __name__=='__main__':
    # getdata1('/data/detect/COCO/train2017')
    getdata1('/data/detect/VOC/VOCdevkit/VOC2012/JPEGImages')
    # getdatalist('/data/detect/VOC/VOCdevkit/VOC2010/Seglabels22','../datasets/voctrain.txt','../datasets/vocval.txt')