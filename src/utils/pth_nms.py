import torch
#from ._ext import nms
import numpy as np

def diounms(boxes, scores, overlap=0.5, top_k=200, beta1=1.0):
    """Apply DIoU-NMS at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
        beta1: (float) DIoU=IoU-R_DIoU^{beta1}.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        inx1 = torch.clamp(xx1, min=x1[i])
        iny1 = torch.clamp(yy1, min=y1[i])
        inx2 = torch.clamp(xx2, max=x2[i])
        iny2 = torch.clamp(yy2, max=y2[i])
        center_x1 = (x1[i] + x2[i]) / 2
        center_y1 = (y1[i] + y2[i]) / 2
        center_x2 = (xx1 + xx2) / 2
        center_y2 = (yy1 + yy2) / 2
        d = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2
        cx1 = torch.clamp(xx1, max=x1[i])
        cy1 = torch.clamp(yy1, max=y1[i])
        cx2 = torch.clamp(xx2, min=x2[i])
        cy2 = torch.clamp(yy2, min=y2[i])
        c = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
        u= d / c
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = inx2 - inx1
        h = iny2 - iny1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union - u ** beta1 # store result in diou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def pth_nms(boxes, scores=None,overlap=0.5, top_k=500):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    if scores is None:
        scores = boxes[:,4]
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep,count

def nms_py(boxes, scores, threshold=0.7,topk=200,mode='Union'):
    pick = []
    count = 0
    if len(boxes)==0:
        return pick,count
    # print('score',np.shape(scores))
    # boxes = boxes.detach().numpy().copy()
    # scores = scores.detach().numpy().copy()
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # s  = np.array(scores)
    area = np.multiply(x2-x1+1, y2-y1+1)
    ids = np.array(scores.argsort())
    ids = ids[-topk:]
    #I[-1] have hightest prob score, I[0:-1]->others
    while len(ids)>0:
        pick.append(ids[-1])
        xx1 = np.maximum(x1[ids[-1]], x1[ids[0:-1]]) 
        yy1 = np.maximum(y1[ids[-1]], y1[ids[0:-1]])
        xx2 = np.minimum(x2[ids[-1]], x2[ids[0:-1]])
        yy2 = np.minimum(y2[ids[-1]], y2[ids[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == 'Min':
            iou = inter / np.minimum(area[ids[-1]], area[ids[0:-1]])
        else:
            iou = inter / (area[ids[-1]] + area[ids[0:-1]] - inter)
        count +=1
        ids = ids[np.where(iou<threshold)[0]]
        # print(len(ids))
    #result_rectangle = boxes[pick].tolist()
    return pick,count