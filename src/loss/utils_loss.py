# -*- coding: utf-8 -*-
import torch
import math

def bbox_overlaps_diou(bboxes1, bboxes2):

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious,min=-1.0,max = 1.0)
    if exchange:
        dious = dious.T
    return dious

def bbox_overlaps_ciou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v)
    cious = iou - (u + alpha * v)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return cious

def bbox_overlaps_iou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    union = area1+area2-inter_area
    ious = inter_area / union
    ious = torch.clamp(ious,min=0,max = 1.0)
    if exchange:
        ious = ious.T
    return ious

def bbox_overlaps_giou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])

    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])

    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])

    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1+area2-inter_area
    closure = outer_area

    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious,min=-1.0,max = 1.0)
    if exchange:
        ious = ious.T
    return ious
