import torch
import torch.nn as nn
import torch.nn.functional as Function
import numpy as np
import cv2
from torch.autograd import Variable

class WMAE(nn.Module):
    def __init__(self): # t1 = perd , t2 = gt
        super(WMAE,self).__init__()

    def forward(self,t1,t2):
        # input : (3 ,num of data)
        w = [300,1,200]
        output = 0.
        if len(t2.shape) == 1:
            n_sample = t2.shape[0]
            n_feature = 1
            for i in range(n_sample):
                for j in range(n_feature):
                    output += w[j]*torch.abs(t1[i] - t2[i])
        else:
            n_sample = t2.shape[0]
            n_feature = t2.shape[1]
            for i in range(n_sample):
                for j in range(n_feature):
                    output += w[j]*torch.abs(t1[i][j] - t2[i][j])
        return torch.Tensor(output/n_sample)

class NAE(nn.Module):
    def __init__(self): # t1 = perd , t2 = gt
        super(NAE,self).__init__()

    def forward(self,pred_Y,gt_Y):
        # input : (3 ,num of data)
        output = 0.
        if len(gt_Y.shape) == 1:
            n_sample = gt_Y.shape[0]
            n_feature = 1
            for i in range(n_sample):
                for j in range(n_feature):
                    output += torch.abs(pred_Y[i] - gt_Y[i])/gt_Y[i]
        else:
            n_sample = gt_Y.shape[0]
            n_feature = gt_Y.shape[1]
            for i in range(n_sample):
                for j in range(n_feature):
                    output += torch.abs(pred_Y[i][j] - gt_Y[i][j])/gt_Y[i][j]
        return torch.Tensor(output/n_sample)
