import time
import cv2
import torch
from torch2trt import TRTModule
from torch2trt import torch2trt

import os
from utils.config import Config
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from data.constant import culane_row_anchor
 


if __name__ == '__main__': 
    print('Load  LaneDetection model...')

    cfg = Config.fromfile('/home/r320/ComputerVisionADASProject/detections/Ultra/configs/culane.py')
    cfg.test_model = '/home/r320/ComputerVisionADASProject/detections/Ultra/weights/culane_18.pth'

    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    
    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError
     
    
    net = parsingNet(pretrained = False, backbone=cfg.backbone, \
            cls_dim = (cfg.griding_num+1,cls_num_per_lane,cfg.num_lanes), use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    model = net


    x = torch.ones([1, 3, 288, 800]).cuda()
    model_trt = torch2trt(model, [x], fp_16=True)
	
    torch.save(model_trt.state_dict(), '/home/r320/ComputerVisionADASProject/detections/Ultra/weights/culane_18_fp16.pth')

    model = TRTModule()
    model.load_state_dict(torch.load('/home/r320/ComputerVisionADASProject/detections/Ultra/weights/culane_18_fp16.pth'))
    model.cuda()
    print('Convert and Test done')