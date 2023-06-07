import os
import sys
import time
import random
import glob
import numpy as np


import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F

from data.data import get_loaders , get_retrain_loaders,AID_retrain_loader
from search_arc.hierarchical_controller import  separable_LSTM #,LSTM_index , LSTM_operation
from tqdm import tqdm
from model_arc.model_maker import model_maker
from model_arc.rebuilder import model_rebuild
import utils
from model_arc.operation_storage import conv2d_std , Mish , Downsample


model = model_maker(cell_nums=5,out_filters=40,normal_block_repeat = [2,2],classes = 1000,aux = True)
#dag = [[0, 3, 0, 1, 1, 4, 0, 1, 0, 1, 0, 4, 1, 3, 0, 1, 0, 1, 3, 3], [0, 1, 0, 1, 1, 4, 1, 1, 1, 0, 1, 3, 0, 3, 0, 2, 1, 2, 1, 4]]#aid_20 , small  slnas
#dag = [[0, 4, 0, 1, 0, 4, 1, 1, 0, 1, 3, 4, 1, 0, 0, 3, 1, 1, 0, 4],[0, 3, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 4, 1, 0, 1, 1, 1, 4]]#prob
#dag = [[0, 2, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 1, 1, 0, 0, 0, 3, 1, 3], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 3, 4, 1, 4, 0, 4, 2, 0, 1, 1]]#lstm small
dag = [[0, 2, 0, 3, 0, 3, 1, 1, 1, 3, 1, 0, 0, 1, 1, 3, 0, 1, 1, 0], [1, 1, 0, 3, 0, 1, 0, 1, 1, 2, 1, 3, 1, 0, 1, 0, 1, 1, 0, 3]]#lstm aid 50
#dag = [[0, 1, 1, 1, 1, 1, 2, 4, 0, 3, 0, 0, 1, 1, 4, 3, 1, 0, 5, 2], [1, 4, 1, 0, 1, 1, 1, 0, 0, 4, 1, 1, 0, 1, 1, 1, 1, 0, 1, 3]]#new my lstm
weight = model.parameters_selection(dag)
del model
model = model_rebuild(weight,dag)
model.start_conv1 =  nn.Sequential(conv2d_std(in_channels = 3,out_channels = 40,kernel_size = 3,stride=2),
                                      nn.BatchNorm2d( 40 , track_running_stats=False),Mish(),conv2d_std(in_channels = 40,out_channels = 40,kernel_size = 3,stride=2),
                                      nn.BatchNorm2d( 40 , track_running_stats=False),Mish())
model.start_conv2 = nn.Sequential(conv2d_std(in_channels = 3,out_channels = 40,kernel_size = 3,stride=2),
                                      nn.BatchNorm2d( 40, track_running_stats=False),Mish(),Downsample(channels=40, filt_size=3, stride=2))
def BatchNorm2d_replace(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, child_name, nn.BatchNorm2d(child.num_features,affine=True))
        else:
            BatchNorm2d_replace(child)
BatchNorm2d_replace(model)
del weight

model.cpu()

model.eval()
start = time.time()

for i in range(5):
    with torch.no_grad():
        model(torch.randn(32,3,224,224).cpu())
totol_time = time.time() - start
print(totol_time/32/5)
print(totol_time/5)



import torch
import time
from tqdm import tqdm
from torchvision import models

model =  models.vgg16(False).cpu()
model.eval()


start = time.time()

for i in range(5):
    with torch.no_grad():
        model(torch.randn(32,3,224,224).cpu())
totol_time = time.time() - start
print('vgg16')
print(totol_time/32/5)
print(totol_time/5)

model =  models.shufflenet_v2_x1_0().cpu()
model.eval()
start = time.time()

for i in range(5):
    with torch.no_grad():
        model(torch.randn(32,3,224,224).cpu())
totol_time = time.time() - start
print('shufflenet_v2_x1_0')
print(totol_time/32/5)
print(totol_time/5)
