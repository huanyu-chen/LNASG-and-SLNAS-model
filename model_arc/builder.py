#builder.py
#SuperNet's part include "node" , "block" , "operations"
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from model_arc.operation_storage import InvertedResidual  ,DropPath_, cat_std ,add_std, conv2d_std , conv1x1,activat_func,InvertedResidual_shuffle,Mish

#from model_arc.utils import batchnorm_modify
import copy
import model_arc.operation_storage as operation_storage
operation_storage.own_activate = Mish()#nn.ReLU6(inplace=True)

class node(nn.Module):
    """
    node 
    """
    def __init__(self ,input_,out,reduction = False,cat=False):
        super(node, self).__init__()
        self.x = cell(input_,out,reduction)
        self.y = cell(input_,out,reduction)
        self.x_drop = DropPath_()
        self.y_drop = DropPath_()
        self.cat_add =  add_std() if cat == False else cat_std(axis=1)

    def forward(self, arc ,x,y):
        x = self.x(arc[0],x)
        y = self.y(arc[1],y)
        if x.numel() != y.numel():
            x,y =  self._rescale(x,y)
        return self.cat_add((self.x_drop(x),self.y_drop(y)))
    def _rescale(self,x,y):
        if x.shape[-1] > y.shape[-1] :
            y =F.interpolate(y, scale_factor=x.shape[-1] / y.shape[-1], mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=y.shape[-1] / x.shape[-1], mode='bilinear', align_corners=False)
        return x , y
    def param_select(self,arc):
        #batchnorm_modify(self.x.all_oper[arc[0]])
        #batchnorm_modify(self.y.all_oper[arc[1]])
        return nn.ModuleList([copy.deepcopy(self.x.all_oper[arc[0]]),copy.deepcopy(self.y.all_oper[arc[1]])])
class cell(nn.Module):
    """
    operation unit
    """
    def __init__(self,nin,nout,reduction = False):
        super(cell, self).__init__()
        self.in_out = [nin,nout]
        self.all_oper =  nn.ModuleList([
                    #InvertedResidual(**{'in_channels':nin,'out_channels':nout,'kernel_size':3,'groups':nin,'stride':reduction+1, 'padding':3//2,'bias':False}),
                    #InvertedResidual(**{'in_channels':nin,'out_channels':nout,'kernel_size':5,'groups':nin,'stride':reduction+1, 'padding':5//2,'bias':False}),
                    InvertedResidual_shuffle(**{'inp':nin,'oup':nout,'kernel_size':3,'stride':reduction+1}),
                    InvertedResidual_shuffle(**{'inp':nin,'oup':nout,'kernel_size':5,'stride':reduction+1}),

                    nn.Sequential(conv1x1(in_channels = nin, out_channels = nout),
                                  nn.AvgPool2d(**{'kernel_size':3, 'stride':reduction+1, 'padding':1}),
                                  nn.BatchNorm2d(nout,track_running_stats=False)),
                    nn.Sequential(conv1x1(in_channels = nin, out_channels = nout),
                                  nn.MaxPool2d(**{'kernel_size':3, 'stride':reduction+1, 'padding':1}),
                                  nn.BatchNorm2d(nout,track_running_stats=False)),
                    nn.Sequential(conv1x1(in_channels = nin, out_channels = nout),
                                  nn.BatchNorm2d(nout,track_running_stats=False)
                                  )
                                        ])
    def forward(self, arc ,x):
        x = self.all_oper[arc](x)
        return x
class layer_maker(nn.Module):
    """
    block 
    """
    def __init__(self,cell_num,inputs_ ,output_, reduction ):
        super(layer_maker, self).__init__()
        self.cell_num = cell_num
        self.inp = inputs_
        self.opt = output_
        self.reduction = reduction
        self.cell_bank = nn.ModuleList([])
        self.final_conv = nn.Sequential(nn.BatchNorm2d(self.inp*self.cell_num,track_running_stats=False),
                                        activat_func(),
                                        SELayer(self.inp*self.cell_num),
                                        conv1x1(in_channels=self.inp*self.cell_num, out_channels=output_),
                                        #SELayer(output_)
                                        )
        if self.inp != self.opt:
            self.calibrate = nn.Sequential(conv1x1(in_channels=self.opt, out_channels=self.inp),
                                           #nn.BatchNorm2d( self.inp, track_running_stats=False),
                                           #activat_func()
                                           )

        self._build()
    def forward(self,arc,x1,x2):
        self.input_shape_x = x2.shape[-1]
        x1 = F.interpolate(x1, size=(self.input_shape_x,self.input_shape_x) ,mode='bilinear', align_corners=False)
        layers = [x1,x2]
        for n,model in enumerate(self.cell_bank):
            output = model([arc[4*n+1],arc[4*n+3]],layers[arc[4*n]],layers[arc[4*n+2]])
            if ((self.reduction ==1) & (output.shape[-1] !=self.input_shape_x//2)):
                output = F.interpolate(output, size=(self.input_shape_x//2,self.input_shape_x//2) ,mode='bilinear', align_corners=False)
            if  ((self.reduction ==1) & (output.shape[1] != self.inp)):
                output =  self.calibrate(output)
            layers.append(output)
        cat_out = torch.cat(layers[2:],dim=1)
        return self.final_conv(cat_out)
    def _build(self):
        for i in range(0,self.cell_num):
            self.cell_bank.append(node(self.inp ,self.opt,reduction = self.reduction))
    def param_select(self,arc):
        temp = nn.ModuleList()
        for n,model in enumerate(self.cell_bank):
            temp.append(nn.ModuleList(model.param_select([arc[4*n+1],arc[4*n+3]])))
        return copy.deepcopy(temp) , (copy.deepcopy(self.final_conv) , None if self.inp == self.opt else copy.deepcopy(self.calibrate))

class SELayer(nn.Module):
    #SE modual 
    #https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)