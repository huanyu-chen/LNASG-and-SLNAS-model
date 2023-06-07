#model_maker.py
#Target : The create model for final training or searching
#Func : model_maker , complex_model_maker , rebuilder
#

#Torch
import torch.nn as nn

#Python Standard Lib
import copy

#ours 
from model_arc.builder import layer_maker
from model_arc.operation_storage import DropPath_ , conv2d_std , conv1x1  ,activat_func,Mish
import model_arc.operation_storage as operation_storage
import numpy as np

operation_storage.own_activate = Mish()
class model_maker(nn.Module):
    """
    build SuperNet 

        forward : input data , img and return prediction 
        _build  : build whole supernet
        parameters_selection : collection all wegiht for rebuilder
        drop_path_prob : set drop path  probability
    """
    def __init__(self,cell_nums=5,out_filters=32,normal_block_repeat = [2,2,2],classes = 120,aux = True):
        #out_filters : the first layer's output size
        #cell_nums : the NAS search space , is node number
        #normal_block_repeat : the squance of the network structurec
        #total_layer : all layer's number include reduction block and normal block
        #block_bank : all possible operation's parameters
        #start_conv1 : 
        #fc : fully connect
        #aux_fc : aux loss's fully connect
        #aux ind: aux loss's index 
        super(model_maker, self).__init__()
        self.out_filters = out_filters
        self.cell_nums = cell_nums
        self.pool_num = len(normal_block_repeat)
        self.normal_block_repeat = normal_block_repeat
        self.total_layer = sum(normal_block_repeat) + self.pool_num
        self.block_bank = nn.ModuleList([])
        self.arc = None

        self._build()
        self.start_conv1 =  nn.Sequential(conv2d_std(in_channels = 3,out_channels = out_filters,kernel_size = 3),
                                          nn.BatchNorm2d( out_filters , track_running_stats=False))
        self.start_conv2 = nn.Sequential(conv2d_std(in_channels = 3,out_channels = out_filters,kernel_size = 5),
                                        nn.BatchNorm2d( out_filters , track_running_stats=False))
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                nn.Flatten(),
                                nn.Linear(out_filters*(2**(len(normal_block_repeat)-1))*2 ,classes ,bias=False))
        if aux == True :
            self.aux_fc = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                                    nn.Flatten(),
                                                    nn.Linear(out_filters*2**(i+1) ,classes ,bias=False)) for i in range(0,len(self.normal_block_repeat)-1,1)])
            self.aux_ind  = (np.cumsum(np.array(self.normal_block_repeat[:-1])+1)-1).tolist()
    def forward(self,arc,img):
        self.arc = arc
        layers_output = [self.start_conv1(img) , self.start_conv2(img)]
        #del img
        aux_pred = []
        for i in range(self.total_layer):
            arc_ = self.arc[self.block_bank[i][0].reduction]
            output = self.block_bank[i][0](arc_,layers_output[0],layers_output[1])
            layers_output = [layers_output[-1],output]
            if self.block_bank[i][0].reduction:
                layers_output[0] =self.block_bank[i][1](layers_output[0])
            if i in self.aux_ind:
                aux_pred.append(output)
        aux = [self.aux_fc[n](i) for n,i in enumerate(aux_pred)]
        del layers_output
        return self.fc(output),aux
    def _build(self):
        channel = self.out_filters
        for n,i in enumerate(self.normal_block_repeat,1):
            for j in range(i):
                self.block_bank.append(nn.ModuleList([layer_maker(self.cell_nums , channel , channel , False)]))
            if n < len(self.normal_block_repeat)+1:
                multiple = 2 if n == 3 else 2
                self.block_bank.append(nn.ModuleList([layer_maker(self.cell_nums ,  channel , channel*multiple , True),
                                                     nn.Sequential(conv1x1(in_channels=channel, out_channels=channel*multiple),
                                                                   #nn.BatchNorm2d( channel*multiple, track_running_stats=False),
                                                                   )]))
                channel *=multiple

    def parameters_selection(self , fix_arc ):
        assert (fix_arc != None) , "fix_arc must exit or use model_maker to find best arc"
        param = []
        for i in range(self.total_layer):
            arc_ = fix_arc[self.block_bank[i][0].reduction]
            temp = []
            temp.append(self.block_bank[i][0].param_select(arc_))
            if self.block_bank[i][0].reduction:
                temp.append(self.block_bank[i][1])
            param.append(temp)
        return copy.deepcopy(param) , (copy.deepcopy(self.start_conv1),copy.deepcopy(self.start_conv2)) , copy.deepcopy(self.fc) ,(copy.deepcopy(self.aux_fc),copy.deepcopy(self.aux_ind))
    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.p = p

class complex_model_maker(model_maker):
    """
    special case every block have different architecture sequence
    """
    def __init__(self,cell_nums=5,out_filters=32,normal_block_repeat = [2,2,2,2],classes=120,aux = True):
        #out_filters : the first layer's output size
        #cell_nums : the NAS search space , is node number
        #normal_block_repeat : the squance of the network structurec
        #total_layer : all layer's number include reduction block and normal block
        #block_bank : all possible operation's parameters
        #start_conv1 :
        #fc
        #aux_fc
        #aux ind
        super(model_maker, self).__init__()
        self.out_filters = out_filters
        self.cell_nums = cell_nums
        self.pool_num = len(normal_block_repeat)-1
        self.normal_block_repeat = normal_block_repeat
        self.total_layer = sum(normal_block_repeat) + self.pool_num
        self.block_bank = nn.ModuleList([])
        self.arc = None

        self._build()
        self.start_conv1 =  nn.Sequential(conv2d_std(in_channels = 3,out_channels = out_filters,kernel_size = 3),
                                          nn.BatchNorm2d( out_filters , track_running_stats=False))
        self.start_conv2 = nn.Sequential(conv2d_std(in_channels = 3,out_channels = out_filters,kernel_size = 5),
                                        nn.BatchNorm2d( out_filters , track_running_stats=False))
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                nn.Flatten(),
                                nn.Linear(out_filters*(2**(len(normal_block_repeat)-1))*2 ,classes ,bias=False))
        if aux == True :
            self.aux_fc = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Flatten(),
                                        nn.Linear(out_filters*(2**(len(normal_block_repeat)-1)//2) ,classes ,bias=False))
            self.aux_ind  = self.total_layer*2//3
    def forward(self,arc,img):
        self.arc = arc
        layers_output = [self.start_conv1(img) , self.start_conv2(img)]
        #print(layers_output[0].shape)
        #del img
        for i in range(0,self.total_layer):
            arc_ = self.arc[i]
            output = self.block_bank[i][0](arc_,layers_output[0],layers_output[1])
            layers_output = [layers_output[-1],output]
            if self.block_bank[i][0].reduction:
                layers_output[0] =self.block_bank[i][1](layers_output[0])
            if i == self.aux_ind:
                aux_pred = self.aux_fc(output)

        del layers_output
        return self.fc(output),aux_pred
    def parameters_selection(self , fix_arc ):
        assert (fix_arc != None) , "fix_arc must exit or use model_maker to find best arc"
        param = []
        for i in range(self.total_layer):
            arc_ = fix_arc[i]
            temp = []
            temp.append(self.block_bank[i][0].param_select(arc_))
            if self.block_bank[i][0].reduction:
                temp.append(self.block_bank[i][1])
            param.append(temp)
        return copy.deepcopy(param) , (copy.deepcopy(self.start_conv1),copy.deepcopy(self.start_conv2)) , copy.deepcopy(self.fc) ,(copy.deepcopy(self.aux_fc),copy.deepcopy(self.aux_ind))
