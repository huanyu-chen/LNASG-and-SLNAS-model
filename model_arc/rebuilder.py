#rebuilder.py
#from SuperNet get weight and rebuild CNN according architecture sequence
import torch
from torch import nn
import torch.nn.functional as F

from model_arc.operation_storage import  DropPath_ , cat_std ,add_std
from model_arc.builder import node , layer_maker
class node_from_inherit(node):
    """
    this node will inheritance SuperNet's weight

    """
    def __init__(self ,x_oper,y_oper,cat = False ):
        super(node, self).__init__()
        self.x = x_oper
        self.y = y_oper
        self.x_drop = DropPath_()
        self.y_drop = DropPath_()
        self.cat_add =  add_std() if cat == False else cat_std(axis=1)
    def forward(self,x,y):
        x = self.x(x)
        y = self.y(y)
        if x.numel() != y.numel():
            x,y =  self._rescale(x,y)
        return self.cat_add((self.x_drop(x),self.y_drop(y)))
    def _rescale(self,x,y):
        if x.shape[-1] > y.shape[-1] :
            y =F.interpolate(y, scale_factor=x.shape[-1] / y.shape[-1], mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=y.shape[-1] / x.shape[-1], mode='bilinear', align_corners=False)
        return x , y

class layer_maker_from_inherit(layer_maker):
    """
    this block will inheritance SuperNet's weight

    """
    def __init__(self,transfer_arc_weight ):
        super(layer_maker, self).__init__()
        self.cell_bank = nn.ModuleList()
        self.transfer_arc_weight,(self.final_conv,self.calibrate) = transfer_arc_weight

        self._build()
        del self.transfer_arc_weight
        self.inp = self.final_conv[3].in_channels // len(self.cell_bank)
    def forward(self,arc,x1,x2):
        self.input_shape_x = x2.shape[-1]
        x1 = F.interpolate(x1, size=(self.input_shape_x,self.input_shape_x) ,mode='bilinear', align_corners=False)
        layers = [x1,x2]
        #print(layers[0].shape, layers[1].shape)
        for n,model in enumerate(self.cell_bank):
            output = model(layers[arc[4*n]],layers[arc[4*n+2]])
            if ((self.calibrate !=None) & (output.shape[-1] !=self.input_shape_x//2)):
                output = F.interpolate(output, size=(self.input_shape_x//2,self.input_shape_x//2) ,mode='bilinear', align_corners=False)
            if  ((self.calibrate !=None) & (output.shape[1] != self.inp)):
                output =  self.calibrate(output)
            layers.append(output)
        layers = torch.cat(layers[2:],dim=1)
        return self.final_conv(layers)##self.dropout2
    def _build(self):
        for i in self.transfer_arc_weight:
            self.cell_bank.append(node_from_inherit(i[0],i[1],cat=False))
class model_rebuild(nn.Module):
    """
    create the final cnn according given architecture sequence 

    """
    def __init__(self,transfer_model_weight,fix_arc):
        super(model_rebuild, self).__init__()
        self.fix_arc = fix_arc
        self.layer_weight,(self.start_conv1, self.start_conv2 ),self.fc ,(self.aux_fc ,self.aux_ind)= transfer_model_weight
        self.block_bank = nn.ModuleList([])
        self._build()
    def forward(self,img):
        layers_output = [self.start_conv1(img) , self.start_conv2(img)]
        #del img
        aux_pred = [] 
        for i ,bank in enumerate(self.block_bank):
            arc = self.fix_arc[bank[0].calibrate != None]
            output = bank[0](arc ,layers_output[0],layers_output[1])
            layers_output = [layers_output[-1],output]
            if bank[0].calibrate != None:
                layers_output[0] =bank[1](layers_output[0])
            if i in self.aux_ind:
                aux_pred.append(output)
        aux = [self.aux_fc[n](i) for n,i in enumerate(aux_pred)]
                
        del layers_output
        return self.fc(output) , aux
    def _build(self):
        for layer in self.layer_weight:
            if len(layer) == 2:
                self.block_bank.append(nn.ModuleList([layer_maker_from_inherit(layer[0]),layer[-1]]))
            else:
                self.block_bank.append(nn.ModuleList([layer_maker_from_inherit(layer[0])]))
    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.p = p
class complex_model_rebuild(model_rebuild):
    """
    create the final cnn according given architecture sequence 
    special case every block have different architecture sequence
    """
    def __init__(self,transfer_model_weight,fix_arc):
        super(model_rebuild, self).__init__()
        self.fix_arc = fix_arc
        self.layer_weight,(self.start_conv1, self.start_conv2 ),self.fc ,(self.aux_fc ,self.aux_ind)= transfer_model_weight
        self.block_bank = nn.ModuleList([])
        self._build()
    def forward(self,img):
        layers_output = [self.start_conv1(img) , self.start_conv2(img)]
        #del img
        for i ,bank in enumerate(self.block_bank):
            arc = self.fix_arc[i]
            output = bank[0](arc ,layers_output[0],layers_output[1])
            layers_output = [layers_output[-1],output]
            if bank[0].calibrate != None:
                layers_output[0] =bank[1](layers_output[0])
            if i == self.aux_ind:
                aux_pred = self.aux_fc(output)
        del layers_output
        return self.fc(output) , aux_pred

