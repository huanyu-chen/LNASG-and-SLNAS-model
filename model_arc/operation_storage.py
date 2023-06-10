#operation_storage.py
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn.parallel
import numpy as np

#global can use for change all activate function 
own_activate = nn.ReLU6(inplace=True)

class cat_std(nn.Module):
    """
    cat operation 
    """
    def __init__(self,axis = 0):
        super().__init__()
        self.axis = axis
    def forward(self,cat_list):
        return torch.cat(cat_list, self.axis)
class add_std(nn.Module):
    """
    add operation 
    """
    def __init__(self):
        super().__init__()
    def forward(self,add_list):
        return torch.add(*add_list)
class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        self.drop_path_(x, self.p, self.training)
        return x
    def drop_path_(self,x, drop_prob, training):
        if training and drop_prob > 0.:
            keep_prob = 1. - drop_prob
            # per data point mask; assuming x in cuda.
            mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
            x.div_(keep_prob).mul_(mask)
        return x


#https://medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76
#https://github.com/rwightman/pytorch-image-models/commit/7ac6db4543a07d44c6b30327a9e8fe31a4ee8e08
# below Swish & Mish implement is faster than normal 
class SwishAutoFn(torch.autograd.Function):
        """Swish - Described in: https://arxiv.org/abs/1710.05941
        Memory efficient variant from:
         https://medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76
        """
        @staticmethod
        def forward(ctx, x):
            result = x.mul(torch.sigmoid(x))
            ctx.save_for_backward(x)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_variables[0]
            sigmoid_x = torch.sigmoid(x)
            return grad_output.mul(sigmoid_x * (1 + x * (1 - sigmoid_x)))
class Swish(nn.Module):
    def forward(self, input_tensor):
        return SwishAutoFn.apply(input_tensor)
class MishAutoFn(torch.autograd.Function):
        """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
        Experimental memory-efficient variant
        """

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            y = x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))
            return y

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_variables[0]
            x_sigmoid = torch.sigmoid(x)
            x_tanh_sp = F.softplus(x).tanh()
            return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))

class Mish(nn.Module):
    def forward(self, input_tensor):
        return MishAutoFn.apply(input_tensor)

class activat_func(nn.Module):
    def __init__(self):
        super().__init__()
        global own_activate
        self.target_func = own_activate

    def forward(self, x):
        return self.target_func(x)

class conv2d_std(nn.Module):
    """
    standard conv2d when stride = 1the the output feature map w , h is same as input w,h
    """
    def __init__(self,in_channels,out_channels,kernel_size,stride = 1,dilation=1, groups=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = stride, padding = kernel_size//2, bias = False,padding_mode='reflect',
                              dilation = dilation, groups = groups)
    def forward(self, x):
        return self.conv(x)
class conv1x1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = 1,
                              stride=1 , padding = 0,bias = False,padding_mode='zeros',
                              dilation = 1,groups = 1)
    def forward(self, x):
        return self.conv(x)
class InvertedResidual(nn.Module):
    """
    mobilenet V2 
    """

    def __init__(self, in_channels, out_channels, kernel_size,stride,padding ,groups,bias):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        self.hidden_dim = int(in_channels * 3)

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels = in_channels,out_channels =self.hidden_dim*2,kernel_size = 1,stride=1, padding=0),
            nn.BatchNorm2d(self.hidden_dim*2,track_running_stats=False),
            nn.ReLU6(inplace=True),
            #Mish(),
            # dw
            nn.Conv2d(in_channels=self.hidden_dim*2, out_channels = self.hidden_dim*2, kernel_size=kernel_size, stride=stride,padding= padding, groups=self.hidden_dim, bias=bias),
            nn.BatchNorm2d(self.hidden_dim*2,track_running_stats=False),
            #Mish(),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(in_channels = self.hidden_dim*2,out_channels =out_channels,kernel_size = 1,stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
       # x = x + self.convF(x)

        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
class Interpolate(nn.Module):
    """
    resize features map 
    """
    def __init__(self, scale_factor=0.5, mode = 'bilinear'):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


def channel_shuffle(x, groups):
    """
    shufflenet's shuffle implement 
    """
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class InvertedResidual_shuffle(nn.Module):
    """
    ShufleNet modual implement if stride = 2 reduction mode , stride=1 normal mode
    """
    def __init__(self, inp, oup, kernel_size ,stride):
        super(InvertedResidual_shuffle, self).__init__()
        self.benchmodel =2 if stride == 2 else 1
        self.stride = stride
        self.kernel_size = kernel_size
        assert stride in [1, 2]

        oup_inc = oup//2
        self.mish = Mish()
        if self.benchmodel == 1:
            #assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                #nn.ReLU(inplace=True),
                Mish(),
                # dw
                nn.Conv2d(oup_inc, oup_inc, self.kernel_size, stride, self.kernel_size//2, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                #nn.ReLU(inplace=True),
                Mish(),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, self.kernel_size, 1, self.kernel_size//2, groups=inp, bias=False),
                Downsample(channels=inp, filt_size=3, stride=stride),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                #nn.ReLU(inplace=True),
                Mish(),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                #nn.ReLU(inplace=True),
                Mish(),
                # dw
                nn.Conv2d(oup_inc, oup_inc, self.kernel_size, 1, self.kernel_size//2, groups=oup_inc, bias=False),
                Downsample(channels=oup_inc, filt_size=3, stride=stride),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                #nn.ReLU(inplace=True),
                Mish(),
            )
            #self.downsample = Downsample(channels=inp, filt_size=3, stride=2)

    def _concat(self,x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            #
            # x2 = self.mish(x2+self.banch2(x2))
            # out = self._concat(x1, x2)
            out = self._concat(x1, self.banch2(x2))
        else:
            #
            # x_rescale = self.downsample(x)#F.avg_pool2d(x, kernel_size = 3, stride = 2, padding = 1)
            # x1 = self.mish(x_rescale+self.banch1(x))
            # x2 = self.mish(x_rescale+self.banch2(x))
            # out = self._concat(x1, x2)
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)

#below is from https://github.com/adobe/antialiased-cnns
class Downsample(nn.Module):
    """
    MaxBlur implement 
    from paper 
        Making Convolutional Networks Shift-Invariant Again , https://arxiv.org/abs/1904.11486

    """
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class Downsample1D(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer
