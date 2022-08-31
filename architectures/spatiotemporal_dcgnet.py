import math

import torch.nn as nn
from torch.nn.modules.utils import _triple


##### ST_DCGNet
class ST_DCGNet(nn.Module):
    def __init__(self, n_channel, n_z, sizing=1):
        super(ST_DCGNet, self).__init__()
        
        self.name = 'ST_DCGNet'
        
        # number of output channel
        self.n_channel=n_channel
        # size of 1d - white noise/latent space
        self.n_z=n_z
        # multiplying factor for the number of features maps
        self.sizing=sizing
        
        self.main = nn.Sequential(
            unsqz(),
            
            nn.ConvTranspose3d(self.n_z, 128, (2,8,8), 1, 0, bias=False),
            
            nn.Upsample(scale_factor = (2,2,2), mode='trilinear', align_corners=False),
            SpatioTemporalConv(128,64,3),

            nn.Upsample(scale_factor = (2,2,2), mode='trilinear', align_corners=False),
            SpatioTemporalConv(64,32,3),

            nn.Upsample(scale_factor = (2,2,2), mode='trilinear', align_corners=False),
            SpatioTemporalConv(32,16,3),
            
            nn.Upsample(scale_factor = (2,2,2), mode='trilinear', align_corners=False),
            SpatioTemporalConv_nobnr(16,1,3),
            
            sqz(),
            sqz(),
        )

    def forward(self, input):
        return self.main(input)
    
#### utils
class unsqz(nn.Module):
    def forward(self, input):
        return input.unsqueeze(0)
    
class pshape(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input
    
class sqz(nn.Module):
    def forward(self, input):
        return input.squeeze(0)

##### Weight Init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

##### Spatio Temporal Conv R(2+1)D
## https://github.com/irhum/R2Plus1D-PyTorch/blob/master/module.py
## https://arxiv.org/abs/1711.11248

class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input 
    planes with distinct spatial and time axes, by performing a 2D convolution over the 
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time 
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x
    


class SpatioTemporalConv_nobnr(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super(SpatioTemporalConv_nobnr, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        #self.bn = nn.BatchNorm3d(intermed_channels)
        #self.relu = nn.ReLU()
        
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        #x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        return x