"""
RealNVP model

Class
1. ResBlock
    define 1 residual block

    Input: parameters of conv layer
    Output: x = f(x) + x

2. ScalingNetwork
    define scaling network s
    cifar10: 8 residual blocks, 64 feature maps, downscale once
    others: same with README.md
    use tanh activation to last layer

    Input: resblock_num, parameters of ResBlock
    Output: output of scaling netowrk

3. TransformNetwork
    define transform network
    cifar10: 8 residual blocks, 64 feature maps, downscale once
    others: same with README.md
    use affine output

    Input: resblock_num, parameters of ResBlock
    Output: output of transform netowrk

4. CouplingLayer
    define coupling layer
    3 checkerboard masking(B*C*N*N) -> 
    squeezing -> 
    3 channel-wise masking(B*4C*N/2*N/2) -> 
    4 checkerboard masking(B*4C*N/2*N/2)

5. WeightNormConv2d
    Conv2d layer with weightnorm

6. RealNVP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(WeightNormConv2d, self).__init__()

        self.weight_norm_conv2d = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

    def forward(self, x):
        return self.weight_norm_conv2d(x)


class ResBlock(nn.Module):
    def __init__(self, hid_channels):
        super(ResBlock, self).__init__()

        self.res_block = nn.Sequential(
            WeightNormConv2d(hid_channels, hid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(),
            WeightNormConv2d(hid_channels, hid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hid_channels)
        )
        
    def forward(self, x):
        x = self.res_block(x) + x
        x = F.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, in_channels, hid_channels, resblock_num):
        super(ResNet, self).__init__()

        self.preprocessing_block = nn.Sequential(
            WeightNormConv2d(in_channels, hid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU()
        )

        resblock_list = [ResBlock(hid_channels) for _ in range(resblock_num)]
        self.resblocks = nn.Sequential(*resblock_list)

        self.postprocessing_block = \
            WeightNormConv2d(hid_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.preprocessing_block(x)
        x = self.resblocks(x)
        x = self.postprocessing_block(x)

        return x

class ScalingNetwork(nn.Module):
    def __init__(self, in_channels, hid_channels, resblock_num):
        super(ScalingNetwork, self).__init__()

        self.scaling_layer = ResNet(in_channels, hid_channels, resblock_num)

    def forward(self, x):
        x = self.scaling_layer(x)
        x = torch.tanh(x)

        return x

class TransformNetwork(nn.Module):
    def __init__(self, in_channels, hid_channels, resblock_num):
        super(TransformNetwork, self).__init__()

        self.transform_layer = ResNet(in_channels, hid_channels, resblock_num)

    def forward(self, x):
        x = self.transform_layer(x)

        return x

class CouplingLayer(nn.Module):
    def __init__(self, in_channels, hid_channels, resblock_num, mask_name, immobile):
        super(CouplingLayer, self).__init__()

        self.mask_name = mask_name
        self.immobile = immobile

        self.scaling_layer = ScalingNetwork(in_channels, hid_channels, resblock_num)
        self.transform_layer = TransformNetwork(in_channels, hid_channels, resblock_num)
    
    def make_mask(self, input_x, mask_name, immobile):
        B, C, H, W = input_x.size()

        if mask_name == 'checkerboard':
            mask_odd = torch.zeros(1,W)
            mask_odd[:,1::2] = 1
            mask_even = torch.zeros(1,W)
            mask_even[:,0::2] = 1
            mask = torch.Tensor([])

            if immobile == 'odd':    
                for _ in range(int(H/2)):
                    mask = torch.cat((mask, mask_odd, mask_even))
                if H % 2 == 1:
                    mask = torch.cat((mask, mask_odd))
            elif immobile == 'even':
                for _ in range(int(H/2)):
                    mask = torch.cat((mask, mask_even, mask_odd))
                if H % 2 == 1:
                    mask = torch.cat((mask, mask_even))
        elif mask_name == "channelwise":
            half_channel = int(C/2)
            mask_one = torch.ones(half_channel, H, W)
            mask_zero = torch.zeros(C-half_channel, H, W)
            if immobile == 'first':
                mask = torch.cat((mask_one, mask_zero))
            elif immobile == 'second':
                mask = torch.cat((mask_zero, mask_one))

        return mask

    def forward(self, x):
        b = self.make_mask(x, self.mask_name, self.immobile)
        s = self.scaling_layer(b*x)
        t = self.transform_layer(b*x)
        x = b*x + (1-b)*(x*torch.exp(s)+t)

        return x
