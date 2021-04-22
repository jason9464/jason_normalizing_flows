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
"""
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
            

