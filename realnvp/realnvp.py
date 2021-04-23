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

class RealNVPBlock(nn.Module):
    def __init__(self, in_channels, hid_channels, resblock_num):
        super(RealNVPBlock, self).__init__()
        before_squeeze_list = [
            CouplingLayer(in_channels, hid_channels, resblock_num, 'checkerboard', 'odd'),
            CouplingLayer(in_channels, hid_channels, resblock_num, 'checkerboard', 'even'),
            CouplingLayer(in_channels, hid_channels, resblock_num, 'checkerboard', 'odd')
        ]

        after_squeeze_list = [
            CouplingLayer(in_channels*4, hid_channels, resblock_num, 'channelwise', 'first'),
            CouplingLayer(in_channels*4, hid_channels, resblock_num, 'channelwise', 'second'),
            CouplingLayer(in_channels*4, hid_channels, resblock_num, 'channelwise', 'first'),
            CouplingLayer(in_channels*4, hid_channels, resblock_num, 'checkerboard', 'odd'),
            CouplingLayer(in_channels*4, hid_channels, resblock_num, 'checkerboard', 'even'),
            CouplingLayer(in_channels*4, hid_channels, resblock_num, 'checkerboard', 'odd'),
            CouplingLayer(in_channels*4, hid_channels, resblock_num, 'checkerboard', 'even')
        ]

        self.before_squeeze_layers = nn.Sequential(*before_squeeze_list)
        self.after_squeeze_layers = nn.Sequential(*after_squeeze_list)

    def squeeze(self, x):
        x1 = x[:,:,0::2,0::2]
        x2 = x[:,:,0::2,1::2]
        x3 = x[:,:,1::2,0::2]
        x4 = x[:,:,1::2,1::2]

        x = torch.cat((x1,x2,x3,x4),dim=1)

        return x

    def forward(self, x):
        x = self.before_squeeze_layers(x)
        x = self.squeeze(x)
        x = self.after_squeeze_layers(x)

        return x

class RealNVP(nn.Module):
    def __init__(self, in_dim, in_channels, hid_channels, resblock_num):
        super(RealNVP, self).__init__()
        self.recursion_num = int(torch.log2(torch.Tensor([in_dim])).item()) -1

        realnvp_block_list = [RealNVPBlock(in_channels*2**i, hid_channels, resblock_num) \
            for i in range(self.recursion_num)]
        self.realnvp_layers = nn.ModuleList(realnvp_block_list)

    def unsqueeze(self, x):
        B, C, H, W = x.size()
        C_div4 = C//4
        ret = torch.zeros(B, C//4, H*2, W*2)

        ret[:,:,0::2,0::2] = x[:,:C_div4]
        ret[:,:,0::2,1::2] = x[:,C_div4:2*C_div4]
        ret[:,:,1::2,0::2] = x[:,2*C_div4:3*C_div4]
        ret[:,:,1::2,1::2] = x[:,3*C_div4:4*C_div4]

        return ret

    def figure_out(self, x):
        B, C, H, W = x.size()
        half_channel = C//2
        x1 = x[:,:half_channel]
        x2 = x[:,half_channel:]

        return x1, x2

    def forward(self, x):
        fig_out_list = []
        for i in range(self.recursion_num):
            x = self.realnvp_layers[i](x)
            xf , x = self.figure_out(x)
            fig_out_list.append(xf)

        for i in range(self.recursion_num):
            xf = fig_out_list.pop()
            x = torch.cat((xf,x),dim=1)
            x = self.unsqueeze(x)

        return x