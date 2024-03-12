import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math
from einops import rearrange

class DCNv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1):

        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        # init weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # offset conv
        self.conv_offset_mask = nn.Conv2d(in_channels, 
                                          3 * kernel_size * kernel_size,
                                          kernel_size=kernel_size, 
                                          stride=stride,
                                          padding=self.padding, 
                                          bias=True)
        
        # init        
        self.reset_parameters()
        self._init_weight()

    def reset_parameters(self):
        n = self.in_channels * (self.kernel_size**2)
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()


    def _init_weight(self):
        # init offset_mask conv
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)


    def forward(self, x, offset):
        out = self.conv_offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.weight, 
                                          bias=self.bias, 
                                          #mask=mask,
                                          padding=self.padding,
                                          stride=self.stride)
        return offset, x


class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                       kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, inputs):
        '''
        :param inputs: (T, B, C, H, W)
        :param hidden_state: (hx: (B, C, H, W), cx: (B, C, H, W))
        :return:
        '''
        outputs = []
        T, B, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(T):
            combined = torch.cat([inputs[t,:,:,:,:], # (B, C, H, W)
                                  hx], dim=1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs) # (T, B, C, H, W) 
    
class Channel_Spatial_Attention(nn.Module):
    def __init__(self, inChannels, reduction=16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.inChannels = inChannels
        self.channel_weight_max = nn.Sequential(*[
            nn.Conv2d(inChannels*2, inChannels*2//reduction, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(inChannels*2//reduction, inChannels*2, kernel_size=1, padding=0, stride=1)])
        
        self.channel_weight_avg = nn.Sequential(*[
            nn.Conv2d(inChannels*2, inChannels*2//reduction, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(inChannels*2//reduction, inChannels*2, kernel_size=1, padding=0, stride=1)])
        
        self.spatial_weight = nn.Sequential(*[
            nn.Conv2d(inChannels*2, 2, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()])
    
    def forward(self, feat_rgb, feat_ev):
        feat_cat = torch.cat((feat_rgb, feat_ev), dim=1)
        # channel attention
        channel_avg = self.avg_pool(feat_cat)
        channel_max = self.max_pool(feat_cat)
        channel_avg = self.channel_weight_avg(channel_avg)
        channel_max = self.channel_weight_max(channel_max)
        channel_weight = torch.sigmoid(channel_avg + channel_max)

        feat_rgb = feat_rgb * channel_weight[:,:self.inChannels,:,:]
        feat_ev = feat_ev * channel_weight[:,self.inChannels:,:,:]
        # spatial attention
        feat_cat = torch.cat((feat_rgb, feat_ev), dim=1)
        spatial_weight = self.spatial_weight(feat_cat)
        feat_rgb = feat_rgb * spatial_weight[:,0:1,:,:]
        feat_ev = feat_ev * spatial_weight[:,1:2,:,:]

        return feat_rgb, feat_ev


# Spatial Alignment and complementary Fusion Block
class SAFusionBlock(nn.Module):
    def __init__(self, inChannels, outChannels, radius=5, recurrent=False):
        super(SAFusionBlock, self).__init__()
        self.radius = radius
        self.recurrent = recurrent
        
        if recurrent:
            print('Using recurrent offset calculation')
            self.cal_offset = nn.Sequential(*[
            ConvLSTMBlock(2 * inChannels + (2*radius+1)**2, inChannels, kernel_size=3, padding=1, stride=1),
            ConvLSTMBlock(inChannels, inChannels, kernel_size=3, padding=1, stride=1)])
        else:
            self.cal_offset = nn.Sequential(*[
                nn.Conv2d(2 * inChannels + (2*radius+1)**2, inChannels, kernel_size=5, padding=2, stride=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1, stride=1)]
                                            )
        self.DCN = DCNv2(inChannels, outChannels, 3, padding=1)

        self.channel_spatial_attention = Channel_Spatial_Attention(inChannels)
        self.fusion = nn.Sequential(*[
            nn.Conv2d(inChannels*2, outChannels, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1, stride=1)])
        
    def forward(self, Fi, Fe, time_step=5):
        b,c,h,w = Fi.shape

        feat_ev, feat_rgb = Fe, Fi
        # cost volume
        Fi_unfold = F.unfold(feat_rgb, kernel_size=self.radius*2+1, padding=self.radius)
        Fi_unfold = Fi_unfold.view(b,c,-1,h,w)
        Fi_unfold = Fi_unfold.permute(0,3,4,2,1)
        Fe_ = feat_ev.permute(0,2,3,1)
        cost_volume = torch.einsum('bhwc,bhwdc->bhwd', Fe_, Fi_unfold)
        cost_volume = cost_volume.permute(0,3,1,2)
        cost_volume /=  torch.sqrt(torch.tensor(c).float())

        # spatial alignment
        if self.recurrent:
            x = torch.cat((feat_rgb, feat_ev, cost_volume), 1)
            x = rearrange(x, '(t b) c h w -> t b c h w', t=time_step)
            offset = self.cal_offset(x)
            offset = rearrange(offset, 't b c h w -> (t b) c h w')
        else:
            offset = self.cal_offset(torch.cat((feat_rgb, feat_ev, cost_volume), 1))

        # deformable convolution
        offset_out, aligned_rgb = self.DCN(feat_rgb, offset)
        
        # channel spatial attention
        aligned_rgb, feat_ev = self.channel_spatial_attention(aligned_rgb, feat_ev)
        
        # fusion
        feat_cat = torch.cat((aligned_rgb, feat_ev), dim=1)
        Ff = self.fusion(feat_cat)

        return Ff
    
    