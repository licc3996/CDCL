from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn
import torch
import torch.nn as nn
import torch.fft
import torchvision as tv
import torchvision
from functools import reduce
from torchvision import datasets, models, transforms
#from models.cycle_mlp import CycleBlock
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from torch.nn.modules.utils import _pair
import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv
#from timm.models import resnet
#from timm.models import resnet
#from models.dla import dla34, dla102
from einops.layers.torch import Rearrange
from models.focal_net import FocalNetBlock
from models.maxvit import MBConv
from models.dynamicConv import DynamicConvBlock, CTXDownsample  
class SEModule(nn.Module):
    def __init__(self, dim, red=8):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, 1),
            nn.GELU(),
            nn.Conv2d(inner_dim, dim, 1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Conv2d(dim, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        channel_attn = self.attn(x)
        spatial_attn = torch.sigmoid(self.spatial_attn(x))
        return x * channel_attn * spatial_attn  
class Vec2Patch(nn.Module):
    def __init__(self, channel, hidden, output_size, kernel_size, stride, padding):
        super(Vec2Patch, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.to_patch = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding)
        h, w = output_size

    def forward(self, x):
        feat = self.embedding(x)
        b, n, c = feat.size()
        feat = feat.permute(0, 2, 1)
        feat = self.to_patch(feat)

        return feat

class Backbone(nn.Module):
    def __init__(self, resnet):
        super().__init__()
       
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.se1 = SEModule(256, red=8)
        self.layer2 = resnet.layer2
        self.se2 = SEModule(512, red=4)
        self.layer3 = resnet.layer3
        self.se3 = SEModule(1024, red=4)
       
        self.dynamic_blocks = nn.ModuleList([
            DynamicConvBlock(
                dim=1024,          
                ctx_dim=512,     
                kernel_size=7,    
                smk_size=5,      
                num_heads=8,      
                mlp_ratio=4,      
                drop_path=0.1,    
                use_gemm=True,
                ls_init_value=1,
                res_scale=True   
            ),
            DynamicConvBlock(
                dim=1024,
                ctx_dim=512,
                kernel_size=7,
                smk_size=5,
                num_heads=8,
                mlp_ratio=4,
                drop_path=0.1,
                use_gemm=True,
                ls_init_value=1,
                res_scale=True 
            )
        ])
        
        self.ctx_adapter = nn.Sequential(
            nn.Conv2d(512, 512, 1), 
            nn.BatchNorm2d(512)
        )
        self.ctx_downsample = CTXDownsample(1024, 512)
        self.proj1 = nn.Conv2d(256, 1024, 1)   
        self.proj2 = nn.Conv2d(512, 1024, 1)
        self.out_channels = 1024

    def forward(self, x):
      
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        se1_out = self.se1(x)
        
        x = self.layer2(se1_out)
        se2_out = self.se2(x)
        
        x = self.layer3(se2_out)
        se3_out = self.se3(x)
        
        
        target_size = se3_out.shape[2:]
        feat1 = F.interpolate(self.proj1(se1_out), size=target_size, mode='bilinear', align_corners=False)
        feat2 = F.interpolate(self.proj2(se2_out), size=target_size, mode='bilinear', align_corners=False)
        enhanced_context = se3_out + feat1 + feat2  
        ctx = self.ctx_downsample(enhanced_context)
        ctx = self.ctx_adapter(ctx) 
        x = se3_out
        for block in self.dynamic_blocks:
            x, ctx = block(x, ctx, ctx) 
        return OrderedDict([["feat_res4", x]])

"""class Res5Head(nn.Module): 
    def __init__(self, resnet):
        super().__init__()
        self.layer4 = resnet.layer4
        self.context_refine = DynamicConvBlock(
                dim=2048,
                ctx_dim=1024,
                kernel_size=5,
                smk_size=3,
                num_heads=16,
                mlp_ratio=2,
                is_last=True
            )
        self.out_channels = [1024, 2048]

    def forward(self, x):
      
        base_feat = self.layer4(x)  
        x = F.adaptive_max_pool2d(base_feat, 1)
        feat = F.adaptive_max_pool2d(x, 1)
        return OrderedDict([
            ["feat_res4", x],
            ["feat_res5", feat]
        ])"""
class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__()  # res5
        #self.res5feat=OrderedDict([["layer4", resnet.layer4]])
        #self.layer4 = nn.Sequential(resnet.layer4)
        self.out_channels = [1024, 2048]
        hidden = 256
        output_size = (14,14)
        #self.mlP_model = MLPMixer(in_channels=256, image_size=14, patch_size=1)
        # self.sc_mlp=MLPMixer(
        # input_size = (14,14),
        # patch_size = (1,14),
        # dim = 256)
        #self.simam = SimAM()
        self.focalNet = FocalNetBlock(dim=hidden, input_resolution=196)
        self.norm = nn.BatchNorm2d(hidden)
        self.qconv1 = nn.Conv2d(in_channels=1024, out_channels=hidden, kernel_size=1)
        self.qconv2 = nn.Conv2d(in_channels=hidden, out_channels=1024, kernel_size=1)
        #self.mb_conv = MBConv(in_channels=d_model, out_channels=d_model)
        self.patch2vec = nn.Conv2d(1024, hidden, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.vec2patch = Vec2Patch(1024, hidden, output_size, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        
        self.mbconv = MBConv(hidden, hidden)
        self.final_in = nn.Conv2d(in_channels=1024, out_channels=hidden, kernel_size=1)
        self.final_in2 = nn.Conv2d(in_channels=hidden, out_channels=1024, kernel_size=1)
        self.final_out = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)
        
                
    def forward(self, x):
        input = x
        b, c, h, w = x.size()
        final_in = self.norm(self.final_in(x))
        
        #mbconv=self.mbconv(final_in)
        final_in2 = self.final_in2(final_in)
        
        trans_feat = self.patch2vec(final_in2)

        _, c, h, w = trans_feat.size()
        trans_feat = trans_feat.view(b, c, -1).permute(0, 2, 1)
        
        x_focal_feat=self.focalNet(trans_feat)
        trans_feat = self.vec2patch(x_focal_feat)  + final_in2
        
        final_out = self.final_out(trans_feat)
 
    
        x_feat = F.adaptive_max_pool2d(trans_feat, 1)

        feat = F.adaptive_max_pool2d(final_out, 1)
        trans_features = {}
        trans_features["feat_res4"] = x_feat
        trans_features["feat_res5"] = feat
        return trans_features
    
        



def build_resnet(name="resnet50", weights=torchvision.models.ResNet50_Weights.DEFAULT):
    resnet = torchvision.models.resnet.__dict__[name](weights=weights)
    
    
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)
    
    
    return Backbone(resnet), Res5Head(resnet)


if __name__ == '__main__':
    backbone, head = build_resnet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    head = head.to(device) 
    dummy_input = torch.randn(5, 3, 224, 224)
    dummy_input = dummy_input.to(device)
   
    features = backbone(dummy_input)
    print("Backbone_out:", features['feat_res4'].shape)  
    
    outputs = head(features['feat_res4'])
    print("Head_out:")
    print("feat_res4:", outputs['feat_res4'].shape)  
    print("feat_res5:", outputs['feat_res5'].shape) 
    
    
    loss = outputs['feat_res4'].mean() + outputs['feat_res5'].mean()
    loss.backward()
    
   
    assert backbone.conv1.weight.grad is None, "Conv1quanzhongyinggaibeidongjie"
    assert backbone.bn1.weight.grad is None, "BN1quanzhongyinggaibeidongjie"
    print("tidujianchatongguo")