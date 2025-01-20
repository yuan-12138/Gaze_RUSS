import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
     
class DoubleConv2d(nn.Module):
    def __init__(self,in_channels,features):
        super(DoubleConv2d, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
                # nn.GroupNorm(num_groups=int(features/2), num_channels=features),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                # nn.GroupNorm(num_groups=int(features/2), num_channels=features),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
                )

    def forward(self,x):
        return self.block(x)
    
class UpConv2d_s2(nn.Module):
    def __init__(self,in_channels,features):
        super(UpConv2d_s2, self).__init__()
        self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, features, kernel_size=2, stride=2),
                # nn.GroupNorm(num_groups=int(features/2), num_channels=features),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
                )

    def forward(self,x):
        return self.block(x)
    
class Res_block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Res_layer(nn.Module):
    def __init__(self, inplanes, planes, blocks, stride=1):
        super(Res_layer, self).__init__()
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(Res_block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(Res_block(planes, planes))
        
        self.res_layer = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.res_layer(x)
    
class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias, stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x): 
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

class Trans_Att(nn.Module):
    
    def __init__(self, in_channels_x, in_channels_m, heads=4, dim_head=64, reduce_size=16, projection='interp'):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
       
        self.to_kv = depthwise_separable_conv(in_channels_m, self.inner_dim*2)
        self.to_q = depthwise_separable_conv(in_channels_x, self.inner_dim)
        self.to_out = depthwise_separable_conv(self.inner_dim, in_channels_x)

    def forward(self, q, x):

        B, C, H, W = x.shape # low-res feature shape
        BH, CH, HH, WH = q.shape # high-res feature shape

        k, v = self.to_kv(x).chunk(2, dim=1) #B, inner_dim, H, W
        q = self.to_q(q) #BH, inner_dim, HH, WH

        if self.projection == 'interp' and H != self.reduce_size:
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=HH, w=WH)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
       
        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HH, w=WH, dim_head=self.dim_head, heads=self.heads)

        out = self.to_out(out)

        return out, q_k_attn


class Gaze_Net_trans(nn.Module):
    def __init__(self, in_channels=1, init_features=32, blocks_I=2, blocks_M=2, num_heads=[2, 4, 8, 16]):
        super(Gaze_Net_trans, self).__init__()

        features = init_features

        self.res_layer_M_1 = Res_layer(in_channels, features, blocks=blocks_M, stride=1)  # 1->64
        self.pool_M_1 = nn.AvgPool2d(2, 2)
        self.res_layer_M_2 = Res_layer(features, 2 * features, blocks=blocks_M, stride=1)  # 64->128
        self.pool_M_2 = nn.AvgPool2d(2, 2)
        self.res_layer_M_3 = Res_layer(2 * features, 4 * features, blocks=blocks_M, stride=1)  # 128->256
        self.pool_M_3 = nn.AvgPool2d(2, 2)
        self.res_layer_M_4 = Res_layer(4 * features, 8 * features, blocks=blocks_M, stride=1)  # 256->512
        self.pool_M_4 = nn.AvgPool2d(2, 2)
        self.bottleneck_M = Res_layer(8 * features, 8 * features, blocks=blocks_M, stride=1)  # 512->1024

        self.res_layer_I_1 = Res_layer(in_channels, features, blocks=blocks_I, stride=1)  # 1->64
        self.pool_I_1 = nn.AvgPool2d(2, 2)
        self.res_layer_I_2 = Res_layer(features, 2 * features, blocks=blocks_I, stride=1)  # 64->128
        self.pool_I_2 = nn.AvgPool2d(2, 2)
        self.res_layer_I_3 = Res_layer(2 * features, 4 * features, blocks=blocks_I, stride=1)  # 128->256
        self.pool_I_3 = nn.AvgPool2d(2, 2)
        self.res_layer_I_4 = Res_layer(4 * features, 8 * features, blocks=blocks_I, stride=1)  # 256->512
        self.pool_I_4 = nn.AvgPool2d(2, 2)
        self.bottleneck_I = Res_layer(8 * features, 8 * features, blocks=blocks_I, stride=1)  # 512->1024

        self.att_gate_4 = Trans_Att(16 * features, 8 * features, heads=num_heads[3],
                                    dim_head=8 * init_features // num_heads[3])
        self.upconv_4 = UpConv2d_s2(16 * features, 8 * features)
        self.decoder_4 = DoubleConv2d(16 * features, 8 * features)

        self.att_gate_3 = Trans_Att(8 * features, 4 * features, heads=num_heads[2],
                                    dim_head=4 * init_features // num_heads[2])
        self.upconv_3 = UpConv2d_s2(8 * features, 4 * features)
        self.decoder_3 = DoubleConv2d(8 * features, 4 * features)

        self.att_gate_2 = Trans_Att(4 * features, 2 * features, heads=num_heads[1],
                                    dim_head=2 * init_features // num_heads[1])
        self.upconv_2 = UpConv2d_s2(4 * features, 2 * features)
        self.decoder_2 = DoubleConv2d(4 * features, 2 * features)

        self.att_gate_1 = Trans_Att(2 * features, features, heads=num_heads[0], dim_head=init_features // num_heads[0])
        self.upconv_1 = UpConv2d_s2(2 * features, features)
        self.decoder_1 = DoubleConv2d(2 * features, features)

        self.decoder_0 = nn.Sequential(
            nn.Conv2d(features, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, m):
        I_1 = self.res_layer_I_1(x)
        I_2 = self.res_layer_I_2(self.pool_I_1(I_1))
        I_3 = self.res_layer_I_3(self.pool_I_2(I_2))
        I_4 = self.res_layer_I_4(self.pool_I_3(I_3))
        bottleneck_I = self.bottleneck_I(self.pool_I_4(I_4))

        M_1 = self.res_layer_M_1(m)
        M_2 = self.res_layer_M_2(self.pool_M_1(M_1))
        M_3 = self.res_layer_M_3(self.pool_M_2(M_2))
        M_4 = self.res_layer_M_4(self.pool_M_3(M_3))
        bottleneck_M = self.bottleneck_M(self.pool_M_4(M_4))

        temp = self.upconv_4(torch.cat([bottleneck_I, bottleneck_M], dim=1))
        dec_4 = torch.cat([I_4, temp], dim=1)
        dec_4, _ = self.att_gate_4(dec_4, M_4)
        dec_4 = self.decoder_4(dec_4)

        dec_3 = torch.cat([I_3, self.upconv_3(dec_4)], dim=1)
        dec_3, _ = self.att_gate_3(dec_3, M_3)
        dec_3 = self.decoder_3(dec_3)

        dec_2 = torch.cat([I_2, self.upconv_2(dec_3)], dim=1)
        dec_2, _ = self.att_gate_2(dec_2, M_2)
        dec_2 = self.decoder_2(dec_2)

        dec_1 = torch.cat([I_1, self.upconv_1(dec_2)], dim=1)
        # dec_1, _ = self.att_gate_1(dec_1, M_1)
        dec_1 = self.decoder_1(dec_1)

        output = self.decoder_0(dec_1)

        return output