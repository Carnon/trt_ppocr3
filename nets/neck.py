import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.common import Activation


class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias_attr=False, groups=1, act='swish'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=(padding, padding), groups=groups, bias=bias_attr)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = Activation(act_type=act, inplace=True)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='swish'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = Activation(act_type=act_layer, inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, mixer='Global', qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        _, N, C = x.shape

        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C // self.num_heads)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q.matmul(k.permute(0, 1, 3, 2)))
        attn = F.softmax(attn, dim=-1)
        x = (attn.matmul(v)).permute(0, 2, 1, 3).reshape((-1, N, C))
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mixer='Global', mlp_ratio=2., qkv_bias=True, qk_scale=None, act_layer='swish', norm_layer='nn.LayerNorm', epsilon=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=epsilon)
        self.mixer = Attention(dim, num_heads=num_heads, mixer=mixer, qk_scale=qk_scale)
        self.norm2 = nn.LayerNorm(dim, eps=epsilon)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        x = x.squeeze(dim=2)
        x = x.permute(0, 2, 1)
        return x


class EncoderWithSVTR(nn.Module):
    def __init__(self, in_channels, dims=64, depth=2, hidden_dims=120, use_guide=False, num_heads=8, qkv_bias=True, mlp_ratio=2.0, drop_rate=0.1, attn_drop_rate=0.1, drop_path=0., qk_scale=None):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer(in_channels, in_channels // 8, padding=1, act='swish')
        self.conv2 = ConvBNLayer(in_channels // 8, hidden_dims, kernel_size=1, act='swish')

        self.svtr_block = nn.ModuleList([
            Block(dim=hidden_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, act_layer='swish', norm_layer='nn.LayerNorm', epsilon=1e-05) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvBNLayer(hidden_dims, in_channels, kernel_size=1, act='swish')

        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = ConvBNLayer(2 * in_channels, in_channels // 8, padding=1, act='swish')

        self.conv1x1 = ConvBNLayer(in_channels // 8, dims, kernel_size=1, act='swish')
        self.out_channels = dims
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # weight initialization
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        z = x
        # for short cut
        h = z
        # reduce dim
        z = self.conv1(z)
        z = self.conv2(z)
        # SVTR global block
        B, C, H, W = z.shape
        z = z.flatten(2).permute(0, 2, 1)

        for blk in self.svtr_block:
            z = blk(z)

        z = self.norm(z)

        z = z.reshape([-1, H, W, C]).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        z = self.conv4(z)
        z = self.conv1x1(z)
        return z


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels

        self.encoder = EncoderWithSVTR(self.encoder_reshape.out_channels, **kwargs)
        self.out_channels = self.encoder.out_channels

    def forward(self, x):
        x = self.encoder(x)
        x = self.encoder_reshape(x)
        return x
