import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.common import Activation


class ConvBNLayer(nn.Module):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 act='hard_swish'):
        super(ConvBNLayer, self).__init__()
        self.act = act
        self._conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias=False)

        self._batch_norm = nn.BatchNorm2d(
            num_filters,
        )
        if self.act is not None:
            self._act = Activation(act_type=act, inplace=True)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act is not None:
            y = self._act(y)
        return y


class DepthwiseSeparable(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters1,
                 num_filters2,
                 num_groups,
                 stride,
                 scale,
                 dw_size=3,
                 padding=1,
                 use_se=False):
        super(DepthwiseSeparable, self).__init__()
        self.use_se = use_se
        self._depthwise_conv = ConvBNLayer(num_channels=num_channels, num_filters=int(num_filters1 * scale),
            filter_size=dw_size, stride=stride,
            padding=padding, num_groups=int(num_groups * scale))
        if use_se:
            self._se = SEModule(int(num_filters1 * scale))
        self._pointwise_conv = ConvBNLayer(
            num_channels=int(num_filters1 * scale),
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        if self.use_se:
            y = self._se(y)
        y = self._pointwise_conv(y)
        return y


class MobileNetV1Enhance(nn.Module):
    def __init__(self, in_channels=3, scale=0.5, last_conv_stride=(1, 2), last_pool_type='avg', **kwargs):
        super().__init__()
        self.scale = scale
        self.block_list = []

        self.conv1 = ConvBNLayer(num_channels=in_channels, filter_size=3, channels=3, num_filters=int(32 * scale), stride=2, padding=1)

        conv2_1 = DepthwiseSeparable(num_channels=int(32 * scale), num_filters1=32, num_filters2=64, num_groups=32, stride=1, scale=scale)
        self.block_list.append(conv2_1)

        conv2_2 = DepthwiseSeparable(
            num_channels=int(64 * scale),
            num_filters1=64,
            num_filters2=128,
            num_groups=64,
            stride=1,
            scale=scale)
        self.block_list.append(conv2_2)

        conv3_1 = DepthwiseSeparable(
            num_channels=int(128 * scale),
            num_filters1=128,
            num_filters2=128,
            num_groups=128,
            stride=1,
            scale=scale)
        self.block_list.append(conv3_1)

        conv3_2 = DepthwiseSeparable(
            num_channels=int(128 * scale),
            num_filters1=128,
            num_filters2=256,
            num_groups=128,
            stride=(2, 1),
            scale=scale)
        self.block_list.append(conv3_2)

        conv4_1 = DepthwiseSeparable(
            num_channels=int(256 * scale),
            num_filters1=256,
            num_filters2=256,
            num_groups=256,
            stride=1,
            scale=scale)
        self.block_list.append(conv4_1)

        conv4_2 = DepthwiseSeparable(
            num_channels=int(256 * scale),
            num_filters1=256,
            num_filters2=512,
            num_groups=256,
            stride=(2, 1),
            scale=scale)
        self.block_list.append(conv4_2)

        for _ in range(5):
            conv5 = DepthwiseSeparable(
                num_channels=int(512 * scale),
                num_filters1=512,
                num_filters2=512,
                num_groups=512,
                stride=1,
                dw_size=5,
                padding=2,
                scale=scale,
                use_se=False)
            self.block_list.append(conv5)

        conv5_6 = DepthwiseSeparable(
            num_channels=int(512 * scale),
            num_filters1=512,
            num_filters2=1024,
            num_groups=512,
            stride=(2, 1),
            dw_size=5,
            padding=2,
            scale=scale,
            use_se=True)
        self.block_list.append(conv5_6)

        conv6 = DepthwiseSeparable(
            num_channels=int(1024 * scale),
            num_filters1=1024,
            num_filters2=1024,
            num_groups=1024,
            stride=last_conv_stride,
            dw_size=5,
            padding=2,
            use_se=True,
            scale=scale)
        self.block_list.append(conv6)

        self.block_list = nn.Sequential(*self.block_list)
        if last_pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = int(1024 * scale)

    def forward(self, inputs):
        y = self.conv1(inputs)
        y = self.block_list(y)
        y = self.pool(y)
        return y


def hardsigmoid(x):
    return F.relu6(x + 3., inplace=True) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,  padding=0, bias=True)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = hardsigmoid(outputs)
        x = torch.mul(inputs, outputs)

        return x


# class ConvBNLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, is_vd_mode=False, act=None, name=None):
#         super(ConvBNLayer, self).__init__()
#         self.act = act
#         self.is_vd_mode = is_vd_mode
#         self._pool2d_avg = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)
#         self._conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1 if is_vd_mode else stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False)
#         self._batch_norm = nn.BatchNorm2d(out_channels,)
#         if self.act is not None:
#             self._act = Activation(act_type=act, inplace=True)
#
#     def forward(self, inputs):
#         if self.is_vd_mode:
#             inputs = self._pool2d_avg(inputs)
#         y = self._conv(inputs)
#         y = self._batch_norm(y)
#         if self.act is not None:
#             y = self._act(y)
#         return y
#
#
# class BottleneckBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride, shortcut=True, if_first=False, name=None):
#         super(BottleneckBlock, self).__init__()
#
#         self.conv0 = ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, act='relu', name=name + "_branch2a")
#         self.conv1 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, act='relu', name=name + "_branch2b")
#         self.conv2 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, act=None, name=name + "_branch2c")
#
#         if not shortcut:
#             self.short = ConvBNLayer(in_channels=in_channels, out_channels=out_channels * 4, kernel_size=1, stride=stride, is_vd_mode=not if_first and stride[0] != 1, name=name + "_branch1")
#
#         self.shortcut = shortcut
#
#     def forward(self, inputs):
#         y = self.conv0(inputs)
#
#         conv1 = self.conv1(y)
#         conv2 = self.conv2(conv1)
#
#         if self.shortcut:
#             short = inputs
#         else:
#             short = self.short(inputs)
#         y = short + conv2
#         y = F.relu(y)
#         return y
#
#
# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride, shortcut=True, if_first=False, name=None):
#         super(BasicBlock, self).__init__()
#         self.stride = stride
#         self.conv0 = ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, act='relu', name=name + "_branch2a")
#         self.conv1 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, act=None, name=name + "_branch2b")
#
#         if not shortcut:
#             self.short = ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, is_vd_mode=not if_first and stride[0] != 1, name=name + "_branch1")
#
#         self.shortcut = shortcut
#
#     def forward(self, inputs):
#         y = self.conv0(inputs)
#         conv1 = self.conv1(y)
#
#         if self.shortcut:
#             short = inputs
#         else:
#             short = self.short(inputs)
#         y = short + conv1
#         y = F.relu(y)
#         return y
#
#
# class ResNet(nn.Module):
#     def __init__(self, in_channels=3, layers=50, **kwargs):
#         super(ResNet, self).__init__()
#
#         self.layers = layers
#         supported_layers = [18, 34, 50, 101, 152, 200]
#         assert layers in supported_layers, \
#             "supported layers are {} but input layer is {}".format(
#                 supported_layers, layers)
#
#         if layers == 18:
#             depth = [2, 2, 2, 2]
#         elif layers == 34 or layers == 50:
#             depth = [3, 4, 6, 3]
#         elif layers == 101:
#             depth = [3, 4, 23, 3]
#         elif layers == 152:
#             depth = [3, 8, 36, 3]
#         elif layers == 200:
#             depth = [3, 12, 48, 3]
#         num_channels = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
#         num_filters = [64, 128, 256, 512]
#
#         self.conv1_1 = ConvBNLayer(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, act='relu', name="conv1_1")
#         self.conv1_2 = ConvBNLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, act='relu', name="conv1_2")
#         self.conv1_3 = ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, act='relu', name="conv1_3")
#         self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         # self.block_list = list()
#         self.block_list = nn.Sequential()
#         if layers >= 50:
#             for block in range(len(depth)):
#                 shortcut = False
#                 for i in range(depth[block]):
#                     if layers in [101, 152, 200] and block == 2:
#                         if i == 0:
#                             conv_name = "res" + str(block + 2) + "a"
#                         else:
#                             conv_name = "res" + str(block + 2) + "b" + str(i)
#                     else:
#                         conv_name = "res" + str(block + 2) + chr(97 + i)
#
#                     if i == 0 and block != 0:
#                         stride = (2, 1)
#                     else:
#                         stride = (1, 1)
#
#                     bottleneck_block = BottleneckBlock(in_channels=num_channels[block] if i == 0 else num_filters[block] * 4, out_channels=num_filters[block], stride=stride, shortcut=shortcut, if_first=block == i == 0, name=conv_name)
#                     shortcut = True
#                     # self.block_list.append(bottleneck_block)
#                     self.block_list.add_module('bb_%d_%d' % (block, i), bottleneck_block)
#                 self.out_channels = num_filters[block]
#         else:
#             for block in range(len(depth)):
#                 shortcut = False
#                 for i in range(depth[block]):
#                     conv_name = "res" + str(block + 2) + chr(97 + i)
#                     if i == 0 and block != 0:
#                         stride = (2, 1)
#                     else:
#                         stride = (1, 1)
#
#                     basic_block = BasicBlock(in_channels=num_channels[block] if i == 0 else num_filters[block], out_channels=num_filters[block], stride=stride, shortcut=shortcut, if_first=block == i == 0, name=conv_name)
#
#                     shortcut = True
#                     # self.block_list.append(basic_block)
#                     self.block_list.add_module('bb_%d_%d' % (block, i), basic_block)
#                 self.out_channels = num_filters[block]
#         self.out_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#
#     def forward(self, inputs):
#         y = self.conv1_1(inputs)
#         y = self.conv1_2(y)
#         y = self.conv1_3(y)
#         y = self.pool2d_max(y)
#         for block in self.block_list:
#             y = block(y)
#         y = self.out_pool(y)
#
#         return y


if __name__ == '__main__':
    from torchinfo import summary

    # x = torch.rand(size=(1, 3, 48, 480))
    # res = ResNet(layers=34)
    # y = res(x)
    # print(y.shape)
    # summary(res, input_size=(1, 3, 32, 320))

    print("hello world")
