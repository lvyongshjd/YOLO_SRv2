import torch
import torch.nn as nn

from nets.CSPdarknet53_tiny import darknet53_tiny
from nets.attention import cbam_block, eca_block, se_block, CA_Block

attention_block = [se_block, cbam_block, eca_block, CA_Block]

#-------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class yolo_decouple_head(nn.Module):
 
    def __init__(self, filters_list, in_filters):
        super(yolo_decouple_head, self).__init__()


        self.conv0 = BaseConv(
                     in_channels=in_filters,
                    out_channels=filters_list[0],
                    ksize=1,
                    stride=1,
                    act="silu",
                    )

        self.conv1 = BaseConv(
                     in_channels=filters_list[0],
                    out_channels=2*filters_list[0],
                    ksize=3,
                    stride=1,
                    act="silu",
                    )

        self.conv2 = nn.Conv2d(2*filters_list[0],(filters_list[1]//3-5)*3,1)

        self.conv3 =  BaseConv(
                     in_channels=2*filters_list[0],
                    out_channels=2*filters_list[0],
                    ksize=3,
                    stride=1,
                    act="silu",
                    )
        self.conv4 = nn.Conv2d(2*filters_list[0],4*3,1)            
        self.conv5 = nn.Conv2d(2*filters_list[0],1*3,1)

   

    def forward(self, x,):
        x =  self.conv0(x)
        x1 =  self.conv1(x)
        x2 = self.conv3(x1)
        cls_head = self.conv2(x2)

        x3 = self.conv1(x)

        x4 = self.conv3(x3)

        x5 = self.conv3(x3)

        reg_head = self.conv4(x4)

        IOU_head = self.conv5(x5)

        output_head = torch.cat([reg_head,IOU_head,cls_head],1)

        return output_head
#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi=0, pretrained=False):
        super(YoloBody, self).__init__()
        self.phi            = phi
        self.backbone       = darknet53_tiny(pretrained)

        self.conv_for_P5    = BasicConv(512,256,1)
        # self.yolo_headP5    = yolo_head([512, len(anchors_mask[0]) * (5 + num_classes)],256)

        self.yolo_headP5    = yolo_decouple_head([512, len(anchors_mask[0]) * (5 + num_classes)],256)

        self.upsample       = Upsample(256,128)

        self.yolo_headP4    = yolo_decouple_head([256, len(anchors_mask[1]) * (5 + num_classes)],384)

        # self.yolo_headP4    = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)],384)

        if 1 <= self.phi and self.phi <= 4:
            self.feat1_att      = attention_block[self.phi - 1](256)
            self.feat2_att      = attention_block[self.phi - 1](512)
            self.upsample_att   = attention_block[self.phi - 1](128)

    def forward(self, x):
        #---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        #---------------------------------------------------#
        feat1, feat2 = self.backbone(x)
        if 1 <= self.phi and self.phi <= 4:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)

        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2)
        # 13,13,256 -> 13,13,512 -> 13,13,255
        out0 = self.yolo_headP5(P5) 

        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample(P5)
        # 26,26,256 + 26,26,128 -> 26,26,384
        if 1 <= self.phi and self.phi <= 4:
            P5_Upsample = self.upsample_att(P5_Upsample)
        P4 = torch.cat([P5_Upsample,feat1],axis=1)

        # 26,26,384 -> 26,26,256 -> 26,26,255
        out1 = self.yolo_headP4(P4)

        # print(out0.size())

        # print(out1.size())
        
        return out0, out1

