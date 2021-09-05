import torch
import torch.nn as nn

from mobilenet_v2 import get_inverted_residual_blocks, InvertedResidual
import torch.nn.functional as F


def conv3x3(input_dim, output_dim, rate=1):
    return nn.Sequential(
        nn.Conv2d(input_dim,
                  output_dim,
                  kernel_size=3,
                  dilation=rate,
                  padding=rate,
                  bias=False), nn.BatchNorm2d(output_dim), nn.ELU(True))


class Decoderv1(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoderv1, self).__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.conv2 = conv3x3(channels, out_channels)

    def forward(self, x, e=None, shape=None):
        if e is not None:
            x = F.upsample(input=x,
                           size=(e.size(2), e.size(3)),
                           mode='bilinear',
                           align_corners=False)
            x = torch.cat([x, e], 1)
        else:
            x = F.upsample(input=x,
                           size=(shape[0], shape[1]),
                           mode='bilinear',
                           align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DeepLabV3MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3MobileNetV2, self).__init__()

        # same config params in MobileNetV2
        # each layer channel
        self.c = [32, 16, 24, 32, 64, 96, 160]
        # each layer expansion times
        self.t = [1, 1, 6, 6, 6, 6, 6]
        # each layer expansion stride
        self.s = [2, 1, 2, 2, 2, 1, 2]
        # each layer repeat time
        self.n = [1, 1, 2, 3, 4, 3, 3]
        self.down_sample_rate = 32
        self.output_stride = 16
        self.multi_grid = (1, 2, 4)
        self.aspp = (6, 12, 18)

        # build MobileNetV2 backbone first
        self.inconv = nn.Sequential(
            nn.Conv2d(3, self.c[0], 3, stride=self.s[0], padding=1,
                      bias=False), nn.BatchNorm2d(self.c[0]), nn.ReLU6())
        self.down1 = nn.Sequential(*(
            get_inverted_residual_blocks(
                self.c[0], self.c[1], t=self.t[1], s=self.s[1], n=self.n[1]) +
            get_inverted_residual_blocks(
                self.c[1], self.c[2], t=self.t[2], s=self.s[2], n=self.n[2])))
        self.down2 = get_inverted_residual_blocks(self.c[2],
                                                  self.c[3],
                                                  t=self.t[3],
                                                  s=self.s[3],
                                                  n=self.n[3])[0]

        self.down3 = get_inverted_residual_blocks(self.c[3],
                                                  self.c[4],
                                                  t=self.t[4],
                                                  s=self.s[4],
                                                  n=self.n[4])[0]

        self.down4 = nn.Sequential(*(
            get_inverted_residual_blocks(
                self.c[4], self.c[5], t=self.t[5], s=self.s[5], n=self.n[5]) +
            get_inverted_residual_blocks(
                self.c[5], self.c[6], t=self.t[6], s=self.s[6], n=self.n[6])))

        self.decode5 = Decoderv1(160 + 64, 160, 64)
        self.decode4 = Decoderv1(64 + 32, 64, 32)
        self.decode3 = Decoderv1(32 + 24, 32, 24)
        self.decode2 = Decoderv1(24 + 32, 64, 64)
        self.decode1 = Decoderv1(64, 32, 64)

        self.logit = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                   nn.ELU(True),
                                   nn.Conv2d(64, 1, kernel_size=1, bias=False))

    def forward(self, x):  # x.size = 3,256,256
        x1 = self.inconv(x)  # 128
        # print(x1.size())
        x2 = self.down1(x1)  # 64
        # print(x2.size())
        x3 = self.down2(x2)  # 32
        # print(x3.size())
        x4 = self.down3(x3)  # 16
        # print(x4.size())
        x5 = self.down4(x4)  # 16
        # print(x5.size())
        e5 = self.decode5(x5, x4)
        e4 = self.decode4(e5, x3)
        e3 = self.decode3(e4, x2)
        e2 = self.decode2(e3, x1)
        e1 = self.decode1(e2, shape=(x.size(2), x.size(3)))
        out = self.logit(e1)
        return out


class DeepLabV3MobileNetV2_V2(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3MobileNetV2_V2, self).__init__()

        # same config params in MobileNetV2
        # each layer channel
        self.c = [32, 16, 24, 32, 64, 96, 160]
        # each layer expansion times
        self.t = [1, 1, 6, 6, 6, 6, 6]
        # each layer expansion stride
        self.s = [2, 1, 2, 2, 2, 1, 2]
        # each layer repeat time
        self.n = [1, 1, 2, 3, 4, 3, 3]
        self.down_sample_rate = 32
        self.output_stride = 16
        self.multi_grid = (1, 2, 4)
        self.aspp = (6, 12, 18)

        # build MobileNetV2 backbone first
        self.inconv = nn.Sequential(
            nn.Conv2d(3, self.c[0], 3, stride=self.s[0], padding=1,
                      bias=False), nn.BatchNorm2d(self.c[0]), nn.ReLU6())
        self.down1 = nn.Sequential(*(
            get_inverted_residual_blocks(
                self.c[0], self.c[1], t=self.t[1], s=self.s[1], n=self.n[1]) +
            get_inverted_residual_blocks(
                self.c[1], self.c[2], t=self.t[2], s=self.s[2], n=self.n[2])))
        self.down2 = get_inverted_residual_blocks(self.c[2],
                                                  self.c[3],
                                                  t=self.t[3],
                                                  s=self.s[3],
                                                  n=self.n[3])[0]

        self.down3 = get_inverted_residual_blocks(self.c[3],
                                                  self.c[4],
                                                  t=self.t[4],
                                                  s=self.s[4],
                                                  n=self.n[4])[0]

        self.down4 = nn.Sequential(*(
            get_inverted_residual_blocks(
                self.c[4], self.c[5], t=self.t[5], s=self.s[5], n=self.n[5]) +
            get_inverted_residual_blocks(
                self.c[5], self.c[6], t=self.t[6], s=self.s[6], n=self.n[6])))

        self.decode5 = Decoderv1(160 + 64, 64, 32)
        self.decode4 = Decoderv1(32 + 32, 32, 32)
        self.decode3 = Decoderv1(32 + 24, 32, 32)
        self.decode2 = Decoderv1(32 + 32, 32, 32)
        self.decode1 = Decoderv1(32, 32, 32)

        self.logit = nn.Sequential(
            nn.Conv2d(160, 32, kernel_size=3, padding=1), nn.ELU(True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False))

    def forward(self, x):  # x.size = 3,256,256
        x1 = self.inconv(x)  # 128
        print(x1.size())
        x2 = self.down1(x1)  # 64
        print(x2.size())
        x3 = self.down2(x2)  # 32
        print(x3.size())
        x4 = self.down3(x3)  # 16
        print(x4.size())
        x5 = self.down4(x4)  # 16
        print(x5.size())
        e5 = self.decode5(x5, x4)
        e4 = self.decode4(e5, x3)
        e3 = self.decode3(e4, x2)
        e2 = self.decode2(e3, x1)
        e1 = self.decode1(e2, shape=(x.size(2), x.size(3)))

        f = torch.cat((e1,
                       F.upsample(e2,
                                  size=(e1.size(2), e1.size(3)),
                                  mode='bilinear',
                                  align_corners=False),
                       F.upsample(e3,
                                  size=(e1.size(2), e1.size(3)),
                                  mode='bilinear',
                                  align_corners=False),
                       F.upsample(e4,
                                  size=(e1.size(2), e1.size(3)),
                                  mode='bilinear',
                                  align_corners=False),
                       F.upsample(e5,
                                  size=(e1.size(2), e1.size(3)),
                                  mode='bilinear',
                                  align_corners=False)), 1)
        out = self.logit(f)
        return out


if __name__ == "__main__":
    img = torch.rand(1, 3, 512, 512).cuda()
    model = DeepLabV3MobileNetV2_V2(1).cuda().eval()
    # print(sum(p.numel() for p in model.parameters()))
    with torch.no_grad():
        out = model(img)
    # print(out.size())
