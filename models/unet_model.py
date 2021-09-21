import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch import EfficientNet
# from encoding.models import resnet34


class FPAv2(nn.Module):  # modify upsample   （32, 32）size
    def __init__(self, input_dim, output_dim):
        super(FPAv2, self).__init__()
        self.glob = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))
        self.down1_1 = nn.Sequential(
            nn.Conv2d(input_dim,
                      input_dim,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False), nn.BatchNorm2d(input_dim), nn.ELU(True))
        self.down1_2 = nn.Sequential(
            nn.Conv2d(input_dim,
                      output_dim,
                      kernel_size=7,
                      padding=3,
                      bias=False), nn.BatchNorm2d(output_dim), nn.ELU(True))
        self.down2_1 = nn.Sequential(
            nn.Conv2d(input_dim,
                      input_dim,
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False), nn.BatchNorm2d(input_dim), nn.ELU(True))
        self.down2_2 = nn.Sequential(
            nn.Conv2d(input_dim,
                      output_dim,
                      kernel_size=5,
                      padding=2,
                      bias=False), nn.BatchNorm2d(output_dim), nn.ELU(True))

        self.down3_1 = nn.Sequential(
            nn.Conv2d(input_dim,
                      input_dim,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False), nn.BatchNorm2d(input_dim), nn.ELU(True))
        self.down3_2 = nn.Sequential(
            nn.Conv2d(input_dim,
                      output_dim,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm2d(output_dim), nn.ELU(True))

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim), nn.ELU(True))

    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1
        # x_glob = F.upsample(x_glob, scale_factor=16, mode='bilinear', align_corners=True)  # 256, 16, 16
        x_glob = F.upsample(x_glob,
                            size=(x.size(2), x.size(3)),
                            mode='bilinear',
                            align_corners=True)  # 256, 16, 16

        d1 = self.down1_1(x)  #
        d2 = self.down2_1(d1)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4

        d1 = self.down1_2(d1)
        d2 = self.down2_2(d2)  # 256, 8, 8
        d3 = self.down3_2(d3)  # 256, 4, 4

        d3 = F.upsample(d3,
                        size=(d2.size(2), d2.size(3)),
                        mode='bilinear',
                        align_corners=True)  # 256, 8, 8
        d2 = d2 + d3

        d2 = F.upsample(d2,
                        size=(d1.size(2), d1.size(3)),
                        mode='bilinear',
                        align_corners=True)  # 256, 16, 16
        d1 = d1 + d2
        d1 = F.upsample(d1,
                        size=(x.size(2), x.size(3)),
                        mode='bilinear',
                        align_corners=True)  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d1

        x = x + x_glob

        return x


class FPAv1(nn.Module):  #  (16, 16)
    def __init__(self, input_dim, output_dim):
        super(FPAv1, self).__init__()
        self.glob = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down2_1 = nn.Sequential(
            nn.Conv2d(input_dim,
                      input_dim,
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False), nn.BatchNorm2d(input_dim), nn.ELU(True))
        self.down2_2 = nn.Sequential(
            nn.Conv2d(input_dim,
                      output_dim,
                      kernel_size=5,
                      padding=2,
                      bias=False), nn.BatchNorm2d(output_dim), nn.ELU(True))

        self.down3_1 = nn.Sequential(
            nn.Conv2d(input_dim,
                      input_dim,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False), nn.BatchNorm2d(input_dim), nn.ELU(True))
        self.down3_2 = nn.Sequential(
            nn.Conv2d(input_dim,
                      output_dim,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm2d(output_dim), nn.ELU(True))

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim), nn.ELU(True))

    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1
        x_glob = F.upsample(x_glob,
                            size=(x.size(2), x.size(3)),
                            mode='bilinear',
                            align_corners=True)  # 256, 16, 16

        d2 = self.down2_1(x)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4

        d2 = self.down2_2(d2)  # 256, 8, 8
        d3 = self.down3_2(d3)  # 256, 4, 4

        d3 = F.upsample(d3,
                        size=(d2.size(2), d2.size(3)),
                        mode='bilinear',
                        align_corners=True)  # 256, 8, 8
        d2 = d2 + d3

        d2 = F.upsample(d2,
                        size=(x.size(2), x.size(3)),
                        mode='bilinear',
                        align_corners=True)  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d2

        x = x + x_glob

        return x


def conv3x3(input_dim, output_dim, rate=1):
    return nn.Sequential(
        nn.Conv2d(input_dim,
                  output_dim,
                  kernel_size=3,
                  dilation=rate,
                  padding=rate,
                  bias=False), nn.BatchNorm2d(output_dim), nn.ELU(True))


class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim,
                               input_dim // reduction,
                               kernel_size=1,
                               stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction,
                               input_dim,
                               kernel_size=1,
                               stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


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
                           align_corners=True)
            x = torch.cat([x, e], 1)
        else:
            x = F.upsample(input=x,
                           size=(shape[0], shape[1]),
                           mode='bilinear',
                           align_corners=True)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoderv2(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super(Decoderv2, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1, bias=True)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(True)
        self.s_att = SpatialAttention2d(n_out)
        self.c_att = GAB(n_out, 16)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)

        cat_p = torch.cat([up_p, x_p], 1)
        cat_p = self.relu(self.bn(cat_p))
        s = self.s_att(cat_p)
        c = self.c_att(cat_p)
        return s + c


class Decoderv3(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoderv3, self).__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.conv2 = conv3x3(channels, out_channels)
        self.s_att = SpatialAttention2d(out_channels)
        self.c_att = GAB(out_channels, 16)

    def forward(self, x, e=None, shape=None):
        if e is not None:
            x = F.upsample(input=x,
                           size=(e.size(2), e.size(3)),
                           mode='bilinear',
                           align_corners=True)
            x = torch.cat([x, e], 1)
        else:
            x = F.upsample(input=x,
                           size=(shape[0], shape[1]),
                           mode='bilinear',
                           align_corners=True)
        x = self.conv1(x)
        x = self.conv2(x)
        s = self.s_att(x)
        c = self.c_att(x)
        output = s + c
        return output


class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)


#  -- decoder+scSE, hyper(no FPA)
class Res34Unetv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )

        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  # 128
        self.encoder4 = self.resnet.layer3  # 256
        self.encoder5 = self.resnet.layer4  # 512

        self.center = nn.Sequential(
            conv3x3(512, 512),
            conv3x3(512, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = Decoderv3(256 + 512, 512, 64)
        self.decoder4 = Decoderv3(64 + 256, 256, 64)
        self.decoder3 = Decoderv3(64 + 128, 128, 64)
        self.decoder2 = Decoderv3(64 + 64, 64, 64)
        self.decoder1 = Decoderv3(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        e1 = self.conv1(x)
        # print(e1.size())
        e2 = self.encoder2(e1)
        # print('e2',e2.size())
        e3 = self.encoder3(e2)
        # print('e3',e3.size())
        e4 = self.encoder4(e3)
        # print('e4',e4.size())
        e5 = self.encoder5(e4)
        # print('e5',e5.size())

        f = self.center(e5)
        # print('f',f.size())
        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        # print("d2:", d2.size())
        d1 = self.decoder1(d2, shape=(x.size(2), x.size(3)))
        # print('d1',d1.size())

        f = torch.cat((
            F.upsample(e1,
                       size=(d1.size(2), d1.size(3)),
                       mode='bilinear',
                       align_corners=False),
            d1,
            F.upsample(d2,
                       size=(d1.size(2), d1.size(3)),
                       mode='bilinear',
                       align_corners=False),
            F.upsample(d3,
                       size=(d1.size(2), d1.size(3)),
                       mode='bilinear',
                       align_corners=False),
            F.upsample(d4,
                       size=(d1.size(2), d1.size(3)),
                       mode='bilinear',
                       align_corners=False),
            F.upsample(d5,
                       size=(d1.size(2), d1.size(3)),
                       mode='bilinear',
                       align_corners=False),
        ), 1)

        f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)
        return logit

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


# v4:encoder+scSE, center：FPA; Decoderv2
class Res34Unetv4(nn.Module):
    def __init__(self):
        super(Res34Unetv4, self).__init__()
        self.resnet = torchvision.models.resnet34(True)

        self.conv1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1,
                                   self.resnet.relu)

        self.encode2 = nn.Sequential(self.resnet.layer1, SCse(64))
        self.encode3 = nn.Sequential(self.resnet.layer2, SCse(128))
        self.encode4 = nn.Sequential(self.resnet.layer3, SCse(256))
        self.encode5 = nn.Sequential(self.resnet.layer4, SCse(512))

        self.center = nn.Sequential(FPAv2(512, 256), nn.MaxPool2d(2, 2))

        self.decode5 = Decoderv2(256, 512, 64)
        self.decode4 = Decoderv2(64, 256, 64)
        self.decode3 = Decoderv2(64, 128, 64)
        self.decode2 = Decoderv2(64, 64, 64)
        self.decode1 = Decoder(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1), nn.ELU(True),
            nn.Conv2d(64, 1, kernel_size=1, bias=False))

    def forward(self, x):
        # x: (batch_size, 3, 256, 256)               (3, 256, 1600)

        e1 = self.conv1(x)  # 64, 128, 128            (64,128, 800)
        e2 = self.encode2(e1)  # 64, 128, 128         (64,128, 800)
        e3 = self.encode3(e2)  # 128, 64, 64         (128, 64, 400)
        e4 = self.encode4(e3)  # 256, 32, 32         (256, 32, 200)
        e5 = self.encode5(e4)  # 512, 16, 16         (512, 16, 100)

        f = self.center(e5)  # 256, 8, 8
        d5 = self.decode5(f, e5)  # 64, 16, 16
        d4 = self.decode4(d5, e4)  # 64, 32, 32
        d3 = self.decode3(d4, e3)  # 64, 64, 64
        d2 = self.decode2(d3, e2)  # 64, 128, 128
        d1 = self.decode1(d2)  # 64, 256, 256

        f = torch.cat(
            (d1,
             F.upsample(
                 d2, scale_factor=2, mode='bilinear', align_corners=False),
             F.upsample(
                 d3, scale_factor=4, mode='bilinear', align_corners=False),
             F.upsample(
                 d4, scale_factor=8, mode='bilinear', align_corners=False),
             F.upsample(
                 d5, scale_factor=16, mode='bilinear', align_corners=False)),
            1)  # 320, 256, 256

        logit = self.logit(f)  # 4, 256, 256

        return logit

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


# v5:encoder+scSE, center：FPAv2; Decoderv3;
class Res34Unetv5(nn.Module):
    def __init__(self):
        super(Res34Unetv5, self).__init__()
        self.resnet = torchvision.models.resnet34(True)

        self.conv1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1,
                                   self.resnet.relu)

        self.encode2 = nn.Sequential(self.resnet.layer1, SCse(64))
        self.encode3 = nn.Sequential(self.resnet.layer2, SCse(128))
        self.encode4 = nn.Sequential(self.resnet.layer3, SCse(256))
        self.encode5 = nn.Sequential(self.resnet.layer4, SCse(512))
        self.center = nn.Sequential(FPAv2(512, 256), nn.MaxPool2d(2, 2))

        self.decode5 = Decoderv3(256 + 512, 512, 64)
        self.decode4 = Decoderv3(64 + 256, 256, 64)
        self.decode3 = Decoderv3(64 + 128, 128, 64)
        self.decode2 = Decoderv3(64 + 64, 64, 64)
        self.decode1 = Decoderv3(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1), nn.ELU(True),
            nn.Conv2d(64, 1, kernel_size=1, bias=False))

    def forward(self, x):
        # x: (batch_size, 3, 256, 256)               (3, 513, 513)
        e1 = self.conv1(x)  # 64, 128, 128            (64,257, 257)
        # print("e1:",e1.size())
        e2 = self.encode2(e1)  # 64, 128, 128         (64,257, 257)
        # print("e2:",e2.size())
        e3 = self.encode3(e2)  # 128, 64, 64         (128, 129, 129)
        # print("e3:",e3.size())
        e4 = self.encode4(e3)  # 256, 32, 32         (256, 65, 65)
        # print("e4:",e4.size())

        e5 = self.encode5(e4)  # 512, 16, 16         (512, 33, 33)
        # print("e5:",e5.size())

        f = self.center(e5)  # 256, 8, 8             (512, 16, 16)
        # print("f:", f.size())
        d5 = self.decode5(f, e5)  # 64, 16, 16       (64, 33, 33)
        # print("d5:",d5.size())

        d4 = self.decode4(d5, e4)  # 64, 32, 32      (64, 65, 65)
        # print("d4:",d4.size())

        d3 = self.decode3(d4, e3)  # 64, 64, 64      (64, 129, 129)
        # print("d3:",d3.size())
        d2 = self.decode2(d3, e2)  # 64, 128, 128    (64,257, 257)
        # print("d2:",d2.size())
        d1 = self.decode1(d2, shape=(x.size(2), x.size(3)))  # 64, 256, 256
        # print("d1:",d1.size())

        f = torch.cat((d1,
                       F.upsample(d2,
                                  size=(d1.size(2), d1.size(3)),
                                  mode='bilinear',
                                  align_corners=True),
                       F.upsample(d3,
                                  size=(d1.size(2), d1.size(3)),
                                  mode='bilinear',
                                  align_corners=True),
                       F.upsample(d4,
                                  size=(d1.size(2), d1.size(3)),
                                  mode='bilinear',
                                  align_corners=True),
                       F.upsample(d5,
                                  size=(d1.size(2), d1.size(3)),
                                  mode='bilinear',
                                  align_corners=True)), 1)  # 320, 256, 256

        logit = self.logit(f)  # 4, 256, 256

        return logit


class Res34Unetv6(nn.Module):
    def __init__(self):
        super(Res34Unetv6, self).__init__()
        self.resnet = torchvision.models.resnet34(True)

        self.conv1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1,
                                   self.resnet.relu)

        self.encode2 = nn.Sequential(self.resnet.layer1, SCse(64))
        self.encode3 = nn.Sequential(self.resnet.layer2, SCse(128))
        self.encode4 = nn.Sequential(self.resnet.layer3, SCse(256))
        self.encode5 = nn.Sequential(self.resnet.layer4, SCse(512))
        self.center = nn.Sequential(
            conv3x3(512, 512),
            conv3x3(512, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decode5 = Decoderv3(256 + 512, 512, 64)
        self.decode4 = Decoderv3(64 + 256, 256, 64)
        self.decode3 = Decoderv3(64 + 128, 128, 64)
        self.decode2 = Decoderv3(64 + 64, 64, 64)
        self.decode1 = Decoderv3(64, 32, 64)

        self.logit = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                   nn.ELU(True),
                                   nn.Conv2d(64, 1, kernel_size=1, bias=False))

    def forward(self, x):
        # x: (batch_size, 3, 256, 256)               (3, 513, 513)
        e1 = self.conv1(x)  # 64, 128, 128            (64,257, 257)
        # print("e1:",e1.size())
        e2 = self.encode2(e1)  # 64, 128, 128         (64,257, 257)
        # print("e2:",e2.size())
        e3 = self.encode3(e2)  # 128, 64, 64         (128, 129, 129)
        # print("e3:",e3.size())
        e4 = self.encode4(e3)  # 256, 32, 32         (256, 65, 65)
        # print("e4:",e4.size())

        e5 = self.encode5(e4)  # 512, 16, 16         (512, 33, 33)
        # print("e5:",e5.size())

        f = self.center(e5)  # 256, 8, 8             (512, 16, 16)
        # print("f:", f.size())
        d5 = self.decode5(f, e5)  # 64, 16, 16       (64, 33, 33)
        # print("d5:",d5.size())

        d4 = self.decode4(d5, e4)  # 64, 32, 32      (64, 65, 65)
        # print("d4:",d4.size())

        d3 = self.decode3(d4, e3)  # 64, 64, 64      (64, 129, 129)
        # print("d3:",d3.size())
        d2 = self.decode2(d3, e2)  # 64, 128, 128    (64,257, 257)
        # print("d2:",d2.size())
        d1 = self.decode1(d2, shape=(x.size(2), x.size(3)))  # 64, 256, 256
        # print("d1:",d1.size())

        logit = self.logit(d1)  # 4, 256, 256

        return logit


# only change backbone
class Res34Unetv2(nn.Module):
    def __init__(self):
        super(Res34Unetv2, self).__init__()
        self.resnet = torchvision.models.resnet34(True)

        self.conv1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1,
                                   self.resnet.relu)

        self.encode2 = nn.Sequential(self.resnet.layer1)
        self.encode3 = nn.Sequential(self.resnet.layer2)
        self.encode4 = nn.Sequential(self.resnet.layer3)
        self.encode5 = nn.Sequential(self.resnet.layer4)
        self.center = nn.Sequential(
            conv3x3(512, 512),
            conv3x3(512, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decode5 = Decoderv1(256 + 512, 512, 64)
        self.decode4 = Decoderv1(64 + 256, 256, 64)
        self.decode3 = Decoderv1(64 + 128, 128, 64)
        self.decode2 = Decoderv1(64 + 64, 64, 64)
        self.decode1 = Decoderv1(64, 32, 64)

        self.logit = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                   nn.ELU(True),
                                   nn.Conv2d(64, 1, kernel_size=1, bias=False))

    def forward(self, x):
        # x: (batch_size, 3, 256, 256)               (3, 513, 513)
        e1 = self.conv1(x)  # 64, 128, 128            (64,257, 257)
        # print("e1:",e1.size())
        e2 = self.encode2(e1)  # 64, 128, 128         (64,257, 257)
        # print("e2:",e2.size())
        e3 = self.encode3(e2)  # 128, 64, 64         (128, 129, 129)
        # print("e3:",e3.size())
        e4 = self.encode4(e3)  # 256, 32, 32         (256, 65, 65)
        # print("e4:",e4.size())

        e5 = self.encode5(e4)  # 512, 16, 16         (512, 33, 33)
        # print("e5:",e5.size())

        f = self.center(e5)  # 256, 8, 8             (512, 16, 16)
        # print("f:", f.size())
        d5 = self.decode5(f, e5)  # 64, 16, 16       (64, 33, 33)
        # print("d5:",d5.size())

        d4 = self.decode4(d5, e4)  # 64, 32, 32      (64, 65, 65)
        # print("d4:",d4.size())

        d3 = self.decode3(d4, e3)  # 64, 64, 64      (64, 129, 129)
        # print("d3:",d3.size())
        d2 = self.decode2(d3, e2)  # 64, 128, 128    (64,257, 257)
        # print("d2:",d2.size())
        d1 = self.decode1(d2, shape=(x.size(2), x.size(3)))  # 64, 256, 256
        # print("d1:",d1.size())
        logit = self.logit(d1)  # 4, 256, 256

        return logit


# b7 todo: size
class effunetb7(nn.Module):
    def __init__(self):
        super(effunetb7, self).__init__()
        model = EfficientNet.from_pretrained('efficientnet-b7')
        self.conv1 = nn.Sequential(model._conv_stem, model._bn0,
                                   *[model._blocks[i] for i in range(0, 4)])
        self.encode2 = nn.Sequential(*[model._blocks[i] for i in range(4, 11)])
        self.encode3 = nn.Sequential(
            *[model._blocks[i] for i in range(11, 18)])
        self.encode4 = nn.Sequential(
            *[model._blocks[i] for i in range(18, 38)])
        self.encode5 = nn.Sequential(
            *[model._blocks[i] for i in range(38, 55)])
        del model
        # self.center = FPAv1(640, 224)
        self.center = nn.Sequential(FPAv1(640, 224), nn.MaxPool2d(2, 2))

        # self.decode5 = Decoderv3(224 + 640, 640, 64)
        # self.decode4 = Decoderv3(64 + 224, 224, 64)
        # self.decode3 = Decoderv3(64 + 80, 80, 64)
        # self.decode2 = Decoderv3(64 + 48, 48, 64)
        # self.decode1 = Decoderv3(64, 32, 64)

        self.decode5 = Decoderv3(224 + 640, 224, 48)
        self.decode4 = Decoderv3(48 + 224, 224, 48)
        self.decode3 = Decoderv3(48 + 80, 80, 48)
        self.decode2 = Decoderv3(48 + 48, 48, 48)
        self.decode1 = Decoderv3(48, 32, 48)

        self.logit = nn.Sequential(
            nn.Conv2d(272, 48, kernel_size=3, padding=1), nn.ELU(True),
            nn.Conv2d(64, 1, kernel_size=1, bias=False))

    def forward(self, x):
        # x: (batch_size, 3, 256, 256)                3, 512, 512
        e1 = self.conv1(x)  # 64, 128, 128            32, 256, 256
        print("e1:", e1.size())
        e2 = self.encode2(e1)  # 64, 128, 128         48, 128, 128
        print("e2:", e2.size())
        e3 = self.encode3(e2)  # 128, 64, 64          80, 64, 64
        print("e3:", e3.size())
        e4 = self.encode4(e3)  # 256, 32, 32          224, 32, 32
        print("e4:", e4.size())

        e5 = self.encode5(e4)  # 512, 16, 16          640, 16, 16
        print("e5:", e5.size())

        f = self.center(e5)  # 256, 8, 8              224, 8, 8
        print("f:", f.size())
        d5 = self.decode5(f, e5)  # 64, 16, 16        64, 16, 16
        print("d5:", d5.size())

        d4 = self.decode4(d5, e4)  # 64, 32, 32       64, 32, 32
        print("d4:", d4.size())

        d3 = self.decode3(d4, e3)  # 64, 64, 64       64, 64, 64
        print("d3:", d3.size())
        d2 = self.decode2(d3, e2)  # 64, 128, 128     64, 128, 128
        print("d2:", d2.size())
        d1 = self.decode1(d2, shape=(x.size(2), x.size(3)))  # 64, 512, 512
        print("d1:", d1.size())

        f = torch.cat((F.upsample(e1,
                                  size=(d1.size(2), d1.size(3)),
                                  mode='bilinear',
                                  align_corners=True), d1,
                       F.upsample(d2,
                                  size=(d1.size(2), d1.size(3)),
                                  mode='bilinear',
                                  align_corners=True),
                       F.upsample(d3,
                                  size=(d1.size(2), d1.size(3)),
                                  mode='bilinear',
                                  align_corners=True),
                       F.upsample(d4,
                                  size=(d1.size(2), d1.size(3)),
                                  mode='bilinear',
                                  align_corners=True),
                       F.upsample(d5,
                                  size=(d1.size(2), d1.size(3)),
                                  mode='bilinear',
                                  align_corners=True)), 1)  # 352, 512, 512
        # print("f : ", f.size())
        logit = self.logit(f)  # 4, 256, 256

        return logit


class effunetb4(nn.Module):
    def __init__(self):
        super(effunetb4, self).__init__()
        model = EfficientNet.from_pretrained('efficientnet-b4')

        self.conv1 = nn.Sequential(model._conv_stem, model._bn0,
                                   *[model._blocks[i] for i in range(0, 2)])
        self.encode2 = nn.Sequential(*[model._blocks[i] for i in range(2, 6)])
        self.encode3 = nn.Sequential(*[model._blocks[i] for i in range(6, 10)])
        self.encode4 = nn.Sequential(
            *[model._blocks[i] for i in range(10, 22)])
        self.encode5 = nn.Sequential(
            *[model._blocks[i] for i in range(22, 32)])
        del model
        # self.center = FPAv2(640, 224)
        self.center = nn.Sequential(FPAv1(448, 160), nn.MaxPool2d(2, 2))

        self.decode5 = Decoderv3(160 + 448, 448, 56)
        self.decode4 = Decoderv3(56 + 160, 160, 56)
        self.decode3 = Decoderv3(56 + 56, 56, 56)
        self.decode2 = Decoderv3(56 + 32, 32, 56)
        self.decode1 = Decoderv3(56, 24, 56)

        self.logit = nn.Sequential(
            nn.Conv2d(304, 56, kernel_size=3, padding=1), nn.ELU(True),
            nn.Conv2d(56, 1, kernel_size=1, bias=False))

    def forward(self, x):
        # x: (batch_size, 3, 256, 256)                3, 512, 512
        e1 = self.conv1(x)  # 64, 128, 128            24, 256, 256
        # print("e1:",e1.size())
        e2 = self.encode2(e1)  # 64, 128, 128         32, 128, 128
        # print("e2:",e2.size())
        e3 = self.encode3(e2)  # 128, 64, 64          56, 64, 64
        # print("e3:",e3.size())
        e4 = self.encode4(e3)  # 256, 32, 32          160, 32, 32
        # print("e4:",e4.size())

        e5 = self.encode5(e4)  # 512, 16, 16          448, 16, 16
        # print("e5:",e5.size())

        f = self.center(e5)  # 256, 8, 8              160, 8, 8
        # print("f:", f.size())
        d5 = self.decode5(f, e5)  # 64, 16, 16        80, 8, 8
        # print("d5:",d5.size())

        d4 = self.decode4(d5, e4)  # 64, 32, 32       80, 16, 16
        # print("d4:",d4.size())

        d3 = self.decode3(d4, e3)  # 64, 64, 64      (64, 129, 129)
        # print("d3:",d3.size())
        d2 = self.decode2(d3, e2)  # 64, 128, 128    (64,257, 257)
        # print("d2:",d2.size())
        d1 = self.decode1(d2, shape=(x.size(2), x.size(3)))  # 64, 256, 256
        # print("d1:",d1.size())

        f = torch.cat((F.upsample(e1,
                                  size=(d1.size(2), d1.size(3)),
                                  mode='bilinear',
                                  align_corners=True), d1,
                       F.upsample(d2,
                                  size=(d1.size(2), d1.size(3)),
                                  mode='bilinear',
                                  align_corners=True),
                       F.upsample(d3,
                                  size=(d1.size(2), d1.size(3)),
                                  mode='bilinear',
                                  align_corners=True),
                       F.upsample(d4,
                                  size=(d1.size(2), d1.size(3)),
                                  mode='bilinear',
                                  align_corners=True),
                       F.upsample(d5,
                                  size=(d1.size(2), d1.size(3)),
                                  mode='bilinear',
                                  align_corners=True)), 1)  # 320, 256, 256
        # print("f : ", f.size())
        logit = self.logit(f)  # 4, 256, 256

        return logit


if __name__ == "__main__":
    # img = torch.rand(1,3, 513,513).cuda()
    model = Res34Unetv2().cuda().eval()
    # # print(model)
    print(sum(p.numel() for p in model.parameters()))
    # with torch.no_grad():
    #     out = model(img)
    # print(out.size())
