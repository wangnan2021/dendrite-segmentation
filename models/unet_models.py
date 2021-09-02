# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)#64
        x2 = self.down1(x1)#128
        x3 = self.down2(x2)#256
        x4 = self.down3(x3)#512
        x5 = self.down4(x4)#1024
        x = self.up1(x5, x4)#512
        x = self.up2(x, x3)#256
        x = self.up3(x, x2)#128
        x = self.up4(x, x1)#64
        x = self.outc(x)
        return x

class UNetv1(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetv1, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.up4 = up(64, 32)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)#64
        x2 = self.down1(x1)#128
        x3 = self.down2(x2)#256
        x4 = self.down3(x3)#512
        x5 = self.down4(x4)#1024
        x = self.up1(x5, x4)#512
        x = self.up2(x, x3)#256
        x = self.up3(x, x2)#128
        x = self.up4(x, x1)#64
        x = self.outc(x)
        return x



if __name__ == "__main__":
    img = torch.rand(1,3, 256,256).cuda()
    model = UNetv1(n_channels=3, n_classes=1).cuda().eval()
    # # print(model)
    print(sum(p.numel() for p in model.parameters()))
    # for name, para in model.named_parameters():
    #     print(name, para)
    # with torch.no_grad():
    #     out = model(img)
    # print(out.size())