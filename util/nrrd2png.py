# -*- coding:utf8 -*-
import nrrd
import numpy as np
import cv2
import os
import pdb


class BatchRename():
    '''
    批量处理生成文件夹中的nrrd文件对应的mask.png

    '''
    def __init__(self):
        self.path = '/home/wn/Downloads/shibiao/'
        self.maskpath = '/home/ubuntu/Document/seg/mask/'
        self.imagepath = '/home/ubuntu/Document/seg/image/'
        self.sourcepath = '/home/ubuntu/Document/seg/source/'
        if not os.path.exists(self.maskpath):
            os.makedirs(self.maskpath)
        if not os.path.exists(self.imagepath):
            os.makedirs(self.imagepath)

    def rename(self):
        filelist = os.listdir(self.path)  #获取文件路径
        total_num = len(filelist)  #获取文件长度（个数）
        i = 1  #表示文件的命名是从1开始的
        for item in filelist:
            if item.endswith(
                    '.nrrd'
            ):  #初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
                fnrrd = os.path.join(os.path.abspath(self.path), item)
                #print(fnrrd)
                new = item[:-5]
                src = os.path.join(os.path.abspath(self.sourcepath),
                                   (new + '.tif'))
                dst = os.path.join(
                    os.path.abspath(self.maskpath),
                    (new + '.png'))  #处理后的格式也为jpg格式的，当然这里可以改成png格式
                #dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')    这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
                #pdb.set_trace()
                try:
                    #os.rename(src, dst)
                    #print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                    nrrd_tmp, nrrd_options = nrrd.read(fnrrd)

                    mask = nrrd_tmp * 255
                    mask = np.transpose(mask)
                    mask = np.squeeze(mask)
                    cv2.imwrite(dst, mask)

                    img = cv2.imread(src)
                    imgh = cv2.flip(img, 1)
                    src0 = os.path.join(os.path.abspath(self.imagepath),
                                        (new + '.png'))
                    cv2.imwrite(src0, img)
                    srch = os.path.join(os.path.abspath(self.imagepath),
                                        (new + '_h.png'))
                    cv2.imwrite(srch, imgh)

                    imgv = cv2.flip(img, 0)
                    srcv = os.path.join(os.path.abspath(self.imagepath),
                                        (new + '_v.png'))
                    cv2.imwrite(srcv, imgv)

                    imghv = cv2.flip(img, -1)
                    srchv = os.path.join(os.path.abspath(self.imagepath),
                                         (new + '_hv.png'))
                    cv2.imwrite(srchv, imghv)

                    maskh = cv2.flip(mask, 1)
                    dsth = os.path.join(os.path.abspath(self.maskpath),
                                        (new + '_h.png'))
                    cv2.imwrite(dsth, maskh)

                    maskv = cv2.flip(mask, 0)
                    dstv = os.path.join(os.path.abspath(self.maskpath),
                                        (new + '_v.png'))
                    cv2.imwrite(dstv, maskv)

                    maskhv = cv2.flip(mask, -1)
                    dsthv = os.path.join(os.path.abspath(self.maskpath),
                                         (new + '_hv.png'))
                    cv2.imwrite(dsthv, maskhv)

                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
