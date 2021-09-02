import numpy as np
import os
import nrrd
import cv2


img_path = "/media/user/DATA/测试训练集/测试训练1/train"  # ori_img path
contours_path = "/media/user/DATA/测试训练集/测试训练1/contours"   # save contours path
mask_path = '/media/user/DATA/测试训练集/测试训练1/mask'    # save mask path

img_list = os.listdir(img_path)
for data in img_list:
    if data.endswith("tif"):
        image_path = os.path.join(img_path,data)
        mask = cv2.imread(os.path.join(mask_path,data.replace('tif','png')),0)
        image = cv2.imread(image_path)

        ret, binary = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
        savepath = os.path.join(contours_path,data.replace('tif','png'))
        cv2.imwrite(savepath,image)
print("done!")
