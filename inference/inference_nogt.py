from models.DFN import DFN
from models.poly_lr_schedule import poly_lr_schedule
from models.losses import get_loss, bcelovasz
import torch
import torch.nn as nn
import sys
from unet_mb2 import *
from unet_model import *
from unet_models import *
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import random
import os
import numpy as np
import time
import warnings
import cv2
import copy
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from albumentations import pytorch as AT
import albumentations as albu
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
from network import EMANet
from deeplabv3plus import deeplabv3plus_efficientnet
# SEED = 42

# warnings.filterwarnings("ignore")
# # load_path = "/media/user/DATA/all label cos train/log/res34unetv5/2019-11-22, 16:14:04/models/"
# # load_path = "/media/user/DATA/all_labeldata+new_wei/log/res34unetv5/2019-11-22, 23:54:12/best_model/checkpoint.pth.tar"
# # load_path = "/media/user/DATA/2019-11-20(split2)/split2_wei_ensemblemodel/2019-11-29, 22:14:47/best_model/checkpoint.pth.tar"
# # load_path = "/media/user/DATA/2019-11-20(split2)/unet/2019-12-10, 13:15:15/best_model/checkpoint.pth.tar"
# load_path = "../../code/acta/2019-11-20(split2)/poly/v5/best_model/checkpoint.pth.tar"

# # load_path = '/media/user/DATA/2019-11-20(split2)/poly/v5/best_model/checkpoint.pth.tar'  
# # MODEL = deeplabv3plus_efficientnet(pretrained=True)
# # MODEL = EMANet(n_classes=1, n_layers=101)
# # MODEL = UNetv1(n_channels=3,n_classes=1)
# # MODEL =Res34Unetv1()
# MODEL = Res34Unetv5()
# # MODEL = Res34Unetv2()
# # MODEL = DeepLabV3MobileNetV2(1)
# # IMG_PATH = '/media/user/DATA/测试训练集/wei+/分批加入/600/image'
# # OUT_PATH ='/media/user/DATA/测试训练集/wei+/分批加入/600/mask'
# IMG_PATH = "/media/ubuntu/Transcend/tangyang_data/Segmentation0623/norm"   #BM3D
# # IMG_PATH = '../dataset/images'
# OUT_PATH_BASE ="/media/ubuntu/Transcend/tangyang_data/Segmentation0623/results"
# instance = IMG_PATH.split("/")[-1]
# OUT_PATH = os.path.join(OUT_PATH_BASE,instance)
# if not os.path.exists(OUT_PATH):
#     os.makedirs(OUT_PATH)

# MAP_PATH = os.path.join(OUT_PATH_BASE,instance+'_map')
# if not os.path.exists(MAP_PATH):
#     os.makedirs(MAP_PATH)
# TTA = False
# ensemble = False



class mydataset(Dataset):
    def __init__(self, base_path, is_tta=False, mode='test'):
        super(mydataset, self).__init__()
        self.is_tta = is_tta
        self.mode = mode
        self.base_path = base_path
        self.image_list = [os.path.join(base_path, img) for img in os.listdir(base_path)]
        print("The length of mydataset_" + mode + " is %d" % len(self.image_list))

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index])
        image_name = self.image_list[index].split("/")[-1]
        # print(image_name)
        if self.is_tta:
            augmented = albu.Compose([albu.HorizontalFlip(p=1),
                                      albu.Normalize(),
                                      AT.ToTensor()])(image=image)
            return augmented['image'], image_name, albu.Compose([albu.Normalize(), AT.ToTensor()])(image=image)['image']
        else:
            augmented = albu.Compose([
                albu.Normalize(), AT.ToTensor()])(image=image)
            return augmented['image'], image_name
    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':

    SEED = 42

    warnings.filterwarnings("ignore")
    # load_path = "/media/user/DATA/all label cos train/log/res34unetv5/2019-11-22, 16:14:04/models/"
    # load_path = "/media/user/DATA/all_labeldata+new_wei/log/res34unetv5/2019-11-22, 23:54:12/best_model/checkpoint.pth.tar"
    # load_path = "/media/user/DATA/2019-11-20(split2)/split2_wei_ensemblemodel/2019-11-29, 22:14:47/best_model/checkpoint.pth.tar"
    # load_path = "/media/user/DATA/2019-11-20(split2)/unet/2019-12-10, 13:15:15/best_model/checkpoint.pth.tar"
    load_path = "../../code/acta/2019-11-20(split2)/poly/v5/best_model/checkpoint.pth.tar"

    # load_path = '/media/user/DATA/2019-11-20(split2)/poly/v5/best_model/checkpoint.pth.tar' 
    # MODEL = deeplabv3plus_efficientnet(pretrained=True)
    # MODEL = EMANet(n_classes=1, n_layers=101)
    # MODEL = UNetv1(n_channels=3,n_classes=1)
    # MODEL =Res34Unetv1()
    MODEL = Res34Unetv5()
    # MODEL = Res34Unetv2()
    # MODEL = DeepLabV3MobileNetV2(1)
    # IMG_PATH = '/media/user/DATA/测试训练集/wei+/分批加入/600/image'
    # OUT_PATH ='/media/user/DATA/测试训练集/wei+/分批加入/600/mask'
    TTA = False
    ensemble = False

    model = MODEL.eval().cuda()
    model = nn.DataParallel(model)
    # model.keep_only_smooth()
    # print(sum(i.numel() for i in model.parameters()))

    IMG_ROOT = "/media/ubuntu/Transcend/tangyang_data/Segmentation0623/"   #BM3D
    # IMG_PATH = '../dataset/images'
    OUT_PATH_BASE ="/media/ubuntu/Transcend/tangyang_data/Segmentation0623/results"
    IMG_LIST = os.listdir(IMG_ROOT)
    IMG_LIST.sort()
    for instance in IMG_LIST:
        # instance = IMG_PATH.split("/")[-1]
        IMG_PATH = os.path.join(IMG_ROOT,instance)
        OUT_PATH = os.path.join(OUT_PATH_BASE,instance)
        # if not os.path.exists(OUT_PATH):
        #     os.makedirs(OUT_PATH)

        MAP_PATH = os.path.join(OUT_PATH_BASE,instance+'_map')
        if not os.path.exists(MAP_PATH):
            os.makedirs(MAP_PATH)
        if ensemble:
            for step in range(5):
                print("Predicing Snapshot:", step)
                null_name = []
                pred_null = []
                pred_flip = []
                param = torch.load(os.path.join(load_path, "checkpoint_epoch" + str(step) + ".pth.tar"))
                model.load_state_dict(param['model'])
                mydataset_test = mydataset(IMG_PATH, is_tta=False, mode='test')
                dataset = DataLoader(mydataset_test, batch_size=1, shuffle=False)
                # Prediction with no TTA test data
                for index, (image, image_name) in enumerate(tqdm(dataset)):
                    with torch.no_grad():
                        mask_pre = model(image.cuda())
                        mask_pre = torch.sigmoid(mask_pre.detach())
                        mask_pre = mask_pre.cpu().numpy().squeeze()
                    pred_null.append(mask_pre)
                    # print(image_name)
                    null_name.append(image_name[0].split('.')[0])


                # Prediction with horizontal flip TTA test data
                if TTA :
                    pseudolabels = {}
                    mydataset_test_TTA = mydataset(IMG_PATH, is_tta=True, mode="test")
                    dataset_tta = DataLoader(mydataset_test_TTA, batch_size=1, shuffle=False)
                    # print("!!!!!")
                    for index, (image, image_name) in enumerate(tqdm(dataset_tta)):
                        with torch.no_grad():
                            mask_pre = model(image.cuda())
                            mask_pre = torch.sigmoid(mask_pre.detach())
                            mask_pre = mask_pre.cpu().numpy().squeeze()
                        mask_pre = cv2.flip(mask_pre, 1)
                        pred_flip.append(mask_pre)

                    pred_null = np.concatenate(pred_null).reshape(-1, h, w)
                    pred_flip = np.concatenate(pred_flip).reshape(-1, h, w)
                    overall_pred += (pred_null + pred_flip) / 2
                else:
                    pred_null = np.concatenate(pred_null).reshape(-1, h, w)
                    # print(pred_null.shape)
                    overall_pred += pred_null
            overall_pred /= 5

            # Save prediction
            for i in range(len(dataset)):
                confidence = (np.sum(overall_pred[i] < 0.2) + np.sum(overall_pred[i] > 0.8)) / (h * w)
                name = null_name[i]
                mask = ((overall_pred[i] > 0.5) * 255).astype(np.uint8)
                pseudolabels[name] = (confidence, mask)
                cv2.imwrite(os.path.join(OUT_PATH, name + '.png'), mask)
            # 输出confidence分数
            # pseudolabels_df = pd.DataFrame(pseudolabels).T
            # pseudolabels_df.columns = ['confidence', 'mask']
            # pd.DataFrame(pseudolabels_df).to_csv(os.path.join(OUT_PATH, "pseudolabels.csv"))
        else:
            param = torch.load(os.path.join(load_path),map_location="cuda:0")
            model.load_state_dict(param['model'])
            if TTA:
                mydataset_test = mydataset(IMG_PATH, is_tta=True, mode='test')
                dataset = DataLoader(mydataset_test, batch_size=1, shuffle=False)
                # Prediction with no TTA test data
                for index, (image, image_name, notta_image) in enumerate(tqdm(dataset)):
                    with torch.no_grad():
                        mask_pre = model(image.cuda())
                        mask_pre = torch.sigmoid(mask_pre.detach())
                        mask_pre = mask_pre.cpu().numpy().squeeze()
                        mask_pre_notta = model(notta_image.cuda())
                        mask_pre_notta = torch.sigmoid(mask_pre_notta.detach()).cpu().numpy().squeeze()
                        mask_pre = cv2.flip(mask_pre, 1)
                        tta_result = (mask_pre_notta + mask_pre) / 2
                        cv2.imwrite(os.path.join(OUT_PATH, image_name[0].split('.')[0]+".png"), ((tta_result>0.5) * 255).astype(np.uint8))


                # Prediction with horizontal flip TTA test data
            else:
                mydataset_test = mydataset(IMG_PATH, is_tta=False, mode='test')
                dataset = DataLoader(mydataset_test, batch_size=1, shuffle=False)
                totaltime = 0
                for index, (image, image_name) in enumerate(tqdm(dataset)):
                    with torch.no_grad():
                        torch.cuda.synchronize()
                        start = time.time()

                        mask_pre = model(image.cuda())
                        torch.cuda.synchronize()
                        end = time.time()

                        totaltime += (end-start)
                        mask_pre = torch.sigmoid(mask_pre.detach())
                    mask_pre = mask_pre.cpu().numpy().squeeze()
                    tta_result = mask_pre
                    torch.cuda.empty_cache()
                    # cv2.imwrite(os.path.join(OUT_PATH, image_name[0].split('.')[0]+".png"), ((tta_result>0.5) * 255).astype(np.uint8))
                    
                    cv2.imwrite(os.path.join(MAP_PATH, image_name[0].split('.')[0]+".png"), ((tta_result) * 255).astype(np.uint8))
                print("len_dataset :", len(dataset))
                print(totaltime/len(dataset))
