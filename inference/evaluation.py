from models.DFN import DFN
from models.poly_lr_schedule import poly_lr_schedule
from models.losses import get_loss, bcelovasz
import torch
import torch.nn as nn
import sys
from unet_model import *
from unet_models import *
from unet_mb2 import *
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
from deeplabv3plus import deeplabv3plus_efficientnet
from network import EMANet
SEED = 42

warnings.filterwarnings("ignore")
# load_path = "/media/user/DATA/all label cos train/log/res34unetv5/2019-11-22, 16:14:04/models/"
# no ensemble：the .pth path；with ensemble: the folder path
load_path = '/media/user/DATA/all label cos train/log/res34unetv5/2019-11-23, 05:00:15/models/checkpoint_epoch4.pth.tar'  
# MODEL = EMANet(n_classes=1, n_layers=101)
MODEL = DeepLabV3MobileNetV2(1)
# MODEL = Res34Unetv5()
# MODEL = UNet(3,1)
# MODEL = UNetv1(3,1)
IMG_PATH = '/media/user/Transcend/data(denoised)/NEW_17'
OUT_PATH ="/media/user/DATA/demo/img"
TTA = False
ensemble = False
ALPHA = 1

class Logger():
    def __init__(self,path=OUT_PATH):
        super().__init__()
        self.path = path
        self.file = None
    def write(self, str):
        self.file = open(os.path.join(self.path,"log.txt"),"a")
        self.file.write(str)
        self.file.close()
    def stop():
        self.file.close()


class measure(object):
    def __init__(self):
        super(measure, self).__init__()
        self.alpha = ALPHA

    def total_measure(self, truth, pred):
        pred_ratio = pred > 0.5
        # print(pred.shape)
        f_score_dict = self.f_measure(truth, pred_ratio)
        MAE_score = self.MAE(truth, pred)
        iou = self.cal_iou(pred_ratio, truth)
        return {
            'accuracy': f_score_dict['accuracy'],
            'precision': f_score_dict['precision'],
            'recall': f_score_dict['recall'],
            'f_score': f_score_dict['f_score'],
            'MAE_score': MAE_score,
            "iou" : iou
        }

    def f_measure(self, truth, pred, F_alpha=None):
        if F_alpha is None:
            alpha = self.alpha
        TP = truth * pred
        TN = (1 - truth) * (1 - pred)
        # FP = (1 - truth) * pred
        # accuracy = TP.sum() / ones.sum()
        accuracy = (TP.sum() + TN.sum()) / (np.ones_like(truth).sum())
        precision = TP.sum() / pred.sum() if pred.sum() != 0 else 0   # precision = TP / (TP + FP)
        recall = TP.sum() / truth.sum() if truth.sum() != 0 else 0    # recall = TP / (TP + FN)
        f_measure = (alpha**2 + 1) * precision * recall / (alpha**2 * (recall + precision)) if recall + precision != 0 else 0
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f_score': f_measure
        }

    def MAE(self, truth, pred):
        return np.abs(truth - pred).mean()


    def cal_iou(self, pred, truth):
        TP = truth * pred
        FP = (1 - truth) * pred
        FN = truth * (1 - pred)
        ones = np.zeros(TP.shape, np.uint8)

        ones[TP == 1] = 1
        ones[FP == 1] = 1
        ones[FN == 1] = 1
        iou_measure = TP.sum() / (ones.sum() + 1e-8)
        return iou_measure


class mydataset(Dataset):
    def __init__(self, base_path, is_tta=False, mode='test'):
        super(mydataset, self).__init__()
        self.is_tta = is_tta
        self.mode = mode
        self.base_path = base_path
        self.img_name = os.listdir(base_path)
        print("The length of mydataset_" + mode + " is %d" % len(self.img_name))

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.base_path, self.img_name[index]))
        if self.img_name[index].startswith('aug_'):
            image_name = "aug_mask.tif"
        elif self.img_name[index].startswith("de_"):
            image_name = self.img_name[index][3:]
        else:
            image_name = self.img_name[index]

        mask = cv2.imread(os.path.join(os.path.dirname(self.base_path), "mask",
                                       image_name.replace("tif", "png")),
                          cv2.IMREAD_GRAYSCALE)
        image_name = self.img_name[index].split("/")[-1]
        if self.is_tta:
            augmented = albu.Compose([albu.HorizontalFlip(p=1),
                                      albu.Normalize(),
                                      AT.ToTensor()])(image=image)
            return augmented['image'], image_name, albu.Compose([albu.Normalize(), AT.ToTensor()])(image=image)['image'], mask
        else:
            augmented = albu.Compose([
                albu.Normalize(), AT.ToTensor()])(image=image)
            return augmented['image'], image_name, mask

    def __len__(self):
        return len(self.img_name)


if __name__ == '__main__':
    measure = measure()
    model = MODEL.eval().cuda()
    model = nn.DataParallel(model)
    if ensemble:
        log = Logger()
        result={}
        iou = 0
        pixel_acc = 0
        precision = 0
        recall = 0
        f_score = 0
        MAE_score = 0
        models = []
        if TTA:
            for i in range(len(os.listdir(load_path))):
                param = torch.load(os.path.join(load_path, 'checkpoint_epoch' + str(i) + '.pth.tar'))
                model.load_state_dict(param['model'])
                models.append(model)
            mydataset_test = mydataset(IMG_PATH, is_tta=True, mode='test')
            dataset = DataLoader(mydataset_test, batch_size=1, shuffle=False)
            # Prediction with  TTA test data
            for index, (image, image_name, notta_image, ori_mask) in enumerate(tqdm(dataset)):
                mask = np.zeros_like(ori_mask.squeeze()).astype(np.float32)
                for net in models:
                    with torch.no_grad():
                        mask_pre = net(image.cuda())
                        mask_pre = torch.sigmoid(mask_pre.detach())
                        mask_pre = mask_pre.cpu().numpy().squeeze()
                        mask_pre_notta = net(notta_image.cuda())
                        mask_pre_notta = torch.sigmoid(mask_pre_notta.detach()).cpu().numpy().squeeze()
                    mask_pre = (cv2.flip(mask_pre, 1) + mask_pre_notta) / 2
                    mask += mask_pre
                mask /= len(models)
                cv2.imwrite(os.path.join(OUT_PATH, image_name[0] + '.png'), ((mask > 0.5) * 255).astype(np.uint8))
                measure_dict = measure.total_measure((ori_mask.detach().numpy().squeeze()) / 255, mask)
                pixel_acc += measure_dict["accuracy"]
                precision += measure_dict['precision']
                recall += measure_dict['recall']
                iou += measure_dict['iou']
                f_score += measure_dict['f_score']
                MAE_score += measure_dict['MAE_score']
            result['pixel_acc'] = pixel_acc / len(dataset)
            result['precision'] = precision / len(dataset)
            result['recall'] = recall / len(dataset)
            result['iou'] = iou / len(dataset)
            result['f_score'] = f_score / len(dataset)
            result['MAE_score'] = MAE_score / len(dataset)
            log.write('pixel_acc     | precision        | recall        | iou         | f_score     | MAE_score \n')
            log.write('---------------------------------------------------------------------------------------\n')
            log.write('%0.4f         | %0.4f           | %0.4f         | %0.4f       | %0.4f       | %0.4f      \n' %
                      (result['pixel_acc'], result['precision'], result['recall'], result['iou'], result['f_score'],
                       result['MAE_score']))
            print("done!!!")
        else:
            for i in range(len(os.listdir(load_path))):
                param = torch.load(os.path.join(load_path, 'checkpoint_epoch' + str(i) + '.pth.tar'))
                model.load_state_dict(param['model'])
                models.append(model)

            mydataset_test = mydataset(IMG_PATH, is_tta=False, mode='test')
            dataset = DataLoader(mydataset_test, batch_size=1, shuffle=False)
            # Prediction with no TTA test data
            for index, (image, image_name, ori_mask) in enumerate(tqdm(dataset)):
                mask = np.zeros_like(ori_mask.squeeze()).astype(np.float32)
                for net in models:
                    with torch.no_grad():
                        mask_pre = net(image.cuda())
                        mask_pre = torch.sigmoid(mask_pre.detach())
                        mask_pre = mask_pre.cpu().numpy().squeeze()
                    mask += mask_pre
                mask /= len(models)
                cv2.imwrite(os.path.join(OUT_PATH, image_name[0] + '.png'), ((mask > 0.5) * 255).astype(np.uint8))
                measure_dict = measure.total_measure((ori_mask.detach().numpy().squeeze()) / 255, mask)
                pixel_acc += measure_dict["accuracy"]
                precision += measure_dict['precision']
                recall += measure_dict['recall']
                iou += measure_dict['iou']
                f_score += measure_dict['f_score']
                MAE_score += measure_dict['MAE_score']
            result['pixel_acc'] = pixel_acc / len(dataset)
            result['precision'] = precision / len(dataset)
            result['recall'] = recall / len(dataset)
            result['iou'] = iou / len(dataset)
            result['f_score'] = f_score / len(dataset)
            result['MAE_score'] = MAE_score / len(dataset)
            log.write('pixel_acc     | precision        | recall        | iou         | f_score     | MAE_score \n')
            log.write('---------------------------------------------------------------------------------------\n')
            log.write('%0.4f         | %0.4f           | %0.4f         | %0.4f       | %0.4f       | %0.4f      \n' %
                      (result['pixel_acc'], result['precision'], result['recall'], result['iou'], result['f_score'],
                       result['MAE_score']))
            print("done!!!")
    else:
        param = torch.load(os.path.join(load_path), map_location='cuda:0')
        # param = torch.load(os.path.join(load_path))
        model.load_state_dict(param['model'])
        log = Logger()
        result={}
        iou = 0
        pixel_acc = 0
        precision = 0
        recall = 0
        f_score = 0
        MAE_score = 0
        if TTA:
            img_list = {}
            # Prediction with TTA test data
            mydataset_test = mydataset(IMG_PATH, is_tta=True, mode='test')
            dataset = DataLoader(mydataset_test, batch_size=1, shuffle=False)
            for index, (image, image_name, notta_image, ori_mask) in enumerate(tqdm(dataset)):
                with torch.no_grad():
                    mask_pre = model(image.cuda())
                    mask_pre = torch.sigmoid(mask_pre.detach())  # 0-1之间
                    mask_pre = mask_pre.cpu().numpy().squeeze()
                    mask_pre_notta = model(notta_image.cuda())
                    mask_pre_notta = torch.sigmoid(mask_pre_notta.detach()).cpu().numpy().squeeze()
                mask_pre = cv2.flip(mask_pre, 1)
                tta_result = (mask_pre_notta + mask_pre) / 2
                cv2.imwrite(os.path.join(OUT_PATH, image_name[0] + '.png'), (tta_result * 255).astype(np.uint8))
                measure_dict = measure.total_measure((ori_mask.detach().numpy().squeeze()) / 255, tta_result)
                pixel_acc += measure_dict["accuracy"]
                precision += measure_dict['precision']
                recall += measure_dict['recall']
                iou += measure_dict['iou']
                f_score += measure_dict['f_score']
                MAE_score += measure_dict['MAE_score']
        else:
            mydataset_test = mydataset(IMG_PATH, is_tta=False, mode='test')
            dataset = DataLoader(mydataset_test, batch_size=1, shuffle=False)
            # Prediction with no TTA test data
            for index, (image, image_name, ori_mask) in enumerate(tqdm(dataset)):
                with torch.no_grad():
                    mask_pre = model(image.cuda())
                    mask_pre = torch.sigmoid(mask_pre.detach())    # 0-1之间
                    mask_pre = mask_pre.cpu().numpy().squeeze()
                cv2.imwrite(os.path.join(OUT_PATH, image_name[0].replace('tif','tif')),((mask_pre>0.5)*255).astype(np.uint8))
                cv2.imwrite(os.path.join(OUT_PATH, image_name[0].replace('tif','png')),((mask_pre) * 255).astype(np.uint8))
                measure_dict = measure.total_measure((ori_mask.detach().numpy().squeeze())/255, mask_pre)
                pixel_acc += measure_dict["accuracy"]
                precision += measure_dict['precision']
                recall    += measure_dict['recall']
                iou       += measure_dict['iou']
                f_score   += measure_dict['f_score']
                MAE_score += measure_dict['MAE_score']

        result['pixel_acc'] = pixel_acc/len(dataset)
        result['precision'] = precision / len(dataset)
        result['recall']    = recall / len(dataset)
        result['iou']       = iou / len(dataset)
        result['f_score']   = f_score / len(dataset)
        result['MAE_score'] = MAE_score / len(dataset)
        log.write('pixel_acc     | precision        | recall        | iou         | f_score     | MAE_score \n')
        log.write('---------------------------------------------------------------------------------------\n')
        log.write('%0.4f         | %0.4f           | %0.4f         | %0.4f       | %0.4f       | %0.4f      \n' %
              (result['pixel_acc'], result['precision'], result['recall'], result['iou'], result['f_score'], result['MAE_score']))
        print("done!!!")
