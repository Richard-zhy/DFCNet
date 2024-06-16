# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data_utils
import numpy as np
import pickle
import torchvision.transforms
# 自定义数据操作
class WSIFeatDataset(data_utils.Dataset):
    def __init__(self, ann_file=None, transforms=None,
                 is_training:bool=False):
        super().__init__()
        # 获取txt对应的所有数据地址
        with open(ann_file, 'rt') as infile:
            data = infile.readlines()
        self.wsi_loc = []
        count = 0
        for each in data:
            count += 1
            wsi_name, wsi_path, label = each.strip().split(',')
            self.wsi_loc.append((wsi_name, wsi_path, label))
        sorted(self.wsi_loc, key=lambda x: x[0])
        self.wsiName_2_numberid = {}
        for numberid, wsi_info in enumerate(self.wsi_loc):
            self.wsiName_2_numberid[wsi_info[0]] = numberid  # 确定当前这个wsi_name是第几个数据
        self._transforms = transforms
        self.is_training = is_training


    def __len__(self):
        return len(self.wsi_loc)

    def __getitem__(self, index):
        # 获得数据
        wai_name, wsi_path, label = self.wsi_loc[index]
        if len(label)>1: # 防止多分类情况下，类别数>9, 使用map将字符串变成int
            label = list(map(int, label))
        else:
            label = int(label)
        label = torch.tensor(label).long()
        # 还原切片到原始位置
        with open(wsi_path, 'rb') as infile:
            patch_feat = pickle.load(infile)
        hw = np.ceil(np.sqrt(len(patch_feat)))
        x = hw ** 2 - len(patch_feat)
        for i in range(int(x)):
            patch_feat.append(patch_feat[i])
        sorted_lst = sorted(patch_feat,
                            key=lambda x: (int(x['feat_name'].split('_')[0]), int(x['feat_name'].split('_')[1])))
        t = np.zeros((int(hw), int(hw), len(patch_feat[1]['val'])))
        for i in range(int(hw)):
            for j in range(int(hw)):
                t[j][i] = sorted_lst[int(hw) * i + j]['val']

        tile_names = np.empty((int(hw), int(hw)), dtype=object)
        for i in range(int(hw)):
            for j in range(int(hw)):
                tile_names[j][i] = sorted_lst[int(hw) * i + j]['feat_name']
        target = {'wsi_label': label, 'wsi_numerid': self.wsiName_2_numberid[wai_name], 'wsi_name': wai_name}
        feat_map = t.copy()
        feat_map = feat_map.reshape(-1, len(feat_map[0][0]), order='F')
        tile_names = tile_names.reshape(-1, order='F')
        tile_names = tile_names.tolist()
        target.update({'one_tilenames':tile_names})
        return feat_map, target

class TrFeatureMapAug:
    def __call__(self, feat, p=0.5):
        feat = self.horizontal_flip(feat)
        feat = self.vertical_flip(feat)
        rot_k = np.random.choice([-1, 1])
        feat = np.rot90(feat, k=rot_k, axes=(0, 1))
        return feat

    def horizontal_flip(self, image, rate=0.5):
        if np.random.rand() < rate:
            image = image[:, ::-1, :]
        return image

    def vertical_flip(self, image, rate=0.5):
        if np.random.rand() < rate:
            image = image[::-1, :, :]
        return image


class ValFeatureMapAug:
    def __call__(self, feat):
        feat = self.horizontal_flip(feat)
        return feat

    def horizontal_flip(self, image):
        image = image[:, ::-1, :]
        return image

def make_wsi_transforms(image_set):
    if image_set == 'train':
        return TrFeatureMapAug()
    else:
        return ValFeatureMapAug()

def build(input_data_file, image_set, is_training:bool):
    dataset = WSIFeatDataset(ann_file=input_data_file,
                             transforms=make_wsi_transforms(image_set),
                             is_training=is_training)
    return dataset