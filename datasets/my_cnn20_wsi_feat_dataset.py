# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data_utils
import numpy as np
import pickle
class WSIFeatDataset(data_utils.Dataset):
    def __init__(self, ann_file=None, transforms=None,
                 is_training:bool=False):
        super().__init__()
        with open(ann_file, 'rt') as infile:
            data = infile.readlines()
        self.wsi_loc = []
        count = 0
        for each in data:
            count += 1
            wsi_name, wsi_path, label = each.strip().split(',')
            self.wsi_loc.append((wsi_name, wsi_path, label))
            # 复制数据集
            if is_training == True:
                for i in range(6):
                    self.wsi_loc.append((wsi_name+"{}副本".format(i), wsi_path, label))
            if is_training == False:
                for i in range(1):
                    self.wsi_loc.append((wsi_name+"{}副本".format(i), wsi_path, label))
        sorted(self.wsi_loc, key=lambda x: x[0])
        self.wsiName_2_numberid = {}
        for numberid, wsi_info in enumerate(self.wsi_loc):
            self.wsiName_2_numberid[wsi_info[0]] = numberid  # 确定当前这个wsi_name是第几个数据
        self._transforms = transforms
        self.is_training = is_training

    def __len__(self):
        return len(self.wsi_loc)

    def __getitem__(self, index):
        wai_name, wsi_path, label = self.wsi_loc[index]
        if len(label)>1:
            label = list(map(int, label))
        else:
            label = int(label)
        label = torch.tensor(label).long()
        with open(wsi_path, 'rb') as infile:
            patch_feat = pickle.load(infile)
        hw = np.ceil(np.sqrt(len(patch_feat)))
        x = hw ** 2 - len(patch_feat)
        for i in range(int(x)):
            patch_feat.append(patch_feat[i])
        sorted_lst = sorted(patch_feat,
                            key=lambda x: (int(x['tile_name'].split('+')[1].split('_')[0]), int(x['tile_name'].split('+')[1].split('_')[1])))
        t = np.zeros((int(hw), int(hw), len(patch_feat[1]['feat'])))
        for i in range(int(hw)):
            for j in range(int(hw)):
                t[j][i] = sorted_lst[int(hw) * i + j]['feat']

        target = {'wsi_label': label, 'wsi_numerid': self.wsiName_2_numberid[wai_name], 'wsi_name': wai_name}
        feat_map = t.copy()
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