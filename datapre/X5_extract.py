import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import cv2
import pickle
import argparse
import numpy as np
import os.path as osp
from PIL import Image
from rich import progress
from concurrent.futures import ThreadPoolExecutor
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import albumentations as alb
from albumentations.pytorch import ToTensorV2

# 要使用交叉验证 每个数据都会做train和val，所以提取的时候也要提取其train和val两种
tr_trans = alb.Compose([
    alb.Resize(224, 224),
    alb.RandomRotate90(),
    alb.RandomBrightnessContrast(),
    alb.HueSaturationValue(),
    alb.HorizontalFlip(),
    alb.VerticalFlip(),
    alb.CoarseDropout(max_holes=4),
    alb.Normalize(),
    ToTensorV2(),
])

val_trans = alb.Compose([
    alb.Resize(224, 224),
    alb.Normalize(),
    ToTensorV2(),
])


class PatchDataset(data_utils.Dataset):
    def __init__(self, img_fp_list, trans):
        print("dataset for imagenet")
        self.img_fp_list = img_fp_list
        self.trans = trans

    def __len__(self):
        return len(self.img_fp_list)

    def __getitem__(self, idx):
        try:
            img_fp = self.img_fp_list[idx]
            img = cv2.imread(img_fp)
            img = img[:, :, ::-1]
        except:
            # if img is None:
            img = np.zeros((224, 224, 3)).astype('uint8')

        pid = osp.basename(osp.dirname(img_fp))
        img_bname = osp.basename(img_fp).rsplit('.', 1)[0]
        aug_img = self.trans(image=img)['image']
        return pid, img_bname, aug_img


class PatchDatasetCLR(data_utils.Dataset):
    def __init__(self, img_fp_list):
        print("dataset for CLR")
        self.img_fp_list = img_fp_list

    def __len__(self):
        return len(self.img_fp_list)

    def __getitem__(self, idx):
        try:
            img_fp = self.img_fp_list[idx]
            img_pil = Image.open(img_fp)
            img = cv2.resize(np.array(img_pil), (224, 224))
        except:
            # if img is None:
            img = np.zeros((224, 224, 3)).astype('uint8')
        pid = osp.basename(osp.dirname(img_fp))
        img_bname = osp.basename(img_fp).rsplit('.', 1)[0]
        aug_img = transforms.functional.to_tensor(img)
        return pid, img_bname, aug_img

def save_val_feat_in_thread(batch_pid, batch_img_bname, batch_val_feat, save_dir):
    for b_idx, (pid, img_bname, val_feat) in enumerate(zip(batch_pid, batch_img_bname, batch_val_feat)):
        feat_save_dir = osp.join(save_dir, pid)
        os.makedirs(feat_save_dir, exist_ok=True)
        feat_save_name = osp.join(feat_save_dir, f'{img_bname}.pkl')
        if osp.exists(feat_save_name):
            with open(feat_save_name, 'rb') as infile:
                save_dict = pickle.load(infile)
        else:
            save_dict = {}
        save_dict['val'] = val_feat
        with open(feat_save_name, 'wb') as outfile:
            pickle.dump(save_dict, outfile)

def pred_and_save_with_dataloader(model, img_fp_list, save_dir, batch_size=1, dataset_class="imagenet"):
    print("start dataloader..")
    model.eval()
    import multiprocessing
    num_processes = multiprocessing.cpu_count()
    num_processes = min(48, num_processes)
    executor = ThreadPoolExecutor(max_workers=num_processes)

    # batch-size 指定为1 将每个tile单独做结果，便于后续单独操作
    val_dl = torch.utils.data.DataLoader(
        PatchDataset(img_fp_list, trans=val_trans) if dataset_class == "imagenet" else PatchDatasetCLR(img_fp_list),
        batch_size=batch_size,
        # num_workers=num_processes,
        num_workers=8,  # 太大会出错
        shuffle=False,
        drop_last=False
    )
    print("开始提取")
    for batch in progress.track(val_dl):
        batch_pid, batch_img_bname, batch_tr_img = batch
        batch_tr_img = batch_tr_img.cuda()

        with torch.no_grad():
            val_feat = model(batch_tr_img)
            val_feat = val_feat.cpu().numpy()
            # 启用多线程来保存数据
            executor.submit(save_val_feat_in_thread, batch_pid, batch_img_bname, val_feat, save_dir)
    print("task finished..")

# 使用ImageNet权重提取
class EffNet(nn.Module):
    def __init__(self, efname='efficientnet_v2_s', model_path=""):
        super(EffNet, self).__init__()
        # 使用对比学习权重
        self.model = models.efficientnet_v2_s(pretrained=False)
        print(f'Load pretrain model from {model_path}')
        self.model.load_state_dict(torch.load(model_path),strict=False)
        print("model created")

    def forward(self, data):
        bs = data.shape[0]
        feat = self.model.features(data)
        feat = nn.functional.adaptive_avg_pool2d(feat, output_size=(1))
        feat = feat.view(bs, -1)
        return feat

def get_model(model_name="", model_path="", num_classes=1000):
    if model_name == "efficientnet_v2_s":
        model = EffNet(efname='efficientnet_v2_s', model_path=model_path)
    else:
        print(f'Load pretrain model from {model_path}')
    return model

import yaml
def read_yaml(fpath=None):
    with open(fpath, mode="r", encoding='utf-8') as file:
        yml = yaml.safe_load(file)
        return yml
# %%
if __name__ == "__main__":

    # add config params
    parser = argparse.ArgumentParser(description="WSI patch features extraction")
    parser.add_argument("--cfg", default=r'\path\to\cfg_for_extract_merge.yaml',
                        metavar="FILE", help="path to config file", type=str, )
    args = parser.parse_args()
    cfg = read_yaml(args.cfg)
    model_name = cfg['model_name']
    model_path = cfg['weight_5']
    save_dir = cfg['savedir_5']
    os.makedirs(save_dir, exist_ok=True)
    # creat model
    model = get_model(model_name=model_name, model_path=model_path)  # efficientnet
    model = nn.DataParallel(model).cuda()
    # load path of original images
    wsi_patch_info = cfg['idx2img_5']
    print("load img path {}".format(wsi_patch_info))
    img_fp_list = []
    with open(wsi_patch_info, "rb") as fp:
        dt = pickle.load(fp)
        for k, v in dt.items():
            img_fp_list.extend(v)
    print(f'Len of img {len(img_fp_list)}')
    img_fp_list = sorted(img_fp_list)
    # extract feature to *.pkl
    pred_and_save_with_dataloader(model, img_fp_list, save_dir=save_dir, batch_size=1)