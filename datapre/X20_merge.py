import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import os.path as osp
import pickle
from glob import glob
from multiprocessing import Pool

from tqdm import tqdm
import rich
from rich import print
from rich.progress import track
import argparse

def merge_wsi_feat(wsi_feat_dir) -> None:
    files = glob(osp.join(wsi_feat_dir, '*/*.pkl'))
    save_obj = []
    for fp in files:
        try:
            with open(fp, 'rb') as infile:
                obj = pickle.load(infile)
            # add patch name
            sub_name = osp.basename(osp.dirname(fp))
            obj['feat_name'] = sub_name + '+' + osp.basename(fp).rsplit('.', 1)[0]
            save_obj.append(obj)
        except Exception as e:
            print(f'Error in {fp} as {e}')
            continue

    bname = osp.basename(wsi_feat_dir).lower()  # wsi id
    merge_feat_save_dir = r'path\to\wsi_x20loss'

    save_fp = osp.join(merge_feat_save_dir, f'{bname}.pkl')
    with open(save_fp, 'wb') as outfile:
        pickle.dump(save_obj, outfile)

import yaml
def read_yaml(fpath=None):
    with open(fpath, mode="r", encoding='utf-8') as file:
        yml = yaml.safe_load(file)
        return yml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="WSI patch features extraction"
    )
    parser.add_argument("--cfg", default=r'path\to\cfg_for_extract_merge.yaml',
                        metavar="FILE", help="path to config file", type=str,)
    parser.add_argument("--workers", default=8,  help="number of the workers", type=int,)
    args = parser.parse_args()
    cfg = read_yaml(args.cfg)
    # tile级别存储路径
    feat_save_dir_X20 = cfg['savedir_20']
    # wsi 级别保存路径
    merge_feat_save_dir_X20 = cfg['savedir_com20']
    for feat_save_dir, merge_feat_save_dir in zip([feat_save_dir_X20],[merge_feat_save_dir_X20]):
        print(f'Save to {merge_feat_save_dir}')
        # 创建存储文件夹  将所有Tile文件路径放在一起
        os.makedirs(merge_feat_save_dir, exist_ok=True)
        wsi_dirs = glob(osp.join(feat_save_dir, '*'))
        with Pool(args.workers) as p:
            # 根据Tile路径，使用merge_wsi_feat方法进行存储
            for _ in track(p.imap_unordered(merge_wsi_feat, wsi_dirs), total=len(wsi_dirs)):
                print("程序正在执行，多线程开启")