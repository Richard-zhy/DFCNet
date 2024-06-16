import glob
import os
import pickle as pk
import datetime
import pandas as pd
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from collections.abc import Iterable

class FeatFusion(object):
    def __init__(self, X5_path, X20_path, sim_type, wsi_name=None):
        self.X5_path = X5_path
        self.X20_path = X20_path
        # self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sim_type = sim_type
        self.all_wsi_name = wsi_name

    def load_data(self, mag=5):
        assert mag==5 or mag==20 , "放大倍率不对"
        all_wsi_name = []
        if mag == 5:
            all_wsi = glob.glob(osp.join(self.X5_path,'*.pkl'))
        if mag == 20:
            all_wsi = glob.glob(osp.join(self.X20_path,'*.pkl'))

            for i in range(len(all_wsi)):
                name = os.path.basename(all_wsi[i]).split('.')[0]
                all_wsi_name.append(name)

        all_tile_list = []
        for index, path in enumerate(all_wsi):
            f = open(path, 'rb')
            data = pk.load(f)
            all_tile_list.append(data)

        return all_tile_list , all_wsi_name

    def cal_weight(self, wsi_list_5, wsi_list_20, all_wsi_name):
        # 对所有wsi做操作
        processed = 0
        '''所有wsi'''
        for i in range(len(wsi_list_5)):
            # 拿出单个低倍进行操作
            score_list = []
            start_index = 0
            end_index = 16
            current_wsi_all_feat = []
            low_numbers = len(wsi_list_5[i])
            '''wsi下的低倍率个数'''
            add_flag = False
            break_flag = False
            for j in range(low_numbers):
                check_count = 0
                x5_name = wsi_list_5[i][j]['feat_name']
                x20_stack = np.zeros([1,1280])
                x20_name_list = []
                ''' 寻找tile间对应关系'''
                high_numbers = len(wsi_list_20[i])
                for k in range(start_index, min(high_numbers, end_index)):
                    parent_name = wsi_list_20[i][k]['feat_name'].split('+')[0]
                    # 低倍率与高倍率之间无对应关系【一个对应的都没有】
                    if start_index == k and parent_name != x5_name:
                        print(f"WSI编号{all_wsi_name[i]}的低分辨率切片{wsi_list_5[i][j]['feat_name']}无对应高分辨率数据，已跳过，从下一个低倍继续开始")
                        break_flag = True
                        break
                    '''低倍率与高倍率有对应关系'''
                    if parent_name == x5_name:
                        check_count += 1
                        x20_name_list.append(wsi_list_20[i][k]['feat_name'])
                        if check_count == 1:
                            x20_stack = wsi_list_20[i][k]['val'].reshape(1,1280)
                        else:
                            x20_stack = np.concatenate([x20_stack, wsi_list_20[i][k]['val'].reshape(1,1280)],axis=0)

                if break_flag == True:
                    break_flag = False
                    continue

                # 更新下一轮对应起始上下标
                start_index += check_count
                end_index += check_count
                ''' 当前该组tile 相似度得分计算 并给出融合后的特征 '''
                score, fusion_feature = self.compute_similarity(wsi_list_5[i][j]['val'], x20_stack, sim_type=self.sim_type)

                if len(score) == 1:
                    score = score[0]
                else:
                    score = torch.squeeze(score)

                try:
                    score_list.extend(score)
                except Exception:
                    print('出现单个分数，未添加，无影响，分数为{}'.format(score))
                for f in range(fusion_feature.shape[0]):
                    cur_dict = {}
                    cur_dict['feat'] = np.array(fusion_feature[f])
                    cur_dict['tile_name'] = x20_name_list[f]
                    current_wsi_all_feat.append(cur_dict)
            '''保存当前wsi的特征'''
            ''' 将当前wsi的进行保存，调用保存方法'''
            self.save_feat(current_wsi_all_feat, all_wsi_name[i],'./')
            processed += 1
            print(f"处理完{processed}张WSI, 当前时间{datetime.datetime.now()}, 最大值最小值分别为：")

            print(max(score_list))
            print(min(score_list))
        print("所有特征已保存")

    def compute_similarity(self, a, b, sim_type='cos'):
        if a.dtype!='float32':
            a = a.astype('float32')
        if b.dtype!='float32':
            b = b.astype('float32')

        assert sim_type=='cos' or sim_type =='dot', "选择的相似度类型有无，选择cos 或 dot"
        if sim_type == 'cos':
            a = torch.unsqueeze(torch.tensor(a), dim=0)
            b = torch.tensor(b)
            cos = F.cosine_similarity(a, b)
            similarity = F.softmax(cos, dim=0)
            return similarity
        if sim_type == 'dot':
            result = np.zeros((1, b.shape[0]))
            for i in range(b.shape[0]):
                dot_product = np.dot(a, b[i])
                result[0][i] = dot_product
            # 归一化
            if max(result[0]) == min(result[0]):
                final_score = F.softmax(torch.tensor(result[0]),  dim=0)
            else: # 正常数据正常处理
                maxmin_norm = (result[0] - min(result[0])) / (max(result[0]) - min(result[0]))
                simscore = torch.tensor(maxmin_norm)
                final_score = F.softmax(simscore, dim=0)
            a = torch.tensor(a)
            b = torch.tensor(b)
            for i in range(final_score.shape[0]):
                b[i] = b[i] * final_score[i] + a
            return final_score, b

    def save_feat(self, feat, feat_name, path):
        # 创建路径
        fusion_path = osp.join(path, 'save_fusion_feat');
        if osp.exists(fusion_path) is False:
            os.makedirs(fusion_path)
        # 保存
        save_fp = osp.join(fusion_path, f'{feat_name}.pkl')
        with open(save_fp, 'wb') as outfile:
            pk.dump(feat, outfile)
        outfile.close()

if __name__ == '__main__':
    # 实际数据
    x5_path = r'path\to\result\wsi_x5loss'
    x20_path = r'path\to\result\wsi_x20imagenet'
    my_data = FeatFusion(x5_path, x20_path, sim_type='dot')
    wsi_list_5,_ = my_data.load_data(mag=5)
    wsi_list_20_data, wsi_list_20_name = my_data.load_data(mag=20)
    my_data.cal_weight(wsi_list_5, wsi_list_20_data, wsi_list_20_name)


