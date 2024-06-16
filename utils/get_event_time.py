'''
用于获取生存时间，为计算c-index服务
'''
import numpy as np
import pandas as pd
def get_event_time(wsi_name):
    wsi_name_temp = wsi_name.copy()
    label_path = r'\path\to\label.xlsx'
    df = pd.read_excel(label_path)
    c_name = ['手术病理号根治标本', 'RFS_TIME_M']
    df = df[c_name]
    data = np.array(df)
    event_data = []
    for i in range(data.shape[0]):
        data[i][0] = data[i][0].split('.')[0].lower()
    for i in range(len(wsi_name_temp)):
        for j in range(data.shape[0]):
            # 判断wsi id 是否是以副本结束
            if wsi_name_temp[i].endswith("副本"):
                wsi_name_temp[i] = wsi_name_temp[i][:-3]
            if wsi_name_temp[i] in data[j][0]:
                event_data.append(data[j][1])
    return event_data
