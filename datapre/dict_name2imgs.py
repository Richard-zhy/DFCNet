'''
 为tile生成dict_name2imgs.pkl文件;
 样例：{WSI名:[每个tile绝对地址组成的列表]}
'''
import glob
import os
import pickle
out_base = r'\path\to\pyramid'
ID_list = glob.glob(f"{out_base}/*/*")
# 5倍tile
dict_name2imgs_5 = dict()
temp_list_5 = []
for i in range(len(ID_list)):
    temp_list_5 = glob.glob(f"{ID_list[i]}/*.png")
    dict_name2imgs_5[os.path.basename(ID_list[i])] = temp_list_5
    temp_list_5 = []
file_save = open(r'\path\to\dict_name2imgs_5.pkl', 'wb')
pickle.dump(dict_name2imgs_5, file_save)
file_save.close()
# 20倍
dict_name2imgs_20 = dict()
temp_list_20 = []
for i in range(len(ID_list)):
    temp_list_20 = glob.glob(f"{ID_list[i]}/*/*.png")
    dict_name2imgs_20[os.path.basename(ID_list[i])] = temp_list_20
    temp_list_20 = []
file_save = open(r'\path\to\dict_name2imgs_20.pkl', 'wb')
pickle.dump(dict_name2imgs_20, file_save)
file_save.close()
