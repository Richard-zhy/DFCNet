import os

import pickle
def save_attns(attns_dict, tilenames_dict, save_path, epoch):
    os.makedirs(os.path.join(save_path,'attns'), exist_ok=True)
    attns_file_path = os.path.join(save_path, 'attns','{}_attns_dict.pkl'.format(epoch))
    tilenames_file_path = os.path.join(save_path,'attns', '{}_tilenames_dict.pkl'.format(epoch))
    with open(attns_file_path, 'wb') as file:
        pickle.dump(attns_dict, file)
    with open(tilenames_file_path, 'wb') as file:
        pickle.dump(tilenames_dict, file)

