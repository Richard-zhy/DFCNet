
def build_dataset_5(input_data_file, image_set, is_training):
    from .my_vit_wsi_feat_dataset import build as build_wsi_feat_dataset
    return build_wsi_feat_dataset(input_data_file, image_set,is_training)

def build_dataset_20(input_data_file, image_set, is_training):
    from .my_cnn20_wsi_feat_dataset import build as build_wsi_feat_dataset
    return build_wsi_feat_dataset(input_data_file, image_set,is_training)
