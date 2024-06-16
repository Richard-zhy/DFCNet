# 准备工作
1. 基于DSMIL的[deepzoom_tiler.py](https://github.com/binli123/dsmil-wsi/blob/master/deepzoom_tiler.py)对WSI进行切分处理
# 阶段1 特征解耦
1. 运行`/simclr/run`，使用准备工作中获得的数据训练特征解耦模型
2. 运行`/datapre/dict_name2imgs`，获得所有切片的绝对路径
3. 运行`/datapre/X5_extract`使用获得的解耦模型提取低分辨率切片特征（本文以5倍为例）
3. 运行`/datapre/X5_merge`，将切片特征聚合为wsi级特征
# 阶段2 多分辨率融合
1. 运行`/datapre/X20_extract`基于ImageNet提取高分辨率切片特征（本文以20倍为例）
2. 运行`/datapre/X20_merge`将低分辨率特征与高分辨率特征进行多分辨率融合，并聚合为wsi级特征
2. 运行`/datapre/FeatFusion`融合不同分辨率特征
# 阶段3
1. 对不同分辨率下的特征进行预测，使用Resnet获取局部特征，使用条件位置编码搭配ViT获取全局特征
2. 获得最终的预测结果
