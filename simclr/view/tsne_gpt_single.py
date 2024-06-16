import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data_path = r'path\to\feature'
save_path = r'path\to\save'

# 加载数据
f = open(data_path, 'rb')
data = pickle.load(f)

# 标准化数据
X = np.zeros((len(data), 1280))
for i in range(len(data)):
    X[i] = data[i]['val']
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

# 欧几里得距离计算样本间距离
distances = np.zeros((len(data), len(data)))
for i in range(len(data)):
    for j in range(i+1, len(data)):
        distances[i,j] = np.linalg.norm(X[i,:] - X[j,:])
        distances[j,i] = distances[i,j]

# 使用t-SNE算法将1280维特征映射到2维空间
tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, random_state=0)
X_embedded = tsne.fit_transform(distances)

# 绘图
plt.figure(figsize=(18,18), dpi=600)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=300)
plt.axis('off')  # 隐藏坐标轴
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 调整图形边缘
plt.savefig(os.path.join(save_path, 'wsi_numbers' + '.png'), format='png', bbox_inches='tight', pad_inches=0)
plt.close()