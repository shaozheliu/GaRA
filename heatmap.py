import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置全局字体大小
plt.rcParams.update({'font.size': 14})  # 设置全局字体大小为14
rcParams['font.family'] = 'Times New Roman'  # 使用字体中的无衬线体

# 数据
data = np.array([
    [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    [16, 16, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    [16, 16, 15, 14, 16, 16, 15, 16, 16, 15, 15, 11, 16, 16, 16, 14, 16, 16, 16, 14, 14, 12, 9, 9],
    [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 14, 16, 16, 15, 13, 10, 4, 2, 0, 0, 0, 0, 0],
])

# 创建 DataFrame
df = pd.DataFrame(data, index=['$W_q$', '$W_k$', '$W_v$', '$W_o$'], columns=np.arange(1, 25))

# 创建图形和子图
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))

# 第一个热图
sns.heatmap(df, cmap='RdBu_r', annot=True, fmt='d', linewidths=0.5, ax=axs[0])
axs[0].set_title('Rank Heatmap 1', fontsize=18)
axs[0].set_xlabel('Layer', fontsize=18)
axs[0].set_ylabel('Weights', fontsize=18)

# 第二个热图（可以使用不同的颜色映射进行区分）
sns.heatmap(df, cmap='coolwarm', annot=True, fmt='d', linewidths=0.5, ax=axs[1])
axs[1].set_title('Rank Heatmap 2', fontsize=18)
axs[1].set_xlabel('Layer', fontsize=18)
axs[1].set_ylabel('Weights', fontsize=18)

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('./final_rank_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()
