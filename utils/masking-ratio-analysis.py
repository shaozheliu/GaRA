import matplotlib.pyplot as plt
# 设置全局字体大小
plt.rcParams.update({'font.size': 16})  # 设置全局字体大小为14
# 数据
params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gara_16_acc = [0.422, 0.422, 0.422, 0.422, 0.422, 0.422, 0.422, 0.422]
gara_32_acc = [0.420, 0.422, 0.422, 0.422, 0.422, 0.422, 0.422, 0.422]

# 创建图形和轴
plt.figure(figsize=(8, 6))
plt.plot(params, gara_16_acc, marker='o', linestyle='--', color='orange', label='GaRA-16')
plt.plot(params, gara_16_acc, marker='s', linestyle='-', color='cyan', label='GaRA-32')

# 设置标题和标签
plt.title('PEFT results under different masking ratio')
plt.xlabel('Masking ratio')
plt.ylabel('Mean Squared Error')
plt.xticks(params)  # 设置x轴的刻度
plt.ylim(0.40, 0.45)  # 设置y轴的范围

# 添加图例
plt.legend()

# 显示图表
# plt.grid()
# plt.tight_layout()
# plt.savefig('./masking-ratio.png', dpi=300, bbox_inches='tight')
plt.show()
