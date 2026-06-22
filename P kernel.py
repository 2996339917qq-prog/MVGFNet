import matplotlib.pyplot as plt

# 卷积核数量
kernels = [1, 2, 4]

# 假设的 MAE 值（请替换为实际实验数据）
mae_pems07 = [19.45, 19.36, 19.22]
mae_pems08 = [13.33, 13.23, 13.28]

# 创建一行两列子图
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# PEMS07 折线图
axes[0].plot(kernels, mae_pems07, marker='o', linestyle='-', color='tab:blue')
axes[0].set_xticks(kernels)
axes[0].set_xlabel('Number of Convolution Kernels (P)', fontsize=16)
axes[0].set_ylabel('MAE', fontsize=16)
axes[0].set_title('PEMS07', fontsize=16)
axes[0].grid(True)

# PEMS08 折线图
axes[1].plot(kernels, mae_pems08, marker='o', linestyle='-', color='tab:orange')
axes[1].set_xticks(kernels)
axes[1].set_xlabel('Number of Convolution Kernels (P)', fontsize=16)
axes[1].set_ylabel('MAE', fontsize=16)
axes[1].set_title('PEMS08', fontsize=16)
axes[1].grid(True)

# 调整子图间距
plt.tight_layout()

# 保存图像
plt.savefig('fig9.pdf', dpi=400, bbox_inches='tight')

# 显示图像
plt.show()
