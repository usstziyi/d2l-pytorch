# 随机初始化张量并画出三维立方图
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_tensor_3d():
    """随机初始化张量并画出三维立方图"""
    
    # 1. 随机初始化张量
    print("=== 随机初始化张量 ===")
    X_random = torch.randn(2, 5)  # 随机初始化2x5的张量
    print("随机初始化的张量X_random:")
    print(X_random)
    print(f"张量形状: {X_random.shape}")
    
    # 2. 转换为独热编码
    print("\n=== 独热编码转换 ===")
    # 方法1：使用argmax转换为独热编码
    X_one_hot_argmax = F.one_hot(X_random.argmax(dim=1), 28)
    print("使用argmax的独热编码:")
    print(X_one_hot_argmax)
    print(f"形状: {X_one_hot_argmax.shape}")
    
    # 方法2：直接使用原始张量（需要先转换为整数索引）
    X_indices = torch.randint(0, 28, (2, 5))  # 随机生成0-27的整数索引
    X_one_hot_direct = F.one_hot(X_indices, 28)
    print("\n直接随机索引的独热编码:")
    print(X_one_hot_direct)
    print(f"形状: {X_one_hot_direct.shape}")
    
    # 3. 创建三维立方图
    fig = plt.figure(figsize=(15, 10))
    
    # 第一个子图：原始随机张量
    ax1 = fig.add_subplot(231, projection='3d')
    x_coords, y_coords = np.meshgrid(range(X_random.shape[1]), range(X_random.shape[0]))
    scatter1 = ax1.scatter(x_coords, y_coords, X_random.numpy(), 
                          c=X_random.numpy(), cmap='viridis', s=100)
    ax1.set_xlabel('列索引')
    ax1.set_ylabel('行索引')
    ax1.set_zlabel('张量值')
    ax1.set_title('随机初始化张量的三维图')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    
    # 第二个子图：独热编码张量（第一个样本）
    ax2 = fig.add_subplot(232, projection='3d')
    sample_one_hot = X_one_hot_argmax[0].float()
    x_coords_2 = np.arange(sample_one_hot.shape[0])
    y_coords_2 = np.zeros_like(x_coords_2)
    scatter2 = ax2.scatter(x_coords_2, y_coords_2, sample_one_hot.numpy(), 
                          c=sample_one_hot.numpy(), cmap='plasma', s=100)
    ax2.set_xlabel('词表索引')
    ax2.set_ylabel('样本索引')
    ax2.set_zlabel('独热编码值')
    ax2.set_title('独热编码张量（第一个样本）')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5)
    
    # 第三个子图：独热编码张量（第二个样本）
    ax3 = fig.add_subplot(233, projection='3d')
    sample_one_hot_2 = X_one_hot_argmax[1].float()
    scatter3 = ax3.scatter(x_coords_2, y_coords_2 + 1, sample_one_hot_2.numpy(), 
                          c=sample_one_hot_2.numpy(), cmap='plasma', s=100)
    ax3.set_xlabel('词表索引')
    ax3.set_ylabel('样本索引')
    ax3.set_zlabel('独热编码值')
    ax3.set_title('独热编码张量（第二个样本）')
    plt.colorbar(scatter3, ax=ax3, shrink=0.5)
    
    # 第四个子图：立方体可视化（原始张量）
    ax4 = fig.add_subplot(234, projection='3d')
    # 创建立方体的顶点
    for i in range(X_random.shape[0]):
        for j in range(X_random.shape[1]):
            x, y, z = j, i, X_random[i, j].item()
            # 绘制立方体的8个顶点
            for dx in [-0.2, 0.2]:
                for dy in [-0.2, 0.2]:
                    for dz in [-0.2, 0.2]:
                        ax4.scatter(x + dx, y + dy, z + dz, 
                                  c=X_random[i, j].item(), cmap='viridis', s=50)
    ax4.set_xlabel('列索引')
    ax4.set_ylabel('行索引')
    ax4.set_zlabel('张量值')
    ax4.set_title('张量立方体可视化')
    
    # 第五个子图：热力图
    ax5 = fig.add_subplot(235)
    im = ax5.imshow(X_random.numpy(), cmap='viridis', aspect='auto')
    ax5.set_xlabel('列索引')
    ax5.set_ylabel('行索引')
    ax5.set_title('张量热力图')
    plt.colorbar(im, ax=ax5)
    
    # 第六个子图：独热编码热力图
    ax6 = fig.add_subplot(236)
    im2 = ax6.imshow(X_one_hot_argmax.numpy(), cmap='plasma', aspect='auto')
    ax6.set_xlabel('词表索引')
    ax6.set_ylabel('样本索引')
    ax6.set_title('独热编码热力图')
    plt.colorbar(im2, ax=ax6)
    
    plt.tight_layout()
    plt.show()
    
    # 4. 打印详细信息
    print("\n=== 统计信息 ===")
    print(f"原始随机张量统计信息:")
    print(f"  最小值: {X_random.min():.4f}")
    print(f"  最大值: {X_random.max():.4f}")
    print(f"  均值: {X_random.mean():.4f}")
    print(f"  标准差: {X_random.std():.4f}")
    
    print(f"\n独热编码张量信息:")
    print(f"  形状: {X_one_hot_argmax.shape}")
    print(f"  数据类型: {X_one_hot_argmax.dtype}")
    print(f"  非零元素数量: {X_one_hot_argmax.sum()}")
    print(f"  每个样本的非零元素: {X_one_hot_argmax.sum(dim=1)}")
    
    return X_random, X_one_hot_argmax

def compare_with_original():
    """与原始代码对比"""
    print("\n=== 与原始代码对比 ===")
    
    # 原始代码
    X_original = torch.arange(10).reshape((2, 5))
    print("原始张量:")
    print(X_original)
    print(f"形状: {X_original.shape}")
    
    X_original_one_hot = F.one_hot(X_original.T, 28)
    print(f"\n原始独热编码形状: {X_original_one_hot.shape}")
    
    # 随机初始化
    X_random = torch.randn(2, 5)
    print(f"\n随机张量:")
    print(X_random)
    print(f"形状: {X_random.shape}")
    
    X_random_one_hot = F.one_hot(X_random.argmax(dim=1), 28)
    print(f"\n随机独热编码形状: {X_random_one_hot.shape}")
    
    return X_original, X_original_one_hot, X_random, X_random_one_hot

if __name__ == "__main__":
    # 运行可视化
    X_random, X_one_hot = visualize_tensor_3d()
    
    # 与原始代码对比
    X_orig, X_orig_one_hot, X_rand, X_rand_one_hot = compare_with_original()
