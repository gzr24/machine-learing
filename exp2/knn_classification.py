import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 设置中文字体以修复乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def run_knn_classification():
    # 1. 构造二维数据 (增加更多点以使边界变化更明显)
    X = np.array([
        [1, 2], [2, 3], [3, 3], [1.5, 4], [3, 1.5], [4, 3], [4.5, 4.5], # 类别 0
        [6, 5], [7, 7], [8, 6], [9, 8], [6, 9], [8, 4], [5.5, 6]       # 类别 1
    ])
    y = np.array([0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1])

    # 2. 设置不同 K 值
    # 3. 进行分类预测
    # 4. 绘制分类边界
    
    # 创建网格以绘制决策边界
    xx, yy = np.meshgrid(np.linspace(0, 10, 200),
                         np.linspace(0, 10, 200))
    
    ks = [1, 3, 5]
    plt.figure(figsize=(15, 5))
    
    for i, k in enumerate(ks):
        # 初始化 KNN 分类器
        model = KNeighborsClassifier(n_neighbors=k)
        # 拟合模型
        model.fit(X, y)

        # 预测网格中每个点的类别
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.subplot(1, 3, i + 1)
        # 绘制等高线图（分类区域）
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        # 绘制原始数据点
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
        plt.title(f"KNN Classify (K = {k})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("knn_comparison.png")
    print("图像已保存为 knn_comparison.png")
    plt.show()

if __name__ == "__main__":
    run_knn_classification()


if __name__ == "__main__":
    run_knn_classification()
