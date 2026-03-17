import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体以修复乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def run_gradient_descent():
    # 损失曲线分析
    loss_history = []
    # 初始化参数 x = 5.0，并开启梯度追踪
    x = torch.tensor([5.0], requires_grad=True)
    lr = 0.1 # 学习率

    for i in range(20):
        # 定义目标函数 f(x) = (x+1)^2
        y = x**2 + 2*x + 1
        loss_history.append(y.item())
        
        # 反向传播计算梯度
        y.backward()
        # 参数更新需要关闭梯度记录
        with torch.no_grad():
            x -= lr * x.grad
        # 清零梯度以备下次迭代使用
        x.grad.zero_()

    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, marker='o', linestyle='-', color='b')
    plt.title("Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss Val (y)")
    plt.grid(True)

    # 梯度下降路径可视化
    x_vals = np.linspace(-6, 6, 100)
    y_vals = x_vals**2 + 2*x_vals + 1

    x_path = []
    x_param = torch.tensor([5.0], requires_grad=True)
    lr_path = 0.3 # 较大的学习率以便观察路径

    for i in range(10):
        y_val = x_param**2 + 2*x_param + 1
        x_path.append(x_param.item())

        y_val.backward()
        with torch.no_grad():
            x_param -= lr_path * x_param.grad
        x_param.grad.zero_()

    # 绘制优化路径
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, y_vals, label="f(x) = x^2 + 2x + 1")
    path_y = [xv**2 + 2*xv + 1 for xv in x_path]
    plt.scatter(x_path, path_y, color='r', zorder=5)
    plt.plot(x_path, path_y, color='r', linestyle='--', alpha=0.5, label="GD Path")
    plt.title("Gradient Descent Path")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("gradient_descent_analysis.png")
    print("图像已保存为 gradient_descent_analysis.png")
    plt.show()

if __name__ == "__main__":
    run_gradient_descent()


if __name__ == "__main__":
    run_gradient_descent()
