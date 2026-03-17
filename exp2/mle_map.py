import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

# 设置中文字体以修复乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def run_parameter_estimation():
    # (c) 实验步骤
    # 1. 构造 0-1 样本数据
    data = torch.tensor([1., 1., 0., 1., 0.])
    
    # 2. 使用 PyTorch 计算 MLE (最大似然估计)
    p_mle = torch.mean(data)
    
    # 3. 引入 Beta 先验计算 MAP (最大后验概率估计)
    # 设定 Beta 分布参数 alpha=2, beta=2 (均匀分布的先验)
    alpha = 2
    beta = 2
    p_map = (torch.sum(data) + alpha - 1) / (len(data) + alpha + beta - 2)
    
    print(f"MLE = {p_mle.item():.4f}")
    print(f"MAP = {p_map.item():.4f}")

    # (d) 可视化分析
    data_np = data.numpy()
    N = len(data_np)
    sum_x = np.sum(data_np)
    
    p_range = np.linspace(0, 1, 100)
    
    # 似然函数: p^sum_x * (1-p)^(N-sum_x)
    likelihood = p_range**sum_x * (1-p_range)**(N-sum_x)
    
    # 先验分布: Beta(2, 2)
    prior = beta_dist.pdf(p_range, alpha, beta)
    
    # 后验分布: 比例于 似然 * 先验
    # 基于共轭先验，后验分布为 Beta(sum_x + alpha, N - sum_x + beta)
    posterior = beta_dist.pdf(p_range, sum_x + alpha, N - sum_x + beta)

    plt.figure(figsize=(10, 6))
    # 为方便对比，对曲线进行归一化处理
    plt.plot(p_range, likelihood / np.max(likelihood) if np.max(likelihood) > 0 else likelihood, label="Likelihood (Normalized)", alpha=0.7)
    plt.plot(p_range, prior / np.max(prior), label="Prior (Normalized)", alpha=0.7)
    plt.plot(p_range, posterior / np.max(posterior), label="Posterior (Normalized)", alpha=0.7)
    
    plt.axvline(p_mle.item(), color='r', linestyle='--', label=f"MLE ({p_mle.item():.2f})")
    plt.axvline(p_map.item(), color='g', linestyle='--', label=f"MAP ({p_map.item():.2f})")
    
    plt.legend()
    plt.title("MLE vs MAP Estimation (Bernoulli)")
    plt.xlabel("Parameter p")
    plt.ylabel("Density / Likelihood")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("mle_map_comparison.png")
    print("图像已保存为 mle_map_comparison.png")
    plt.show()

if __name__ == "__main__":
    run_parameter_estimation()


if __name__ == "__main__":
    run_parameter_estimation()
