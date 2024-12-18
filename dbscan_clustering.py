import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from data_visualization import load_data

def find_optimal_eps(X, min_samples_range=[3, 5, 7]):
    """寻找最优eps和min_samples参数"""
    eps_range = np.arange(0.1, 2.1, 0.1)
    best_score = -1
    best_params = None
    scores = []
    
    for min_samples in min_samples_range:
        current_scores = []
        for eps in eps_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # 计算轮廓系数（排除噪声点）
            if len(np.unique(labels)) > 1 and -1 not in labels:
                score = silhouette_score(X, labels)
                current_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)
            else:
                current_scores.append(-1)
        
        scores.append(current_scores)
    
    # 绘制参数搜索结果
    plt.figure(figsize=(12, 6))
    for i, min_samples in enumerate(min_samples_range):
        plt.plot(eps_range, scores[i], label=f'min_samples={min_samples}')
    
    plt.xlabel('eps')
    plt.ylabel('轮廓系数')
    plt.title('DBSCAN参数优化')
    plt.legend()
    plt.grid(True)
    plt.savefig('dbscan_parameter_optimization.png')
    plt.close()
    
    return best_params

def plot_dbscan_clusters(X, labels, title='DBSCAN聚类结果'):
    """绘制DBSCAN聚类结果"""
    plt.figure(figsize=(10, 8))
    
    # 绘制噪声点
    noise_points = X[labels == -1]
    plt.scatter(noise_points[:, 0], noise_points[:, 1], 
               c='black', marker='x', label='噪声点', s=100)
    
    # 绘制聚类结果
    clustered_points = X[labels != -1]
    clustered_labels = labels[labels != -1]
    scatter = plt.scatter(clustered_points[:, 0], clustered_points[:, 1], 
                         c=clustered_labels, cmap='viridis')
    
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')
    plt.legend()
    plt.savefig('dbscan_clusters.png')
    plt.close()

def analyze_clusters(labels):
    """分析聚类结果"""
    n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"聚类数量: {n_clusters}")
    print(f"噪声点数量: {n_noise}")
    print(f"各类别样本数量:")
    for i in range(max(labels) + 1):
        print(f"类别 {i}: {list(labels).count(i)}")

if __name__ == '__main__':
    # 加载数据
    X_scaled, y, _ = load_data()
    
    # 寻找最优参数
    best_eps, best_min_samples = find_optimal_eps(X_scaled)
    print(f"最优参数: eps={best_eps}, min_samples={best_min_samples}")
    
    # 使用最优参数进行聚类
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    # 绘制聚类结果
    plot_dbscan_clusters(X_scaled, labels)
    
    # 分析聚类结果
    analyze_clusters(labels)
    
    # 计算聚类准确率（排除噪声点）
    mask = labels != -1
    correct = sum(labels[mask] == y[mask])
    accuracy = correct / sum(mask) * 100
    print(f"聚类准确率（排除噪声点）: {accuracy:.2f}%") 