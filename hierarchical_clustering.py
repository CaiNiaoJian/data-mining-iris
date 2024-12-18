import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from data_visualization import load_data

def plot_dendrogram(X, method='ward'):
    """绘制层次聚类树状图"""
    plt.figure(figsize=(10, 7))
    
    # 计算链接矩阵
    linkage_matrix = linkage(X, method=method)
    
    # 绘制树状图
    dendrogram(linkage_matrix)
    plt.title(f'层次聚类树状图 ({method}链接)')
    plt.xlabel('样本索引')
    plt.ylabel('距离')
    plt.savefig(f'dendrogram_{method}.png')
    plt.close()

def compare_linkage_methods(X, y):
    """比较不同链接方法的聚类效果"""
    methods = ['ward', 'complete', 'average', 'single']
    n_clusters = 3
    
    plt.figure(figsize=(15, 10))
    
    for i, method in enumerate(methods, 1):
        # 执行聚类
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        labels = clustering.fit_predict(X)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(X, labels)
        
        # 计算准确率
        correct = sum(labels == y)
        accuracy = correct / len(y) * 100
        
        # 绘制聚类结果
        plt.subplot(2, 2, i)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f'{method}链接\n准确率: {accuracy:.2f}%\n轮廓系数: {silhouette_avg:.3f}')
        plt.xlabel('Sepal Length')
        plt.ylabel('Petal Length')
    
    plt.tight_layout()
    plt.savefig('linkage_comparison.png')
    plt.close()

def plot_cluster_evolution(X):
    """绘制不同聚类数量的演化过程"""
    n_clusters_range = [2, 3, 4, 5]
    
    plt.figure(figsize=(15, 10))
    
    for i, n_clusters in enumerate(n_clusters_range, 1):
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(X)
        
        plt.subplot(2, 2, i)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f'聚类数量 = {n_clusters}')
        plt.xlabel('Sepal Length')
        plt.ylabel('Petal Length')
    
    plt.tight_layout()
    plt.savefig('cluster_evolution.png')
    plt.close()

if __name__ == '__main__':
    # 加载数据
    X_scaled, y, _ = load_data()
    
    # 绘制树状图
    plot_dendrogram(X_scaled)
    
    # 比较不同链接方法
    compare_linkage_methods(X_scaled, y)
    
    # 绘制聚类演化过程
    plot_cluster_evolution(X_scaled) 