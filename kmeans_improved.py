import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from data_visualization import load_data

class KMeans:
    def __init__(self, n_clusters=3, max_iters=300):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        # 随机初始化聚类中心
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # 计算每个样本到各个聚类中心的距离
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # 更新聚类中心
            for k in range(self.n_clusters):
                if sum(self.labels == k) > 0:
                    self.centroids[k] = X[self.labels == k].mean(axis=0)
            
            # 检查是否收敛
            if np.all(old_centroids == self.centroids):
                break
                
        return self
    
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

def calculate_sse(X, kmeans):
    """计算SSE（簇内误差平方和）"""
    sse = 0
    for k in range(kmeans.n_clusters):
        cluster_points = X[kmeans.labels == k]
        centroid = kmeans.centroids[k]
        sse += np.sum((cluster_points - centroid) ** 2)
    return sse

def plot_elbow_curve(X):
    """绘制肘部曲线"""
    sse_values = []
    silhouette_values = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        sse = calculate_sse(X, kmeans)
        silhouette = silhouette_score(X, kmeans.labels)
        sse_values.append(sse)
        silhouette_values.append(silhouette)
    
    # 绘制SSE曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, sse_values, 'bo-')
    plt.xlabel('聚类数量 (k)')
    plt.ylabel('SSE')
    plt.title('肘部曲线')
    
    # 绘制轮廓系数曲线
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_values, 'ro-')
    plt.xlabel('聚类数量 (k)')
    plt.ylabel('轮廓系数')
    plt.title('轮廓系数曲线')
    
    plt.tight_layout()
    plt.savefig('kmeans_evaluation.png')
    plt.close()

def plot_clusters(X, kmeans, title='K-means聚类结果'):
    """绘制聚类结果"""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='聚类中心')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')
    plt.legend()
    plt.savefig('kmeans_clusters.png')
    plt.close()

if __name__ == '__main__':
    # 加载数据
    X_scaled, y, _ = load_data()
    
    # 绘制肘部曲线和轮廓系数
    plot_elbow_curve(X_scaled)
    
    # 使用最优k值进行聚类
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_scaled)
    
    # 绘制聚类结果
    plot_clusters(X_scaled, kmeans)
    
    # 计算聚类准确率
    correct = sum(kmeans.labels == y)
    accuracy = correct / len(y) * 100
    print(f"聚类准确率: {accuracy:.2f}%") 