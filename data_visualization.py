import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_data():
    """加载和预处理数据"""
    try:
        # 加载数据，跳过第一行，使用第二行作为列名
        data = pd.read_csv('iris_train_shuzhi.csv', skiprows=[0])
        print("原始数据形状:", data.shape)
        print("原始数据前几行:\n", data.head())
        
        # 删除第一列（索引列）
        data = data.iloc[:, 1:]
        
        # 检查是否有缺失值
        if data.isnull().any().any():
            print("警告：数据中存在缺失值，将进行处理")
            data = data.dropna()
        
        # 分离特征和标签
        X = data.iloc[:, :-1].astype(float)  # 特征
        y = data.iloc[:, -1].astype(int)     # 标签
        
        print("\n数据统计信息:")
        print(X.describe())
        
        return X.values, y.values, X.values
    except Exception as e:
        print(f"数据加载错误：{str(e)}")
        raise

def plot_features_distribution(X, y):
    """绘制特征分布图"""
    try:
        features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(features):
            plt.subplot(2, 2, i+1)
            for class_label in np.unique(y):
                mask = y == class_label
                plt.hist(X[mask, i], 
                        bins=20, 
                        alpha=0.5,
                        label=f'Class {class_label}',
                        density=True)
            plt.title(f'{feature} Distribution')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('feature_distribution.png')
        print("特征分布图数据范围：", np.ptp(X, axis=0))
        plt.close()
    except Exception as e:
        print(f"特征分布图绘制错误：{str(e)}")
        plt.close()

def plot_scatter_matrix(X, y):
    """绘制特征散点矩阵"""
    try:
        plt.figure(figsize=(10, 8))
        
        # 使用不同的颜色和标记绘制散点图
        markers = ['o', 's', '^']  # 不同的标记样式
        for i, class_label in enumerate(np.unique(y)):
            mask = y == class_label
            plt.scatter(X[mask, 0], 
                       X[mask, 2],
                       alpha=0.6,
                       marker=markers[i],
                       s=50,
                       label=f'Class {class_label}')
        
        plt.xlabel('Sepal Length')
        plt.ylabel('Petal Length')
        plt.title('Sepal Length vs Petal Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('scatter_matrix.png')
        print("散点图数据范围：", 
              "\nSepal Length:", np.ptp(X[:, 0]),
              "\nPetal Length:", np.ptp(X[:, 2]))
        plt.close()
    except Exception as e:
        print(f"散点矩阵图绘制错误：{str(e)}")
        plt.close()

def print_data_summary(X, y):
    """打印数据摘要信息"""
    try:
        print("\n数据集摘要信息：")
        print(f"样本数量：{len(X)}")
        print(f"特征数量：{X.shape[1]}")
        print(f"类别数量：{len(np.unique(y))}")
        print("\n各类别样本数量：")
        unique_labels, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"类别 {label}: {count}")
        
        print("\n特征值范围：")
        for i, feature in enumerate(['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']):
            print(f"{feature}: [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")
    except Exception as e:
        print(f"数据摘要信息生成错误：{str(e)}")

if __name__ == '__main__':
    try:
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 加载数据
        X_scaled, y, X_original = load_data()
        
        # 打印数据摘要
        print_data_summary(X_original, y)
        
        # 绘制特征分布图
        plot_features_distribution(X_original, y)
        print("特征分布图已保存为 'feature_distribution.png'")
        
        # 绘制散点矩阵
        plot_scatter_matrix(X_original, y)
        print("散点矩阵图已保存为 'scatter_matrix.png'")
        
    except Exception as e:
        print(f"程序执行出错：{str(e)}") 