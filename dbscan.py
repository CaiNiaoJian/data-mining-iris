import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# 生成样本数据
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,random_state=0)
X = StandardScaler().fit_transform(X)
# 训练DBSCAN模型
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
print(db.labels_)

data = pd.read_csv('d:/studyRoom/datamining/data/data/irls/iris_train_shuzhi.csv')
plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Iris Dataset')
plt.show()

colors = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(data[data['class'] == i].iloc[:, 0], data[data['class'] == i].iloc[:, 1], color=colors[i], label=f'Class {i}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Iris Dataset')
plt.legend()
plt.show()
