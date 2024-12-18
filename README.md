# 聚类算法实现与比较

本项目实现并比较了三种经典的聚类算法（K-means、DBSCAN和层次聚类），使用鸢尾花数据集进行测试和性能评估。

## 项目结构

```
.
├── README.md                    # 项目说明文档
├── data_visualization.py        # 数据加载和可视化
├── kmeans_improved.py          # K-means聚类实现
├── dbscan_clustering.py        # DBSCAN聚类实现
├── hierarchical_clustering.py   # 层次聚类实现
├── iris_train.csv              # 原始训练数据
├── iris_train_shuzhi.csv       # 处理后的训练数据
└── iris_test_y.csv             # 测试数据
```

## 环境要求

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn
- SciPy

## 安装依赖

```bash
pip install numpy pandas matplotlib scikit-learn seaborn scipy
```

## 模块说明

### 1. 数据可视化模块 (data_visualization.py)
- 数据加载和预处理功能
- 特征分布可视化
- 特征散点矩阵可视化
- 输出文件：
  - `feature_distribution.png`：各特征的分布直方图
  - `scatter_matrix.png`：特征散点矩阵图

### 2. K-means聚类模块 (kmeans_improved.py)
- 从零实现K-means算法
- 评估指标：
  - SSE（簇内误差平方和）
  - 轮廓系数
  - 聚类准确率
- 可视化功能：
  - 肘部曲线
  - 轮廓系数曲线
  - 聚类结果散点图
- 输出文件：
  - `kmeans_evaluation.png`：评估指标可视化
  - `kmeans_clusters.png`：聚类结果可视化

### 3. DBSCAN聚类模块 (dbscan_clustering.py)
- 基于密度的聚类实现
- 参数优化：
  - eps参数优化
  - min_samples参数优化
- 聚类分析：
  - 噪声点识别
  - 类别数量统计
  - 聚类准确率计算
- 输出文件：
  - `dbscan_parameter_optimization.png`：参数优化过程
  - `dbscan_clusters.png`：聚类结果可视化

### 4. 层次聚类模块 (hierarchical_clustering.py)
- 实现多种链接策略：
  - Ward链接
  - Complete链接
  - Average链接
  - Single链接
- 可视化功能：
  - 层次聚类树状图
  - 不同链接方法比较
  - 聚类数量演化过程
- 输出文件：
  - `dendrogram_ward.png`：层次聚类树状图
  - `linkage_comparison.png`：不同链接方法比较
  - `cluster_evolution.png`：聚类演化过程

## 使用方法

1. 数据可视化：
```bash
python data_visualization.py
```
- 输出：特征分布图和散点矩阵图

2. K-means聚类：
```bash
python kmeans_improved.py
```
- 输出：聚类评估指标、聚类结果和准确率

3. DBSCAN聚类：
```bash
python dbscan_clustering.py
```
- 输出：最优参数、聚类结果分析和准确率

4. 层次聚类：
```bash
python hierarchical_clustering.py
```
- 输出：不同链接方法的比较结果和聚类演化过程

## 评估指标

1. 聚类准确率
   - K-means：直接比较标签
   - DBSCAN：排除噪声点后比较
   - 层次聚类：不同链接方法的准确率比较

2. 聚类质量评估
   - SSE（簇内误差平方和）
   - 轮廓系数
   - 噪声点比例（DBSCAN）

3. 可视化评估
   - 聚类结果散点图
   - 参数优化曲线
   - 层次聚类树状图

## 数据集

使用经典的鸢尾花（Iris）数据集，包含：
- 150个样本
- 3个品种：Setosa、Versicolor、Virginica
- 4个特征：
  - 萼片长度（Sepal Length）
  - 萼片宽度（Sepal Width）
  - 花瓣长度（Petal Length）
  - 花瓣宽度（Petal Width）

## 实验结果分析

1. 数据特征
   - 通过特征分布图可以观察各类别的分布情况
   - 散点矩阵展示了特征间的相关性

2. 算法比较
   - K-means：适用于球形簇，计算效率高
   - DBSCAN：可以发现任意形状的簇，能识别噪声点
   - 层次聚类：提供了数据的层次结构信息，不同链接方法适用于不同数据分布

3. 性能评估
   - 准确率比较
   - 计算效率
   - 参数敏感度
   - 结果可解释性