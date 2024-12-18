import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

data_url="iris_train_shuzhi.csv"
df = pd.read_csv(data_url)
data = df.T[0:4].T
estimator = KMeans(n_clusters=3)#构造聚类器
result = estimator.fit_predict(data)
print(result)
