import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

data_url="iris_train_shuzhi.csv"
df = pd.read_csv(data_url)
data = df.T[0:4].T
AgglomerativeClustering = AgglomerativeClustering(linkage= 'average' ,n_clusters=3)#构造聚类器
result = AgglomerativeClustering.fit_predict(data)
print(result)
