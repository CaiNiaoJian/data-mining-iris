import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
np.random.seed(0)
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
data_sets = [(noisy_circles, {"n_clusters":2}),(noisy_moons, {"n_clusters":2}),(blobs, {"n_clusters":3})]
colors = ["#377eb8", "#ff7f00", "#4daf4a"]
linkages = ["single","ward", "average", "complete"]
plt.figure(figsize=(20, 15))
for i_dataset, (dataset, algo_params) in enumerate(data_sets):
    params = algo_params

    X, y = dataset
    X = StandardScaler().fit_transform(X)
    for i_linkage, linkage_stragegy in enumerate(linkages):
        ac=cluster.AgglomerativeClustering(n_clusters=params["n_clusters"],linkage=linkage_stragegy)
        ac.fit(X)
        y_pred = ac.labels_.astype(int)
        print(y_pred)
        y_pred_colors = []
        for i in y_pred:
            y_pred_colors.append(colors[i])
        plt.subplot(3,4,4*i_dataset+i_linkage+1)
        plt.scatter(X[:,0],X[:,1],color=y_pred_colors)
        plt.title(linkage_stragegy)
plt.show()