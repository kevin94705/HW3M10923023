from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.datasets import load_files
from sklearn import metrics
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from time import time
import re
import numpy as np
import os
import pandas as pd
import  matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
#%%
def purity_score(y_true, y_pred):
    #計算純度
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
def plot_dendrogram(model, **kwargs):
    '''
    繪圖
    '''
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)
#%%

df = pd.read_csv(filepath_or_buffer="Shill Bidding Dataset.csv",header=0)
label=df['Class']#分類
df = df.drop(['Record_ID','Bidder_ID','Auction_ID','Class'],axis=1)#丟棄不需要的欄位

X=df.to_numpy()
X=np.array(X)

print(len(label))
print(X.shape)

#%%
t0 = time()
km = KMeans(n_clusters=2, random_state=0).fit(X)
print(km.labels_)
print('kmeans :', str(round(time() - t0,2))+"s","Purity :",purity_score(label,km.labels_))
#%%Hierarchical Clustering

agg = AgglomerativeClustering(n_clusters=2).fit(X)#有指定分群樹
print('Hierarchical Clustering :', str(round(time() - t0,2))+"s","Purity:",purity_score(label,agg.labels_))
all_hc = AgglomerativeClustering(distance_threshold=0, n_clusters=None)#無指定分群數
all_hc.fit(X)
#%%
#劃出Hierarchical
plt.figure(figsize=(9,6))
plt.title('Hierarchical clustering ')
plot_dendrogram(all_hc, truncate_mode=None)
plt.xlabel("The number of data in the node (if there are no parentheses, or a point index).")
plt.savefig("Hierarchical Clustering Dendrogram.png")
#劃出Hierarchical 分 2群的
plt.figure(figsize=(9,6))
plt.title('Hierarchical clustering ')
plot_dendrogram(all_hc,  truncate_mode='lastp', p=2)
plt.xlabel("The number of data in the node (if there are no parentheses, or a point index).")
plt.savefig("W3_1_2.png")
print("D")
#%%DBSCAN

res = []
#eps測試
for eps in np.arange(1,2,0.05):
    for min_samples in range(5,15):#一個點附近的樣品數量
        start_time = time()
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        dbscan.fit(X)
        end_time = time()
        n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
        outliners = np.sum(np.where(dbscan.labels_ == -1, 1,0))
        stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
        res.append({'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,'outliners':outliners,'stats':stats, 'purity':purity_score(label, dbscan.labels_), 'time': (end_time - start_time)})     
df = pd.DataFrame(res)
print(df)
print(df.loc[df.n_clusters == 5, :])#印出分5群最好的參數值