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
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    
def plot_dendrogram(model, **kwargs):#繪圖
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
PATH = os.path.join(os.getcwd(),'mini_newsgroups')
label = []
texts = []
count = 0
for j in os.listdir(PATH):
    for k in os.listdir(os.path.join(PATH , j)):
        f = open(os.path.join(PATH , str(j) , str(k)), encoding='latin-1')
        t = f.read()
        i = t.find('writes:')  # skip header
        if 0 < i:
            t = t[i:]
        t = t.replace('writes:', '')
        texts.append(t)
        label.append(count)
        f.close()
    count += 1
print(texts[0])
print(len(texts))
label=np.array(label)
print(label)
#%%去小寫
lowerlist = []
for i in texts:
    lowerlist.append(i.lower())
#%%去數字
removeNum = []
for i in lowerlist:
    removeNum.append(re.sub(r'\d+', '', i))
#%%去特殊符號
removeP = []
punc = '''!()-[]{}|;:'"\,<>./?@#$%^&*_~'''
for i in removeNum:
    new = ""
    for oneS in i:#ele是一個一個字
        if oneS not in punc:#不是punc裡面的字 加入  
            new = new + oneS
    removeP.append(new)
#%%
from gensim.parsing.preprocessing import remove_stopwords
removeStopWords = []
for i in removeP:
    removeStopWords.append(remove_stopwords(i))
vectorizer = TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english',decode_error='ignore')
X = vectorizer.fit_transform(removeStopWords)
print(X.shape)
#前處理結束

#%%
t0 = time()
km = KMeans(n_clusters=20, random_state=0).fit(X)
print('kmeans :', str(round(time() - t0,2))+"s","Purity :",purity_score(label,km.labels_))
#%%Hierarchical Clustering

agg = AgglomerativeClustering(n_clusters=20).fit(X.toarray())
print('Hierarchical Clustering :', str(round(time() - t0,2))+"s","Purity:",purity_score(label,agg.labels_))
all_hc = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
all_hc.fit(X.toarray())



plt.figure(figsize=(12,6))
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(all_hc, truncate_mode=None)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.savefig("Hierarchical Clustering Dendrogram.png")


plt.figure(figsize=(12,6))
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(all_hc,  truncate_mode='lastp', p=20)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.savefig("W3_1.png")
print("D")
#%%DBSCAN

res = []

for eps in np.arange(0.05,1,0.005):
    for min_samples in range(2,10):
        start_time = time()
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        dbscan.fit(X)
        end_time = time()
        n_clusters = len([i for i in set(dbscan.labels_) if i != -1])#分群數
        outliners = np.sum(np.where(dbscan.labels_ == -1, 1,0))#離群數
        stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)#弄成df列格式儲存進DF 不同eps minsample看purity是否有較好
        a= {'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,'outliners':outliners,'stats':stats, 'purity':purity_score(label, dbscan.labels_), 'time': (end_time - start_time)}
        res.append({'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,'outliners':outliners,'stats':stats, 'purity':purity_score(label, dbscan.labels_), 'time': (end_time - start_time)})     
        
df = pd.DataFrame(res)
print(df.loc[df.n_clusters == 20, :])
