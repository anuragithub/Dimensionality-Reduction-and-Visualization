#!/usr/bin/env python
# coding: utf-8

# In[20]:


from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import time


# In[4]:


mnist = fetch_openml('mnist_784')


# In[5]:


X = mnist.data / 255.0
y = mnist.target


# In[6]:


X.shape


# In[7]:


feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None


# In[8]:


pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]


# In[9]:


pca.explained_variance_ratio_


# In[10]:


rndperm = np.random.permutation(df.shape[0])


# In[11]:


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
).get_figure().savefig("pca_mnist.png")


# In[12]:


pca.noise_variance_


# In[13]:


pca.n_components_


# In[14]:


ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["pca-one"].astype(float), 
    ys=df.loc[rndperm,:]["pca-two"].astype(float), 
    zs=df.loc[rndperm,:]["pca-three"].astype(float), 
    c=df.loc[rndperm,:]["y"].astype(float), 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()


# In[18]:


ax.get_figure().savefig("pca_3d_mnist.png")


# In[26]:


N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values


# In[27]:


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[38]:


df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns_plot = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
)


# In[42]:


sns_plot.get_figure().savefig("tsne_mnist.png")


# In[21]:


reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(mnist.data)

fig, ax = plt.subplots(figsize=(12, 10))
color = mnist.target.astype(int)
plt.scatter(
    embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1
)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

plt.show()


# In[23]:


fig.savefig("umap_mnist.png")


# In[ ]:




