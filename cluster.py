import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
df.columns=['sepal length','sepal width','petal length','petal width','class']
samples=df.iloc[:,0:4]
model=KMeans(n_clusters=3)
model.fit(samples)
labels=model.predict(samples)
centroids=model.cluster_centers_
print(centroids)
plt.scatter(samples.iloc[:,0],samples.iloc[:,2],c=labels)
plt.scatter(centroids[:,0],centroids[:,2],marker='D',s=50)
plt.show()

print(model.inertia_)
df_cross=pd.DataFrame({'labels':labels,'species':labels})
ct=pd.crosstab(df_cross['labels'],df_cross['species'])
print(ct)

clusters=np.arange(1,20,1)
inertia=[]
for i in range(1,20):
    model=KMeans(n_clusters=i)
    model.fit(samples)
    inertia.append(model.inertia_)
plt.plot(clusters,inertia,c='r')
