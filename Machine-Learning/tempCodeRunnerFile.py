import numpy as np
from pylab import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def createClusteredData(N, k):
    np.random.seed(1234)
    pointsPerCluster = float (N)/k
    X=[]
    y=[]

    for i in range(k):
        incomeCentroid = np.random.uniform(200000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)

        for j in range(int(pointsPerCluster)):
            x.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            y.append(i)
        
        X = np.array(x)
        y = np.array(y)

        return x,y
    
(X,y) = createClusteredData(100, 5)



scaling = MinMaxScaler(feature_range=(-1,1)).fit(x)
X = scaling.transform(x)

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1] , c=y.astype(np.float))
plt.show()
