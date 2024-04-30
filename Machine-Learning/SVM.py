import numpy as np
from pylab import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import svm, datasets

def createClusteredData(N, k):
    np.random.seed(1234)
    pointsPerCluster = float (N)/k
    X=[]
    y=[]

    for i in range(k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)

        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            y.append(i)
        
    X = np.array(X)
    y = np.array(y)
    return X, y
    
(X, y) = createClusteredData(100, 5)

plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1] , c=y)
# plt.show()

scaling = MinMaxScaler(feature_range=(-1,1)).fit(X) #Makes data friendly to SVC
X = scaling.transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1] , c=y)
# plt.show()


C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X, y)

def PlotPredictions(clf):
    xx, yy = np.meshgrid(np.arange(-1, 1, 0.001), np.arange(-1, 1, 0.001))


    npx = xx.ravel()
    npy = yy.ravel()


    samplePoints = np.c_[npx, npy]

    z = clf.predict(samplePoints)

    plt.figure(figsize=(8,6))

    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha = 0.8)
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()

# PlotPredictions(svc)

print(svc.predict(scaling.transform([[200000, 40]])))
print(svc.predict(scaling.transform([[60000, 60]])))