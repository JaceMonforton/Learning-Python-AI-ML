from sklearn.datasets import load_iris
from sklearn.decomposition import PCA 
import pylab as pl
from itertools import cycle


iris = load_iris()

numSamples, numFeatures = iris.data.shape
print(numSamples)
print(numFeatures) #4 features = 4 dimensions

print(list(iris.target_names))

X = iris.data

pca = PCA(n_components=2, whiten=True).fit(X)

X_pca = pca.transform(X)

# print(pca.components_) #Prints 4D Eigenvectors

print(pca.explained_variance_ratio_) #tells us how much variance in 4d data was preserved as its dimension is reduced.
print(sum(pca.explained_variance_ratio_)) #total sum of variance preserved.


colors = cycle('rgb')

target_ids = range(len(iris.target_names))
pl.figure()

for i, c, label in zip(target_ids, colors, iris.target_names): #iterate thru 3 species
    pl.scatter(X_pca[iris.target == i, 0], X_pca[iris.target == i, 1],
               c = c , label = label)

pl.legend()
pl.show()
