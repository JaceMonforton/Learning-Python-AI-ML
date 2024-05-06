import pandas as pd 
import numpy as np
from scipy import spatial
import operator

r_rols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('C:\\MLCourse\\ml-100k\\u.data', sep='\t', names = r_rols, usecols=range(3))

# print(ratings.head())
movieProperties = ratings.groupby('movie_id').agg(({'rating' : [np.size, 'mean']})) #groups the movies by the #of people & mean rating. 
# print(movieProperties.head())

movieNumRating = pd.DataFrame(movieProperties['rating']['size'])

movieNormalizedRatings = movieNumRating.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))) #Normalizes the size, (0,1) - (0 = no one watched) , (1 = everyone watched)

print(movieNormalizedRatings.head())

movieDict = {}

with open('C:\\MLCourse\\ml-100k\\u.item') as f:
    temp = ''
    for line in f:
        fields = line.rstrip('\n').split('|')
        movieID = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        genres = map(int, genres)
        movieDict[movieID] = (name, np.array(list(genres)), movieNormalizedRatings.loc[movieID].get('size'), movieProperties.loc[movieID].rating.get('mean'))


# print(movieDict[1])

def ComputeDistance(a, b): #far dist = not similar, want closest distance for similarity.
    genresA = a[1]
    genresB = b[1]
    genreDist = spatial.distance.cosine(genresA, genresB)

    popA = a[2]
    popB = b[2]
    PopDist = abs(popA - popB)

    return genreDist + PopDist

distTest = ComputeDistance(movieDict[2], movieDict[4]) 
print(distTest)


def getNeighbors(movieID, K):
    distances = []
    for movie in movieDict:
        if (movie != movieID):
            dist = ComputeDistance(movieDict[movieID], movieDict[movie])
            distances.append((movie, dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []

    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors

K = 10
avgRating = 0
neighbors = getNeighbors(1, K) #using movieID = 1 as example.

for neighbor in neighbors:
    avgRating += movieDict[neighbor][3]
    print(movieDict[neighbor][0] + " " + str(movieDict[neighbor][3]))

avgRating /= float(K)

