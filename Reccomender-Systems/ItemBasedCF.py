import pandas as pd


r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('C:\\MLCourse\\ml-100k\\u.data', sep='\t', names=r_cols, usecols=range(3))
m_cols = ['movie_id', 'title']
movies = pd.read_csv('C:\\MLCourse\\ml-100k\\u.item', sep='|', names=m_cols, usecols=range(2), encoding='latin1')

ratings = pd.merge(movies, ratings)
# print(ratings.head())


userRatings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')
# print(userRatings.head())

corrMatrix = userRatings.corr(method='pearson', min_periods=100) #correlation between any 2 movies. given at least min_periods people rating 
# print(corrMatrix.head())

myRatings = userRatings.loc[0].dropna()
# print(myRatings) 

SimulationUser = pd.Series(0) #initialize @ empty series.
for i in range(0, len(myRatings.index)):
    print('Adding Simulation for ' + myRatings.index[i] + '...')
    #Retrieve Similar movies to the ones i rated
    sims = corrMatrix[myRatings.index[i]].dropna() #Drops NaN Values.
    #Scale the similarity by how well i rated it 
    sims = sims.map(lambda x: x * myRatings.iloc[i]) #iloc for series 
    #Add score to the list of similarity candidates
    SimulationUser = SimulationUser._append(sims)




print("\nSorting...")
SimulationUser = SimulationUser.groupby(SimulationUser.index).sum()
SimulationUser.sort_values(inplace=True, ascending=False)
print(SimulationUser.head(10))

print("\nFiltering....")
filteredSim = SimulationUser.drop(myRatings.index)
print(filteredSim.head(10))