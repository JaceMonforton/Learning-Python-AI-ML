import pandas as pd

# Load the ratings data
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('C:\\MLCourse\\ml-100k\\u.data', sep='\t', names=r_cols, usecols=range(3))

# Load the movies data
m_cols = ['movie_id', 'title']
movies = pd.read_csv('C:\\MLCourse\\ml-100k\\u.item', sep='|', names=m_cols, usecols=range(2), encoding='latin1')

# Merge movies and ratings data
ratings = pd.merge(movies, ratings)

# Pivot table to get user ratings
userRatings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')

# Remove outliers using IQR method
def remove_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[((data >= lower_bound) & (data <= upper_bound))]

userRatings_no_outliers = userRatings.apply(remove_outliers_iqr, axis=0)

# Build correlation matrix
corrMatrix = userRatings_no_outliers.corr(method='pearson', min_periods=100)

# Get user ratings
myRatings = userRatings_no_outliers.loc[0].dropna()

# Initialize SimulationUser
simulationUser = pd.Series()

# Generate recommendation
for i in range(0, len(myRatings.index)):
    print('Adding Simulation for ' + myRatings.index[i] + '...')
    # Retrieve similar movies to the ones I rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Scale the similarity by how well I rated it
    sims = sims.map(lambda x: x * myRatings.iloc[i])
    # Add score to the list of similarity candidates
    simulationUser = simulationUser._append(sims)

print("\nSorting...")
simulationUser = simulationUser.groupby(simulationUser.index).sum()
simulationUser.sort_values(inplace=True, ascending=False)
print(simulationUser.head(10))

print("\nFiltering....")
filteredSim = simulationUser.drop(myRatings.index)
print(filteredSim.head(10))
