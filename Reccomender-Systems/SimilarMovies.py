import pandas as pd 
import numpy as np

# Read the data
ratings = pd.read_csv('C:\\MLCourse\\ml-100k\\u.data', sep='\t', names=['user_id', 'movie_id', 'rating'], usecols=range(3))

movies = pd.read_csv('C:\\MLCourse\\ml-100k\\u.item', sep='|', names=['movie_id', 'title'], usecols=range(2), encoding='latin1')

# Merge dataframes
ratings = pd.merge(movies, ratings)

# Create a pivot table of movie ratings
movieRatings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')

# Extract ratings for "Star Wars (1977)"
StarWarsRating = movieRatings['Star Wars (1977)']

similarMovies = movieRatings.corrwith(StarWarsRating)

# Drop NaN values
similarMovies = similarMovies.dropna()

# Sort by similarity
similarMovies = similarMovies.sort_values(ascending=False)

# Aggregate movie statistics
movieStats = ratings.groupby('title').agg(size=('rating', np.size), mean=('rating', 'mean'))

# Filter popular movies (rated by at least 100 users)
popularMovies = movieStats['size'] >= 100


# Sort by the mean rating
movieStats = movieStats[popularMovies].sort_values(by='mean', ascending=False)[:15]

# Reset index after filtering
movieStats.reset_index(inplace=True)

# Join movie stats with similarity scores
df = pd.merge(movieStats, pd.DataFrame(similarMovies, columns=['similarity']), on='title')

df = df.sort_values(['similarity'], ascending=False)[:15]
print(df.head())
