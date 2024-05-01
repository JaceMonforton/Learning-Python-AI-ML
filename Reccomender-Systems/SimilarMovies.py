import pandas as pd 

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('C:\\MLCourse\\ml-100k\\u.data', sep='\t', names=r_cols, usecols=range(3))

m_cols = ['movie_id', 'title']
movies = pd.read_csv('C:\\MLCourse\\ml-100k\\u.item', sep='|', names=m_cols, usecols=range(2), encoding='latin1')

ratings = pd.merge(movies, ratings)

# print(ratings.head())

movieRatings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')
# print(movieRatings.head())

StarWarsRating = movieRatings['Star Wars (1977)']
# print(StarWarsRating.head())

similarMovies = movieRatings.corrwith(StarWarsRating) #computes correlation with every other column in df with the starwars column tosee similar movies (-1,1)
similarMovies = similarMovies.dropna() #drop NaN values

similarMovies = similarMovies.sort_values(ascending=False) # Sort the Series in descending order

df = pd.DataFrame(similarMovies)
print(df.head(10)) 
