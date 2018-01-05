import pandas as pd

# import the movie_id, user_id and rating for each movie per user.
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('C:/Users/1asch/udemy-datascience/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

# import the movie title and sinc it with the movie_id
m_cols = ['movie_id', 'title']
movies = pd.read_csv('C:/Users/1asch/udemy-datascience/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)

# We want to see the ratings per user, with the movies as columns and the rating as the values. 
userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')

# Making the correlation matrix for all movies compared to eachother. But we only want to see movies who have more than 100 pairs of users.
corrMatrix = userRatings.corr(method='pearson', min_periods=100)

# Take the first person in our matrix. Lets see what he has watched.
myRatings = userRatings.loc[0].dropna()
myRatings

# So we see he has watched Empire Strikes Back, Gone with the Wind, Star Wars.

# Now let's produce an algorithm which finds similar movies.
simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print ("Adding sims for " + myRatings.index[i] + "...")
    # Retrieve similar movies to this one that I rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Now scale its similarity by how well I rated this movie
    sims = sims.map(lambda x: x * myRatings[i])
    # Add the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)
    
#Glance at our results so far:
print ("sorting...")
simCandidates.sort_values(inplace = True, ascending = False)
print (simCandidates.head(10))