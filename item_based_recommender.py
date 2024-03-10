###########################################
# Item-Based Collaborative Filtering
###########################################

# Data set: https://grouplens.org/datasets/movielens/

######################################
# Step 1: Preparation of the Data Set
######################################
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()
#    movieId             title                                       genres  userId  rating            timestamp
# 0        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy     3.0     4.0  1999-12-11 13:36:47
# 1        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy     6.0     5.0  1997-03-13 17:50:52
# 2        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy     8.0     4.0  1996-06-05 13:37:51
# 3        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy    10.0     4.0  1999-11-25 02:44:47
# 4        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy    11.0     4.5  2009-01-02 01:13:41


######################################
# Step 2: Creating User Movie Df
######################################

df.shape
# (20000797, 6)

df["title"].nunique()
# 27262

df["title"].value_counts().head()
# title
# Pulp Fiction (1994)                 67310
# Forrest Gump (1994)                 66172
# Shawshank Redemption, The (1994)    63366
# Silence of the Lambs, The (1991)    63299
# Jurassic Park (1993)                59715
# Name: count, dtype: int64

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["count"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape
# (17766015, 6)

common_movies["title"].nunique()
# 3159
df["title"].nunique()
# 27262

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape
# (138493, 3159)

######################################
# Step 3: Making Item-Based Movie Recommendations
######################################

movie_name = "Matrix, The (1999)"
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)
# title
# Matrix, The (1999)                                           1.000000
# Matrix Reloaded, The (2003)                                  0.516906
# Matrix Revolutions, The (2003)                               0.449588
# Animatrix, The (2003)                                        0.367151
# Blade (1998)                                                 0.334493
# Terminator 2: Judgment Day (1991)                            0.333882
# Minority Report (2002)                                       0.332434
# Edge of Tomorrow (2014)                                      0.326762
# Mission: Impossible (1996)                                   0.320815
# Lord of the Rings: The Fellowship of the Ring, The (2001)    0.318726
# dtype: float64

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name
# 'Mask, The (1994)'
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)
# title
# Mask, The (1994)                         1.000000
# Liar Liar (1997)                         0.505521
# Ace Ventura: Pet Detective (1994)        0.468835
# Ace Ventura: When Nature Calls (1995)    0.459327
# Bruce Almighty (2003)                    0.421551
# Dr. Dolittle 2 (2001)                    0.416060
# Nutty Professor, The (1996)              0.414241
# Men in Black (a.k.a. MIB) (1997)         0.406839
# Mask (1985)                              0.394841
# Dr. Dolittle (1998)                      0.392576
# dtype: float64


def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Lord", user_movie_df)
# ['Lord of Illusions (1995)',
#  'Lord of War (2005)',
#  'Lord of the Flies (1963)',
#  'Lord of the Flies (1990)',
#  'Lord of the Rings, The (1978)',
#  'Lord of the Rings: The Fellowship of the Ring, The (2001)',
#  'Lord of the Rings: The Return of the King, The (2003)',
#  'Lord of the Rings: The Two Towers, The (2002)']


######################################
# Step 4: Preparing the Working Scriptı
######################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)
# title
# Matrix, The (1999)                                           1.000000
# Matrix Reloaded, The (2003)                                  0.516906
# Matrix Revolutions, The (2003)                               0.449588
# Animatrix, The (2003)                                        0.367151
# Blade (1998)                                                 0.334493
# Terminator 2: Judgment Day (1991)                            0.333882
# Minority Report (2002)                                       0.332434
# Edge of Tomorrow (2014)                                      0.326762
# Mission: Impossible (1996)                                   0.320815
# Lord of the Rings: The Fellowship of the Ring, The (2001)    0.318726
# dtype: float64

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)
# title
# Sea of Love (1989)                                 1.000000
# Adventures of Pinocchio, The (1996)                0.678841
# Guys and Dolls (1955)                              0.575418
# Marvin's Room (1996)                               0.560090
# Tom and Huck (1995)                                0.553511
# Frozen (2013)                                      0.526933
# Eagle Eye (2008)                                   0.514546
# Futurama: The Beast with a Billion Backs (2008)    0.512989
# Mask (1985)                                        0.511655
# Houseguest (1994)                                  0.510609
# dtype: float64




