###################################################
# Rating Products
###################################################

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating


############################################
# Application: User and Time Weighted Course Score Calculation
############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# (50+ Hours) Python A-Z™: Data Science ve Machine Learning
# Score: 4.8 (4.764925)
# Total Score: 4611
# Score Percentages: 75, 20, 4, 1, <1
# Approximate Numerical Equivalents: 3458, 922, 184, 46, 6

df = pd.read_csv("datasets/course_reviews.csv")
df.head()

#    Rating            Timestamp             Enrolled  Progress  Questions Asked  Questions Answered
# 0 5.00000  2021-02-05 07:45:55  2021-01-25 15:12:08   5.00000          0.00000             0.00000
# 1 5.00000  2021-02-04 21:05:32  2021-02-04 20:43:40   1.00000          0.00000             0.00000
# 2 4.50000  2021-02-04 20:34:03  2019-07-04 23:23:27   1.00000          0.00000             0.00000
# 3 5.00000  2021-02-04 16:56:28  2021-02-04 14:41:29  10.00000          0.00000             0.00000
# 4 4.00000  2021-02-04 15:00:24  2020-10-13 03:10:07  10.00000          0.00000             0.00000
# 5 4.00000  2021-02-04 12:42:36  2021-02-01 15:40:13   1.00000          0.00000             0.00000
# 6 5.00000  2021-02-04 12:25:30  2020-11-30 19:23:54  85.00000          0.00000             4.00000
# 7 4.50000  2021-02-04 11:13:15  2021-01-08 08:05:42  10.00000          0.00000             0.00000
# 8 5.00000  2021-02-04 08:59:53  2021-02-02 18:14:49   1.00000          0.00000             0.00000
# 9 5.00000  2021-02-03 22:40:04  2020-08-01 22:30:42  35.00000          0.00000             0.00000

df.shape
# (4323, 6)

# rating distribution
df["Rating"].value_counts()
# Rating
# 5.00000    3267
# 4.50000     475
# 4.00000     383
# 3.50000      96
# 3.00000      62
# 1.00000      15
# 2.00000      12
# 2.50000      11
# 1.50000       2
# Name: count, dtype: int64

df["Questions Asked"].value_counts()
# Questions Asked
# 0.00000     3867
# 1.00000      276
# 2.00000       80
# 3.00000       43
# 4.00000       15
# 5.00000       13
# 6.00000        9
# 8.00000        5
# 9.00000        3
# 14.00000       2
# 11.00000       2
# 7.00000        2
# 10.00000       2
# 15.00000       2
# 22.00000       1
# 12.00000       1
# Name: count, dtype: int64

df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})
#                  Questions Asked  Rating
# Questions Asked
# 0.00000                     3867 4.76519
# 1.00000                      276 4.74094
# 2.00000                       80 4.80625
# 3.00000                       43 4.74419
# 4.00000                       15 4.83333
# 5.00000                       13 4.65385
# 6.00000                        9 5.00000
# 7.00000                        2 4.75000
# 8.00000                        5 4.90000
# 9.00000                        3 5.00000
# 10.00000                       2 5.00000
# 11.00000                       2 5.00000
# 12.00000                       1 5.00000
# 14.00000                       2 4.50000
# 15.00000                       2 3.00000
# 22.00000                       1 5.00000

####################
# Average
####################

# Average Score
df["Rating"].mean()
# 4.764284061993986

####################
# Time-Based Weighted Average
####################
# Weighted Average by Scoring Times

df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 4323 entries, 0 to 4322
# Data columns (total 6 columns):
#  #   Column              Non-Null Count  Dtype
# ---  ------              --------------  -----
#  0   Rating              4323 non-null   float64
#  1   Timestamp           4323 non-null   object
#  2   Enrolled            4323 non-null   object
#  3   Progress            4323 non-null   float64
#  4   Questions Asked     4323 non-null   float64
#  5   Questions Answered  4323 non-null   float64
# dtypes: float64(4), object(2)
# memory usage: 202.8+ KB

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
current_date = pd.to_datetime('2021-02-10 0:0:0')
df["days"] = (current_date - df["Timestamp"]).dt.days
df.loc[df["days"] <= 30, "Rating"].mean()
# 4.775773195876289
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()
# 4.763833992094861
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()
# 4.752503576537912
df.loc[(df["days"] > 180), "Rating"].mean()
# 4.76641586867305

df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
    df.loc[(df["days"] > 180), "Rating"].mean() * 22/100
# 4.765025682267194
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100

time_based_weighted_average(df)
# 4.765025682267194
time_based_weighted_average(df, 30, 26, 22, 22)
# 4.765491074653962


####################
# User-Based Weighted Average
####################

df.head()

df.groupby("Progress").agg({"Rating": "mean"})
#            Rating
# Progress
# 0.00000   4.67391
# 1.00000   4.64269
# 2.00000   4.65476
# 3.00000   4.66355
# 4.00000   4.77733
# 5.00000   4.69821
# 6.00000   4.75510
# 7.00000   4.73256
# 8.00000   4.74194
# 9.00000   4.83125
# 10.00000  4.74569
# 11.00000  4.83333
# .
# .
# .
# 66.00000  5.00000
# 67.00000  5.00000
# 69.00000  5.00000
# 70.00000  4.78947
# 71.00000  5.00000
# 72.00000  5.00000
# 73.00000  5.00000
# 74.00000  5.00000
# 75.00000  4.93750
# 77.00000  5.00000
# 78.00000  5.00000
# 80.00000  4.75000
# 83.00000  5.00000
# 84.00000  5.00000
# 85.00000  4.91379
# 87.00000  5.00000
# 89.00000  4.79412
# 90.00000  4.92308
# 91.00000  5.00000
# 93.00000  4.83333
# 94.00000  5.00000
# 95.00000  4.79412
# 97.00000  5.00000
# 98.00000  5.00000
# 100.00000 4.86632

df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100
# 4.800257704672543

def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100


user_based_weighted_average(df, 20, 24, 26, 30)
# 4.803286469062915

####################
# Weighted Rating
####################

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100

course_weighted_rating(df)
# 4.782641693469868
course_weighted_rating(df, time_w=40, user_w=60)
# 4.786164895710403









