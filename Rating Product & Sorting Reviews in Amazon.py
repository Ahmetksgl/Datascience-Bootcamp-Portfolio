
# ###################################################
# PROJECT: Rating Product & Sorting Reviews in Amazon
# ###################################################
#
# ###################################################
# Business Problem
# ###################################################

# One of the most significant challenges in e-commerce is accurately calculating ratings for products post-purchase.
# Solving this problem means ensuring greater customer satisfaction for the e-commerce platform, enhancing product visibility for sellers, and providing a seamless shopping experience for buyers.
# Another issue is accurately sorting the reviews given to products. Since misleading reviews can directly impact a product's sales, they can result in both financial losses and loss of customers.
# By addressing these two fundamental problems, e-commerce platforms and sellers can increase their sales, while customers can complete their purchasing journey smoothly.
# ###################################################
# Data Set Story
# ###################################################

# This dataset contains Amazon product data along with various metadata related to product categories.
# It includes user ratings and reviews for the product with the most reviews in the electronics category.
# Variables:
# reviewerID: User ID
# asin: Product ID
# reviewerName: User Name
# helpful: Helpful review rating
# reviewText: Review
# overall: Product rating
# summary: Review summary
# unixReviewTime: Review time
# reviewTime: Review time (Raw)
# day_diff: Number of days elapsed since the review
# helpful_yes: Number of times the review was found helpful
# total_vote: Total number of votes for the review



import pandas as pd
import math
import scipy.stats as st
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv('amazon_review.csv')
df.head()

#        reviewerID        asin  reviewerName helpful                                         reviewText  overall                                 summary  unixReviewTime  reviewTime  day_diff  helpful_yes  total_vote
# 0  A3SBTW3WS4IQSN  B007WTAJTO           NaN  [0, 0]                                         No issues.  4.00000                              Four Stars      1406073600  2014-07-23       138            0           0
# 1  A18K1ODH1I2MVB  B007WTAJTO          0mie  [0, 0]  Purchased this for my device, it worked as adv...  5.00000                           MOAR SPACE!!!      1382659200  2013-10-25       409            0           0
# 2  A2FII3I2MBMUIA  B007WTAJTO           1K3  [0, 0]  it works as expected. I should have sprung for...  4.00000               nothing to really say....      1356220800  2012-12-23       715            0           0
# 3   A3H99DFEG68SR  B007WTAJTO           1m2  [0, 0]  This think has worked out great.Had a diff. br...  5.00000  Great buy at this price!!!  *** UPDATE      1384992000  2013-11-21       382            0           0
# 4  A375ZM4U047O79  B007WTAJTO  2&amp;1/2Men  [0, 0]  Bought it with Retail Packaging, arrived legit...  5.00000                        best deal around      1373673600  2013-07-13       513            0           0

df.dtypes

# reviewerID         object
# asin               object
# reviewerName       object
# helpful            object
# reviewText         object
# overall           float64
# summary            object
# unixReviewTime      int64
# reviewTime         object
# day_diff            int64
# helpful_yes         int64
# total_vote          int64
# dtype: object

df['overall'].describe().T

# count   4915.00000
# mean       4.58759
# std        0.99685
# min        1.00000
# 25%        5.00000
# 50%        5.00000
# 75%        5.00000
# max        5.00000
# Name: overall, dtype: float64


df['day_diff'].describe().T

# count   4915.00000
# mean     437.36704
# std      209.43987
# min        1.00000
# 25%      281.00000
# 50%      431.00000
# 75%      601.00000
# max     1064.00000
# Name: day_diff, dtype: float64

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe['day_diff'] <= 100, 'overall'].mean() * w1 / 100 + \
        dataframe.loc[(dataframe['day_diff'] > 100) & (dataframe['day_diff'] <= 300), 'overall'].mean() * w2 / 100 + \
        dataframe.loc[(dataframe['day_diff'] > 300) & (dataframe['day_diff'] <= 600), 'overall'].mean() * w3 / 100 + \
        dataframe.loc[dataframe['day_diff'] > 600, 'overall'].mean() * w4 / 100


time_based_weighted_average(df)
# 4.629746389532435

df.loc[df['day_diff'], 'overall'].mean()
# 4.579654120040692
df.loc[df['day_diff']<= 100, 'overall'].mean()
# 4.749049429657795
df.loc[(df['day_diff']<= 100) & (df['day_diff'] <= 300), 'overall'].mean()
# 4.749049429657795
df.loc[(df['day_diff']<= 300) & (df['day_diff'] <= 600), 'overall'].mean()
# 4.699785561115082
df.loc[df['day_diff']> 600, 'overall'].mean()
# 4.446791226645004

###################################################
# Let's Determine 20 Reviews to Be Displayed on the Product Detail Page for the Product
###################################################

df.loc[df['total_vote']> 3].head(10)

#           reviewerID        asin                reviewerName     helpful                                         reviewText  overall                                            summary  unixReviewTime  reviewTime  day_diff  helpful_yes  total_vote
# 93     A837QPVOZ9YAD  B007WTAJTO                     Airedad    [15, 21]  I'm amazed.  I ordered this from Amazon on Tue...  5.00000  Very fast class 10 card - and excellent servic...      1343174400  2012-07-25       866           15          21
# 95      AG7K6P2FN006  B007WTAJTO                         ajb      [4, 4]  I bought the 64GB MicroSD card at Best Buy bec...  5.00000                    Great Quality, Plenty of Memory      1364169600  2013-03-25       623            4           4
# 121   A2Z4VVF1NTJWPB  B007WTAJTO                      A. Lee      [5, 5]  Update: providing an update with regard to San...  5.00000                     ready for use on the Galaxy S3      1346803200  2012-05-09       943            5           5
# 187   A1OQX81M4BVVPT  B007WTAJTO     Amazon Customer "Happy"      [3, 4]  My Samsung Galaxy Note 2 came with 8 gigs of i...  5.00000  Works Perfectly with Samsung Galaxy Note 2 (Th...      1358899200  2013-01-23       684            3           4
# 315   A2J26NNQX6WKAU  B007WTAJTO  Amazon Customer "johncrea"    [38, 48]  Bought this card to use with my Samsung Galaxy...  5.00000  Samsung Galaxy Tab2 works with this card if re...      1344816000  2012-08-13       847           38          48
# 317   A1ZQAQFYSXL5MQ  B007WTAJTO     Amazon Customer "Kelly"  [422, 495]  If your card gets hot enough to be painful, it...  1.00000                                Warning, read this!      1346544000  2012-02-09      1033          422         495
# 323   A15X60NOGL3WDW  B007WTAJTO      Amazon Customer "Milo"      [6, 7]  I replaced a 64GB card in my note 3 with this ...  5.00000                          Working fine in my Note 3      1396310400  2014-01-04       338            6           7
# 648   A2X4A3S0BNSISR  B007WTAJTO              Bobby Shornock      [4, 5]  I purchased 2 128GB SD cards, one for my Nokia...  5.00000  The 128GB card works in quite a few newer devi...      1395792000  2014-03-26       257            4           5
# 861   A1PZHJTGE2TS99  B007WTAJTO                   Cat Daddy      [5, 6]  The advertisement calls this card a "class 10"...  4.00000  Very nice for the price!!! Read for REAL WORLD...      1345507200  2012-08-21       839            5           6
# 1006  A1GDY4MP5QVXPD  B007WTAJTO                cmoisuperlea      [4, 4]  I own a couple of these and I've tested them o...  5.00000                                Works as described.      1388361600  2013-12-30       343            4           4


###################################################
# Step 1. Create the variable helpful_no
###################################################

df['helpful_no'] = df['total_vote'] - df['helpful_yes']

###################################################
# Step 2. Calculate Score_pos_neg_diff, Score_average_rating and wilson_lower_bound Scores and Add them to the Data
###################################################

############### score_pos_neg_diff ###########
def score_up_down_diff(up, down):
    return up - down

df['score_pos_neg_diff'] = df.apply(lambda x: score_up_down_diff(x['helpful_yes'],\
                                                                 x['helpful_no']), axis=1)

############### score_average_rating ###########

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df['score_average_rating'] = df.apply(lambda x: score_average_rating(x['helpful_yes'],\
                                                                 x['helpful_no']), axis=1)

############### wilson_lower_bound ###########

def wilson_lower_bound(up, down, confidence=0.95):
    """

    Wilson Lower Bound Score calculation:

    The lower bound of the confidence interval to be calculated for the Bernoulli parameter p is considered as the WLB
    (Wilson Lower Bound) score.
    The calculated score is used for product ranking.
    Note:
    If the scores are between 1-5, they are categorized as negative for 1-3 and positive for 4-5, and can be adjusted
    to fit Bernoulli distribution. However, this introduces some problems. Therefore, Bayesian average rating should
    be performed

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x['helpful_yes'],\
                                                                 x['helpful_no']), axis=1)

##################################################
# Step 3. Let's Identify 20 Comments
###################################################

df.sort_values('wilson_lower_bound', ascending=False).head(20)


#           reviewerID        asin                          reviewerName       helpful                                         reviewText  overall                                            summary  unixReviewTime  reviewTime  day_diff  helpful_yes  total_vote  helpful_no  score_pos_neg_diff  score_average_rating  wilson_lower_bound
# 2031  A12B7ZMXFI6IXY  B007WTAJTO                  Hyoun Kim "Faluzure"  [1952, 2020]  [[ UPDATE - 6/19/2014 ]]So my lovely wife boug...  5.00000  UPDATED - Great w/ Galaxy S4 & Galaxy Tab 4 10...      1367366400  2013-01-05       702         1952        2020          68                1884               0.96634             0.95754
# 3449   AOEAD7DPLZE53  B007WTAJTO                     NLee the Engineer  [1428, 1505]  I have tested dozens of SDHC and micro-SDHC ca...  5.00000  Top of the class among all (budget-priced) mic...      1348617600  2012-09-26       803         1428        1505          77                1351               0.94884             0.93652
# 4212   AVBMZZAFEKO58  B007WTAJTO                           SkincareCEO  [1568, 1694]  NOTE:  please read the last update (scroll to ...  1.00000  1 Star reviews - Micro SDXC card unmounts itse...      1375660800  2013-05-08       579         1568        1694         126                1442               0.92562             0.91214
# 317   A1ZQAQFYSXL5MQ  B007WTAJTO               Amazon Customer "Kelly"    [422, 495]  If your card gets hot enough to be painful, it...  1.00000                                Warning, read this!      1346544000  2012-02-09      1033          422         495          73                 349               0.85253             0.81858
# 4672  A2DKQQIZ793AV5  B007WTAJTO                               Twister      [45, 49]  Sandisk announcement of the first 128GB micro ...  5.00000  Super high capacity!!!  Excellent price (on Am...      1394150400  2014-07-03       158           45          49           4                  41               0.91837             0.80811
# 1835  A1J6VSUM80UAF8  B007WTAJTO                           goconfigure      [60, 68]  Bought from BestBuy online the day it was anno...  5.00000                                           I own it      1393545600  2014-02-28       283           60          68           8                  52               0.88235             0.78465
# 3981  A1K91XXQ6ZEBQR  B007WTAJTO            R. Sutton, Jr. "RWSynergy"    [112, 139]  The last few days I have been diligently shopp...  5.00000  Resolving confusion between "Mobile Ultra" and...      1350864000  2012-10-22       777          112         139          27                  85               0.80576             0.73214
# 3807   AFGRMORWY2QNX  B007WTAJTO                            R. Heisler      [22, 25]  I bought this card to replace a lost 16 gig in...  3.00000   Good buy for the money but wait, I had an issue!      1361923200  2013-02-27       649           22          25           3                  19               0.88000             0.70044
# 4306   AOHXKM5URSKAB  B007WTAJTO                         Stellar Eller      [51, 65]  While I got this card as a "deal of the day" o...  5.00000                                      Awesome Card!      1339200000  2012-09-06       823           51          65          14                  37               0.78462             0.67033
# 4596  A1WTQUOQ4WG9AI  B007WTAJTO           Tom Henriksen "Doggy Diner"     [82, 109]  Hi:I ordered two card and they arrived the nex...  1.00000     Designed incompatibility/Don't support SanDisk      1348272000  2012-09-22       807           82         109          27                  55               0.75229             0.66359
# 315   A2J26NNQX6WKAU  B007WTAJTO            Amazon Customer "johncrea"      [38, 48]  Bought this card to use with my Samsung Galaxy...  5.00000  Samsung Galaxy Tab2 works with this card if re...      1344816000  2012-08-13       847           38          48          10                  28               0.79167             0.65741
# 1465   A6I8KXYK24RTB  B007WTAJTO                              D. Stein        [7, 7]  I for one have not bought into Google's, or an...  4.00000                                           Finally.      1397433600  2014-04-14       238            7           7           0                   7               1.00000             0.64567
# 1609  A2TPXOZSU1DACQ  B007WTAJTO                                Eskimo        [7, 7]  I have always been a sandisk guy.  This cards ...  5.00000                  Bet you wish you had one of these      1395792000  2014-03-26       257            7           7           0                   7               1.00000             0.64567
# 4302  A2EL2GWJ9T0DWY  B007WTAJTO                             Stayeraug      [14, 16]  So I got this SD specifically for my GoPro Bla...  5.00000                        Perfect with GoPro Black 3+      1395360000  2014-03-21       262           14          16           2                  12               0.87500             0.63977
# 4072  A22GOZTFA02O2F  B007WTAJTO                           sb21 "sb21"        [6, 6]  I used this for my Samsung Galaxy Tab 2 7.0 . ...  5.00000               Used for my Samsung Galaxy Tab 2 7.0      1347321600  2012-11-09       759            6           6           0                   6               1.00000             0.60967
# 1072  A2O96COBMVY9C4  B007WTAJTO                        Crysis Complex        [5, 5]  What more can I say? The 64GB micro SD works f...  5.00000               Works wonders for the Galaxy Note 2!      1349395200  2012-05-10       942            5           5           0                   5               1.00000             0.56552
# 2583  A3MEPYZVTAV90W  B007WTAJTO                               J. Wong        [5, 5]  I bought this Class 10 SD card for my GoPro 3 ...  5.00000                  Works Great with a GoPro 3 Black!      1370649600  2013-08-06       489            5           5           0                   5               1.00000             0.56552
# 121   A2Z4VVF1NTJWPB  B007WTAJTO                                A. Lee        [5, 5]  Update: providing an update with regard to San...  5.00000                     ready for use on the Galaxy S3      1346803200  2012-05-09       943            5           5           0                   5               1.00000             0.56552
# 1142  A1PLHPPAJ5MUXG  B007WTAJTO  Daniel Pham(Danpham_X @ yahoo.  com)        [5, 5]  As soon as I saw that this card was announced ...  5.00000                          Great large capacity card      1396396800  2014-02-04       307            5           5           0                   5               1.00000             0.56552
# 1753   ALPLKR59QMBUX  B007WTAJTO                             G. Becker        [5, 5]  Puchased this card right after I received my S...  5.00000                    Use Nothing Other Than the Best      1350864000  2012-10-22       777            5           5           0                   5               1.00000             0.56552


