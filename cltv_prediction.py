##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma
##############################################################

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# The data set named Online Retail II is from a UK-based online sales store.
# Includes sales between 01/12/2009 - 09/12/2011.

# Features

# InvoiceNo: Invoice number. Unique number for each transaction, i.e. invoice. If it starts with C, the transaction is cancelled.
# StockCode: Product code. Unique number for each product.
# Description: Product name
# Quantity: Number of products. It indicates how many of the products on the invoices were sold.
# InvoiceDate: Invoice date and time.
# UnitPrice: Product price (in Pounds Sterling)
# CustomerID: Unique customer number
# Country: Country name. The country where the customer lives.

# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
#                   count                           mean                  min                  25%                  50%                  75%                  max       std
# Quantity    541910.0000                         9.5522          -80995.0000               1.0000               3.0000              10.0000           80995.0000  218.0810
# InvoiceDate      541910  2011-07-04 13:35:22.342307584  2010-12-01 08:26:00  2011-03-28 11:34:00  2011-07-19 17:17:00  2011-10-19 11:27:00  2011-12-09 12:50:00       NaN
# Price       541910.0000                         4.6111          -11062.0600               1.2500               2.0800               4.1300           38970.0000   96.7598
# Customer ID 406830.0000                     15287.6842           12346.0000           13953.0000           15152.0000           16791.0000           18287.0000 1713.6031

df.head()
#   Invoice StockCode                          Description  Quantity         InvoiceDate  Price  Customer ID         Country
# 0  536365    85123A   WHITE HANGING HEART T-LIGHT HOLDER         6 2010-12-01 08:26:00 2.5500   17850.0000  United Kingdom
# 1  536365     71053                  WHITE METAL LANTERN         6 2010-12-01 08:26:00 3.3900   17850.0000  United Kingdom
# 2  536365    84406B       CREAM CUPID HEARTS COAT HANGER         8 2010-12-01 08:26:00 2.7500   17850.0000  United Kingdom
# 3  536365    84029G  KNITTED UNION FLAG HOT WATER BOTTLE         6 2010-12-01 08:26:00 3.3900   17850.0000  United Kingdom
# 4  536365    84029E       RED WOOLLY HOTTIE WHITE HEART.         6 2010-12-01 08:26:00 3.3900   17850.0000  United Kingdom

df.isnull().sum()
# Invoice             0
# StockCode           0
# Description      1454
# Quantity            0
# InvoiceDate         0
# Price               0
# Customer ID    135080
# Country             0
# dtype: int64

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)


# recency: Time since last purchase. Weekly. (user specific)
# T: Age of the customer. Weekly. (how long before the date of analysis was the first purchase made)
# frequency: total number of recurring purchases (frequency>1)
# monetary: average earnings per purchase

cltv_df = df.groupby('Customer ID').agg(
    {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                     lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
     'Invoice': lambda Invoice: Invoice.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df.describe().T
#               count     mean      std    min      25%      50%      75%       max
# recency   4338.0000 130.4486 132.0396 0.0000   0.0000  92.5000 251.7500  373.0000
# T         4338.0000 223.8310 117.8546 1.0000 113.0000 249.0000 327.0000  374.0000
# frequency 4338.0000   4.2720   7.6980 1.0000   1.0000   2.0000   5.0000  209.0000
# monetary  4338.0000 364.1185 367.2582 3.4500 176.8512 288.2255 422.0294 6207.6700

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

##############################################################
# Establishing the BG-NBD Model
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])
# <lifetimes.BetaGeoFitter: fitted with 2845 subjects, a: 0.12, alpha: 11.41, b: 2.49, r: 2.18>
################################################################
# Who are the 10 customers from whom we expect the most purchases in 1 week?
################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)
# Customer ID
# 12748.0000   3.2495
# 14911.0000   3.1264
# 17841.0000   1.9402
# 13089.0000   1.5374
# 14606.0000   1.4639
# 15311.0000   1.4336
# 12971.0000   1.3569
# 14646.0000   1.2064
# 13408.0000   0.9862
# 18102.0000   0.9685
# dtype: float64

bgf.predict(1,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

################################################################
# Who are the 10 customers we expect to purchase the most in 1 month?
################################################################

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)
# Customer ID
# 12748.0000   12.9633
# 14911.0000   12.4722
# 17841.0000    7.7398
# 13089.0000    6.1330
# 14606.0000    5.8399
# 15311.0000    5.7191
# 12971.0000    5.4131
# 14646.0000    4.8119
# 13408.0000    3.9341
# 18102.0000    3.8636
# dtype: float64

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()
# 1776.8934732202886

################################################################
# What is the Expected Number of Sales of the Entire Company in 3 Months?
################################################################

bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()
# 5271.112433826376

cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
#####################################

plot_period_transactions(bgf)
plt.show()

##############################################################
# Establishing the GAMMA-GAMMA Model
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)
# Customer ID
# 12347.0000    631.9123
# 12348.0000    463.7460
# 12352.0000    224.8868
# 12356.0000    995.9989
# 12358.0000    631.9022
# 12359.0000   1435.0385
# 12360.0000    933.7905
# 12362.0000    532.2318
# 12363.0000    304.2643
# 12364.0000    344.1370
# dtype: float64

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)
# Customer ID
# 12415.0000   5772.1782
# 12590.0000   5029.4196
# 12435.0000   4288.9440
# 12409.0000   3918.8128
# 14088.0000   3917.1297
# 18102.0000   3870.9969
# 12753.0000   3678.5783
# 14646.0000   3654.8148
# 15749.0000   3216.0523
# 14096.0000   3196.4361
# dtype: float64

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
cltv_df.sort_values("expected_average_profit", ascending=False).head(10)
#              recency       T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_purc_3_month  expected_average_profit
# Customer ID
# 12415.0000   44.7143 48.2857         21 5724.3026                0.3796                 1.5139                 4.5080                5772.1782
# 12590.0000    0.0000 30.2857          2 4591.1725                0.0115                 0.0460                 0.1363                5029.4196
# 12435.0000   26.8571 38.2857          2 3914.9450                0.0763                 0.3041                 0.9035                4288.9440
# 12409.0000   14.7143 26.1429          3 3690.8900                0.1174                 0.4674                 1.3854                3918.8128
# 14088.0000   44.5714 46.1429         13 3864.5546                0.2603                 1.0379                 3.0896                3917.1297
# 18102.0000   52.2857 52.5714         60 3859.7391                0.9685                 3.8636                11.5112                3870.9969
# 12753.0000   48.4286 51.8571          6 3571.5650                0.1261                 0.5028                 1.4973                3678.5783
# 14646.0000   50.4286 50.7143         73 3646.0757                1.2064                 4.8119                14.3340                3654.8148
# 15749.0000   13.8571 47.5714          3 3028.7800                0.0280                 0.1116                 0.3320                3216.0523
# 14096.0000   13.8571 14.5714         17 3163.5882                0.7287                 2.8955                 8.5526                3196.4361

##############################################################
# Calculation of CLTV with BG-NBD and GG model.
##############################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3,  # 3 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()
# Customer ID
# 12347.0000   1128.4477
# 12348.0000    538.8089
# 12352.0000    517.5000
# 12356.0000   1083.0903
# 12358.0000    966.6727
# Name: clv, dtype: float64

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)
#       Customer ID  recency       T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_purc_3_month  expected_average_profit        clv
# 1122   14646.0000  50.4286 50.7143         73 3646.0757                1.2064                 4.8119                14.3340                3654.8148 55741.0845
# 2761   18102.0000  52.2857 52.5714         60 3859.7391                0.9685                 3.8636                11.5112                3870.9969 47412.5801
# 843    14096.0000  13.8571 14.5714         17 3163.5882                0.7287                 2.8955                 8.5526                3196.4361 29061.6614
# 36     12415.0000  44.7143 48.2857         21 5724.3026                0.3796                 1.5139                 4.5080                5772.1782 27685.1000
# 1257   14911.0000  53.1429 53.4286        201  691.7101                3.1264                12.4722                37.1641                 692.3264 27377.4115
# 2458   17450.0000  51.2857 52.5714         46 2863.2749                0.7474                 2.9815                 8.8830                2874.1987 27166.0643
# 874    14156.0000  51.5714 53.1429         55 2104.0267                0.8775                 3.5005                10.4298                2110.7542 23424.4032
# 2487   17511.0000  52.8571 53.4286         31 2933.9431                0.5088                 2.0298                 6.0476                2950.5801 18986.6123
# 2075   16684.0000  50.4286 51.2857         28 2209.9691                0.4781                 1.9068                 5.6801                2223.8850 13440.4131
# 650    13694.0000  52.7143 53.4286         50 1275.7005                0.8008                 3.1946                 9.5186                1280.2183 12966.1347


##############################################################
# Creating Segments Based on CLTV
##############################################################

cltv_final
#       Customer ID  recency       T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_purc_3_month  expected_average_profit       clv
# 0      12347.0000  52.1429 52.5714          7  615.7143                0.1413                 0.5635                 1.6784                 631.9123 1128.4477
# 1      12348.0000  40.2857 51.2857          4  442.6950                0.0920                 0.3668                 1.0920                 463.7460  538.8089
# 2      12352.0000  37.1429 42.4286          8  219.5425                0.1824                 0.7271                 2.1631                 224.8868  517.5000
# 3      12356.0000  43.1429 46.5714          3  937.1433                0.0862                 0.3435                 1.0222                 995.9989 1083.0903
# 4      12358.0000  21.2857 21.5714          2  575.2100                0.1223                 0.4862                 1.4388                 631.9022  966.6727
#            ...      ...     ...        ...       ...                   ...                    ...                    ...                      ...       ...
# 2840   18272.0000  34.8571 35.2857          6  513.0967                0.1721                 0.6856                 2.0369                 529.0185 1146.2057
# 2841   18273.0000  36.4286 36.8571          3   68.0000                0.1043                 0.4157                 1.2352                  73.4942   96.5648
# 2842   18282.0000  16.8571 18.1429          2   89.0250                0.1357                 0.5392                 1.5934                  99.5249  168.5946
# 2843   18283.0000  47.5714 48.2857         16  130.9300                0.3017                 1.2034                 3.5831                 132.6012  505.5117
# 2844   18287.0000  22.5714 28.8571          3  612.4267                0.1208                 0.4810                 1.4267                 651.3462  988.3029
# [2845 rows x 10 columns]

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(20)
#       Customer ID  recency       T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_purc_3_month  expected_average_profit        clv segment
# 1122   14646.0000  50.4286 50.7143         73 3646.0757                1.2064                 4.8119                14.3340                3654.8148 55741.0845       A
# 2761   18102.0000  52.2857 52.5714         60 3859.7391                0.9685                 3.8636                11.5112                3870.9969 47412.5801       A
# 843    14096.0000  13.8571 14.5714         17 3163.5882                0.7287                 2.8955                 8.5526                3196.4361 29061.6614       A
# 36     12415.0000  44.7143 48.2857         21 5724.3026                0.3796                 1.5139                 4.5080                5772.1782 27685.1000       A
# 1257   14911.0000  53.1429 53.4286        201  691.7101                3.1264                12.4722                37.1641                 692.3264 27377.4115       A
# 2458   17450.0000  51.2857 52.5714         46 2863.2749                0.7474                 2.9815                 8.8830                2874.1987 27166.0643       A
# 874    14156.0000  51.5714 53.1429         55 2104.0267                0.8775                 3.5005                10.4298                2110.7542 23424.4032       A
# 2487   17511.0000  52.8571 53.4286         31 2933.9431                0.5088                 2.0298                 6.0476                2950.5801 18986.6123       A
# 2075   16684.0000  50.4286 51.2857         28 2209.9691                0.4781                 1.9068                 5.6801                2223.8850 13440.4131       A
# 650    13694.0000  52.7143 53.4286         50 1275.7005                0.8008                 3.1946                 9.5186                1280.2183 12966.1347       A
# 841    14088.0000  44.5714 46.1429         13 3864.5546                0.2603                 1.0379                 3.0896                3917.1297 12875.7762       A
# 1754   16000.0000   0.0000  0.4286          3 2335.1200                0.4220                 1.6639                 4.8439                2479.8048 12751.9305       A
# 1441   15311.0000  53.2857 53.4286         91  667.7791                1.4336                 5.7191                17.0411                 669.0960 12132.2867       A
# 373    13089.0000  52.2857 52.8571         97  606.3625                1.5374                 6.1330                18.2736                 607.4877 11811.7794       A
# 1324   15061.0000  52.5714 53.2857         48 1120.6019                0.7717                 3.0786                 9.1730                1124.7458 10977.9341       A
# 949    14298.0000  50.2857 51.5714         44 1162.8627                0.7277                 2.9026                 8.6470                1167.5522 10742.1077       A
# 1647   15769.0000  51.8571 53.1429         26 1873.6702                0.4329                 1.7268                 5.1449                1886.4041 10326.6579       A
# 2774   18139.0000   0.0000  2.7143          6 1406.3900                0.5289                 2.0904                 6.1111                1448.9171  9403.5742       A
# 2652   17841.0000  53.0000 53.4286        124  330.1344                1.9402                 7.7398                23.0625                 330.6271  8113.3777       A
# 2699   17949.0000  52.8571 53.1429         45  848.4303                0.7280                 2.9040                 8.6524                 851.7980  7842.0081       A

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})
#         Customer ID                          recency                        T                    frequency               monetary                      expected_purc_1_week                 expected_purc_1_month                 expected_purc_3_month                  expected_average_profit                        clv
#               count           sum       mean   count        sum    mean count        sum    mean     count   sum    mean    count         sum     mean                count      sum   mean                 count      sum   mean                 count       sum   mean                   count         sum     mean count          sum      mean
# segment
# D               712 11077635.0000 15558.4761     712 15716.5714 22.0738   712 28811.0000 40.4649       712  2182  3.0646      712 130981.7123 183.9631                  712  50.5898 0.0711                   712 201.4774 0.2830                   712  598.0905 0.8400                     712 141999.6650 199.4377   712  102027.4901  143.2970
# C               711 10885150.0000 15309.6343     711 21806.1429 30.6697   711 27097.0000 38.1111       711  2912  4.0956      711 193174.9847 271.6948                  711  85.7309 0.1206                   711 341.3936 0.4802                   711 1013.2176 1.4251                     711 206179.7461 289.9856   711  270743.0065  380.7919
# B               711 10915854.0000 15352.8186     711 20985.0000 29.5148   711 24751.1429 34.8117       711  3869  5.4416      711 265517.6312 373.4425                  711 115.5066 0.1625                   711 459.6926 0.6465                   711 1362.6595 1.9165                     711 280058.9431 393.8944   711  489356.3925  688.2650
# A               711 10627572.0000 14947.3586     711 22333.1429 31.4109   711 24518.1429 34.4840       711  8076 11.3586      711 469159.4581 659.8586                  711 194.4993 0.2736                   711 774.3298 1.0891                   711 2297.1448 3.2309                     711 487674.7239 685.8998   711 1580097.9284 2222.3600










