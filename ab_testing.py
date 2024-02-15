######################################################
# Basic Statistics Concepts
######################################################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


############################
# Sampling
############################

population = np.random.randint(0, 80, 10000)
population.mean()
# 39.5439
np.random.seed(115)

sample = np.random.choice(a=population, size=100)
sample.mean()
# 34.09

np.random.seed(10)
sample1 = np.random.choice(a=population, size=100)
sample2 = np.random.choice(a=population, size=100)
sample3 = np.random.choice(a=population, size=100)
sample4 = np.random.choice(a=population, size=100)
sample5 = np.random.choice(a=population, size=100)
sample6 = np.random.choice(a=population, size=100)
sample7 = np.random.choice(a=population, size=100)
sample8 = np.random.choice(a=population, size=100)
sample9 = np.random.choice(a=population, size=100)
sample10 = np.random.choice(a=population, size=100)

(sample1.mean() + sample2.mean() + sample3.mean() + sample4.mean() + sample5.mean()
 + sample6.mean() + sample7.mean() + sample8.mean() + sample9.mean() + sample10.mean()) / 10
# 37.801

############################
# Descriptive Statistics
############################

df = sns.load_dataset("tips")
df.describe().T

############################
# Confidence Intervals
############################

# Confidence Interval Calculation for Numerical Variables in Tips Data Set
df = sns.load_dataset("tips")
df.describe().T

#                count     mean     std     min      25%      50%      75%     max
# total_bill 244.00000 19.78594 8.90241 3.07000 13.34750 17.79500 24.12750  50.81000
# tip        244.00000  2.99828 1.38364 1.00000  2.00000  2.90000  3.56250  10.00000
# size       244.00000  2.56967 0.95110 1.00000  2.00000  2.00000  3.00000  6.00000

df.head()

#    total_bill     tip     sex smoker  day    time  size
# 0    16.99000 1.01000  Female     No  Sun  Dinner     2
# 1    10.34000 1.66000    Male     No  Sun  Dinner     3
# 2    21.01000 3.50000    Male     No  Sun  Dinner     3
# 3    23.68000 3.31000    Male     No  Sun  Dinner     2
# 4    24.59000 3.61000  Female     No  Sun  Dinner     4

sms.DescrStatsW(df["total_bill"]).tconfint_mean()
# (18.663331704358473, 20.908553541543167)
sms.DescrStatsW(df["tip"]).tconfint_mean()
# (2.8237993062818205, 3.172758070767359)

# Confidence Interval Calculation for Numerical Variables in the Titanic Data Set

df = sns.load_dataset("titanic")
df.describe().T

#              count     mean      std     min      25%      50%      75%     max
# survived 891.00000  0.38384  0.48659 0.00000  0.00000  0.00000  1.00000  1.00000
# pclass   891.00000  2.30864  0.83607 1.00000  2.00000  3.00000  3.00000  3.00000
# age      714.00000 29.69912 14.52650 0.42000 20.12500 28.00000 38.00000  80.00000
# sibsp    891.00000  0.52301  1.10274 0.00000  0.00000  0.00000  1.00000  8.00000
# parch    891.00000  0.38159  0.80606 0.00000  0.00000  0.00000  0.00000  6.00000
# fare     891.00000 32.20421 49.69343 0.00000  7.91040 14.45420 31.00000  512.32920


sms.DescrStatsW(df["age"].dropna()).tconfint_mean()
# (28.631790041821507, 30.766445252296133)
sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()
# (28.93683123456734, 35.47158470258195)


######################################################
# Correlation
######################################################

df = sns.load_dataset('tips')
df.head()

df["total_bill"] = df["total_bill"] - df["tip"]

df.plot.scatter("tip", "total_bill")
plt.show()

df["tip"].corr(df["total_bill"])
# 0.5766634471096382

######################################################
# AB Testing
######################################################

# 1. Establish Hypotheses
# 2. Assumption Checking
# - 1. Normality Assumption
# - 2. Homogeneity of Variance
# 3. Application of Hypothesis
# - 1. If the assumptions are met, independent two-sample t test (parametric test)
# - 2. If the assumptions are not met, mannwhitneyu test (non-parametric test)
# 4. Interpret the results according to the p-value
# Note:
# - If normality is not achieved, number 2 directly. If variance homogeneity is not achieved, argument number 1 is entered.
# - It may be useful to perform outlier review and correction before normality review.

############################
# Application 1: Is there a difference between the account averages of smokers and non-smokers?
############################
df = sns.load_dataset("tips")

df.groupby("smoker").agg({"total_bill": "mean"})
#         total_bill
# smoker
# Yes       20.75634
# No        19.18828

############################
# 1. Establish the Hypothesis
############################

# H0: M1 = M2
# H1: M1 != M2

############################
# 2. Assumption Checking
############################

# Normality Assumption
# Variance Homogeneity

############################
# Normality Assumption
#############################

# H0: Normal distribution assumption is met.
# H1: ..not provided.


test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9367, p-value = 0.0002

# if p-value < 0.05 HO TO REJECT.
# if not p-value < 0.05 H0 TO NOT REJECT.


test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9045, p-value = 0.0000

############################
# Variance Homogeneity Assumption
############################

# H0: Variances Are Homogeneous
# H1: Variances Are Not Homogeneous

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 4.0537, p-value = 0.0452

# if p-value < 0.05 HO TO REJECT.
# if not p-value < 0.05 H0 TO NOT REJECT.

############################
# Application of Hypothesis 3 and 4
############################

# 1. If the assumptions are met, independent two-sample t-test (parametric test)
# 2. Mannwhitneyu test (non-parametric test) if the assumptions are not met

############################
# 1.1 If the assumptions are met, independent two-sample t test (parametric test)
############################

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 1.3384, p-value = 0.1820

# if p-value < 0.05 HO TO REJECT.
# if not p-value < 0.05 H0 TO NOT REJECT.

############################
# 1.2 If the assumptions are not met, mannwhitneyu test (non-parametric test)
############################

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 7531.5000, p-value = 0.3413


############################
# Application 2: Is there a statistically significant difference between the average ages of Titanic female and male passengers?
############################

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})
#             age
# sex
# female 27.91571
# male   30.72664

# 1. Establish hypotheses:
# H0: M1 = M2 (There is no statistically significant difference between the average ages of female and male passengers)
# H1: M1! = M2 (... exists)


# 2. Examine Assumptions

# Normality assumption
# H0: Normal distribution assumption is met.
# H1: ..not provided


test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9848, p-value = 0.0071

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9747, p-value = 0.0000


############################
# Variance Homogeneity Assumption
############################

# H0: Variances Are Homogeneous
# H1: Variances Are Not Homogeneous

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.0013, p-value = 0.9712

# Nonparametric as assumptions are not met

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 53212.5000, p-value = 0.0261


############################
# Application 3: Is there a statistically significant difference between the average ages of people with and without diabetes?
############################

df = pd.read_csv("datasets/diabetes.csv")
df.head()

#    Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin      BMI  DiabetesPedigreeFunction  Age  Outcome
# 0            6      148             72             35        0 33.60000                  0.62700   50        1
# 1            1       85             66             29        0 26.60000                  0.35100   31        0
# 2            8      183             64              0        0 23.30000                  0.67200   32        1
# 3            1       89             66             23       94 28.10000                  0.16700   21        0
# 4            0      137             40             35      168 43.10000                  2.28800   33        1

df.groupby("Outcome").agg({"Age": "mean"})
#              Age
# Outcome
# 0       31.19000
# 1       37.06716

# 1. Set up hypotheses
# H0: M1 = M2
# There is no statistically significant difference between the average ages of people with and without diabetes.
# H1: M1 != M2
# .... there is a difference.

# 2. Examine Assumptions

# Normality Assumption (H0: Normal distribution assumption is met.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9546, p-value = 0.0000

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.8012, p-value = 0.0000

# Nonparametric because the normality assumption is not met.

# Hypothesis (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 92050.0000, p-value = 0.0000


######################################################
# AB Testing (Two Sample Proportion Test)
######################################################

# H0: p1 = p2
# There is no statistically significant difference between the conversion rate of the new design and the conversion rate of the old design.
# H1: p1 != p2
# ... there is a difference

number_of_successes = np.array([300, 250])
observation_numbers = np.array([1000, 1100])

proportions_ztest(count=number_of_successes, nobs=observation_numbers)
# (3.7857863233209255, 0.0001532232957772221)

number_of_successes / observation_numbers
# array([0.3       , 0.22727273])


############################
# Application: Is There a Statistically Significant Difference Between the Survival Rates of Men and Women?
############################

# H0: p1 = p2
# There is no statistically significant difference between the survival rates of men and women

# H1: p1 != p2
# .. there is a difference

df = sns.load_dataset("titanic")

df.loc[df["sex"] == "female", "survived"].mean()
# 0.7420382165605095

df.loc[df["sex"] == "male", "survived"].mean()
# 0.18890814558058924

female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 16.2188, p-value = 0.0000

######################################################
# ANOVA (Analysis of Variance)
######################################################

# It is used to compare the averages of more than two groups.

df = sns.load_dataset("tips")

df.groupby("day")["total_bill"].mean()
# day
# Thur   17.68274
# Fri    17.15158
# Sat    20.44138
# Sun    21.41000
# Name: total_bill, dtype: float64

# 1. Set up hypotheses

# HO: m1 = m2 = m3 = m4
# There is no difference between group averages.

# H1: .. there is a difference

# 2. assumption check

# Normality assumption
# Assumption of homogeneity of variance

# If the assumption is met, one way anova
# If the assumption is not met, kruskal

# H0: Normal distribution assumption is met.

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, 'p-value: %.4f' % pvalue)
# Sun p-value: 0.0036
# Sat p-value: 0.0000
# Thur p-value: 0.0000
# Fri p-value: 0.0409

# H0: The assumption of variance homogeneity is satisfied.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.6654, p-value = 0.5741

# 3. Hypothesis testing and p-value interpretation

# None of them provide it.
df.groupby("day").agg({"total_bill": ["mean", "median"]})
#      total_bill
#            mean   median
# day
# Thur   17.68274 16.20000
# Fri    17.15158 15.38000
# Sat    20.44138 18.24000
# Sun    21.41000 19.63000

# HO: There is no statistically significant difference between group averages

# parametric anova test:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])
# F_onewayResult(statistic=2.7674794432863363, pvalue=0.04245383328952047)

# Nonparametric anova test:
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])
#  KruskalResult(statistic=10.403076391437086, pvalue=0.01543300820104127)

from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())

# Multiple Comparison of Means - Tukey HSD, FWER=0.05
# ====================================================
# group1 group2 meandiff p-adj   lower   upper  reject
# ----------------------------------------------------
#    Fri    Sat   3.2898 0.4541 -2.4799  9.0595  False
#    Fri    Sun   4.2584 0.2371 -1.5856 10.1025  False
#    Fri   Thur   0.5312 0.9957 -5.4434  6.5057  False
#    Sat    Sun   0.9686 0.8968 -2.6088   4.546  False
#    Sat   Thur  -2.7586 0.2374 -6.5455  1.0282  False
#    Sun   Thur  -3.7273 0.0668 -7.6264  0.1719  False
# ----------------------------------------------------