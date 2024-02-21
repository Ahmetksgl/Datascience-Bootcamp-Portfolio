#####################################################
# Comparison of AB Testing and Conversion of Bidding Methods
#####################################################

#####################################################
# Dataset Story
#####################################################

# What users see and click on in this data set containing a company's website information
# In addition to information such as advertising numbers, there is also earnings information from here.
# There are two separate data sets: Control and Test group. These data sets are located on separate pages of
# ab_testing.xlsx excel is taking.
# Maximum Bidding was applied to the control group and AverageBidding was applied to the test group.

# impression: Number of ad views
# Click: Number of clicks on the displayed ad
# Purchase: Number of products purchased after ads clicked
# Earning: Earnings earned after purchasing products

######################################################
# AB Testing (Independent Two Sample T Test)
######################################################

# 1. Establish Hypotheses
# 2. Assumption Checking
# - 1. Normality Assumption (shapiro)
# - 2. Variance Homogeneity (levene)
# 3. Application of Hypothesis
# - 1. Independent two sample t test if assumptions are met
# - 2. Mannwhitneyu test if assumptions are not met
# 4. Interpret the results according to the p-value
# Note:
# - If normality is not achieved, direct number 2. If variance homogeneity is not achieved, argument number 1 is entered.
# - It may be useful to perform outlier review and correction before normality review.



#####################################################
# Task 1: Prepare and Analyze Data
#####################################################
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

import colorama
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
pd.options.display.max_columns=None
pd.options.display.max_rows=10
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 1000

YEL = colorama.Fore.LIGHTYELLOW_EX
BLU = colorama.Fore.LIGHTBLUE_EX
colorama.init(autoreset=True)


pd.ExcelFile("week_4/ab_testing.xlsx").sheet_names

control = pd.read_excel("week_4/ab_testing.xlsx", sheet_name="Control Group")
test = pd.read_excel("week_4/ab_testing.xlsx", sheet_name="Test Group")


control.describe([0.01,0.99]).T
#             count       mean       std       min        1%       50%        99%        max
# Impression 40.000 101711.449 20302.158 45475.943 58072.042 99790.701 143105.791 147539.336
# Click      40.000   5100.657  1329.985  2189.753  2535.121  5001.221   7761.795   7959.125
# Purchase   40.000    550.894   134.108   267.029   285.771   531.206    790.188    801.795
# Earning    40.000   1908.568   302.918  1253.990  1267.764  1975.161   2481.309   2497.295

test.describe([0.01,0.99]).T
#             count       mean       std       min        1%        50%        99%        max
# Impression 40.000 120512.412 18807.449 79033.835 79112.255 119291.301 158245.264 158605.920
# Click      40.000   3967.550   923.095  1836.630  2005.234   3931.360   6012.877   6019.695
# Purchase   40.000    582.106   161.153   311.630   325.214    551.356    876.576    889.910
# Earning    40.000   2514.891   282.731  1939.611  1942.784   2544.666   3091.941   3171.490


control["Group"] = "control"
test["Group"] = "test"
df = pd.concat([control,test], ignore_index=True)
df
#     Impression    Click  Purchase  Earning    Group
# 0    82529.459 6090.077   665.211 2311.277  control
# 1    98050.452 3382.862   315.085 1742.807  control
# 2    82696.024 4167.966   458.084 1797.827  control
# 3   109914.400 4910.882   487.091 1696.229  control
# 4   108457.763 5987.656   441.034 1543.720  control
# ..         ...      ...       ...      ...      ...
# 75   79234.912 6002.214   382.047 2277.864     test
# 76  130702.239 3626.320   449.825 2530.841     test
# 77  116481.873 4702.782   472.454 2597.918     test
# 78   79033.835 4495.428   425.359 2595.858     test
# 79  102257.454 4800.068   521.311 2967.518     test
# [80 rows x 5 columns]

c,t = sms.DescrStatsW(df["Purchase"]).tconfint_mean()
print(f"{YEL} Confidence Interval: {c: .3f} - {t: .3f}")
# Confidence Interval:  533.533 -  599.467

#####################################################
# Defining the Hypothesis of A/B Testing
#####################################################

# Step 1: Define the hypothesis.

# QUE
# H₀: There is no statistically significant difference between the two groups.
# H₁: There is a statistically significant difference between the two groups.

# Step 2: Analyze purchase averages for the control and test groups

c,t = control['Purchase'].mean(),test['Purchase'].mean()

print(f"{YEL}MaximumBidding Group Purchase Average: {c:.3f}")
# MaximumBidding Group Purchase Average: 550.894
print(f"{YEL}AverageBidding Group Purchase Average: {t:.3f}")
# AverageBidding Group Purchase Average: 582.106


#####################################################
# Performing Hypothesis Testing
#####################################################


# Step 1: Perform assumption checks before hypothesis testing. These are Normality Assumption and Homogeneity of Variance.
# Test separately whether the control and test groups comply with the normality assumption using the Purchase variable.

# QUE
# H₀: Observations in the Control Group are Normally Distributed
# H₁: Observations in the Control Group Are Not Normally Distributed

w,p = shapiro(control['Purchase'])
print(f"{YEL} W: {w: .3f} \n p-Value: {p: .3f}")
#  W:  0.977
#  p-Value:  0.589

# QUE
# H₀: Observations in the Test Group are Normally Distributed
# H₁: Observations in the Test Group Are Not Normally Distributed

w,p = shapiro(test['Purchase'])
print(f"{YEL} W: {w: .3f} \n p-Value: {p: .3f}")
#  W:  0.959
#  p-Value:  0.154

control['Purchase'].plot(kind="kde")
test['Purchase'].plot(kind="kde")
plt.show()




distorted = control[['Purchase']].copy()
distorted.loc[distorted['Purchase'] < 800] = 0
w,p = shapiro(distorted['Purchase'])
print(f"{YEL} W: {w: .3f} \n p-Value: {p: .14f}")
#  W:  0.147
#  p-Value:  0.00000000000007

distorted['Purchase'].plot(kind="kde")
plt.show()

# TODO
# homogeneity (similarity) of variance

# QUE
# H₀: There is no significant difference between the variances of the Control and Test groups.
# H₁: There is a significant difference between the variances of the Control and Test groups.

w,p = levene(control['Purchase'],test['Purchase'])
print(f"{YEL}W: {w:.3f}\np-Value: {p: .3f}")
# W: 2.639
# p-Value:  0.108

# SO:
# Since the p-value value is 0.1082, the null hypothesis for variance similarity cannot be rejected. That is, no strong
# enough evidence has been obtained that there is a significant difference between the variances of the groups.
# (Variances are Similar)

sns.boxplot(y="Purchase", x="Group", data=df, showmeans=True, palette="pastel")
plt.show()
sns.histplot(x="Purchase", hue="Group", data=df, palette="colorblind")
plt.show()


levene(control['Purchase'], distorted['Purchase'])
distorted["Group"] = "distorted"
df2 = pd.concat([control, distorted], ignore_index=True)
sns.histplot(x="Purchase", hue="Group", data=df2, palette="colorblind")
plt.show()

# Step 2: Select the appropriate test according to the Normality Assumption and Variance Homogeneity results

# SO:
# Since the normal distribution hypothesis is not rejected, we need to use a parametic method.

# QUE
# H₀: There is no statistically significant difference between the two group means.
# H₁: There is a statistically significant difference between the two group means.

t, p, = ttest_ind(control['Purchase'], test['Purchase'], equal_var = True)
print(f"{YEL}\tt:  {t:.3f}\n\tp-Value: {p:.3f}")
# 	t:  -0.942
# 	p-Value: 0.349

# SO
# p-value was 0.349, which means there is no statistically significant difference between the two group averages,
# meaning H₀ hypothesis cannot be rejected.

##############################################################
# Analysis of Results
##############################################################
# SO:
# Since we could not reject the normality hypothesis, I used ttest_ind (independent two-sample T-Test),
# and since we could not reject the variance similarity, we entered the equal_var parameter as True.

# Since the variances were similar, we used "Student's t-test". If they were not equal,
# we would have used "Welch's t-test". (selects itself according to the equal_var parameter)


# SO
# No statistically significant difference was found between the Control and Test groups in terms of "Purchase".
# The client must base their ad bidding strategies on other determining factors such as Click or Earning.
def report(group1, group2, column,sig_level=0.05, Conf=False):
    """
    :param group1: Sample
    :param group2: Sample
    :param column: Feature
    :param sig_level: Significance Level
    :param Conf: Confidence interval
    :return: Report
    """
    group1,group2 = group1.copy(),group2.copy()
    g1_mean, g2_mean = group1[column].mean(), group2[column].mean()
    p_ind, p_mann, p_levene = 0,0,0
    if Conf:
        group1["Group"] = "control"
        group2["Group"] = "test"
        concated = pd.concat([group1, group2], ignore_index=True)
        low, up = sms.DescrStatsW(concated[column]).tconfint_mean()

    w_shapiro1, p_shapiro1 = shapiro(group1[column])
    w_shapiro2, p_shapiro2 = shapiro(group2[column])
    if (p_shapiro1 >= sig_level) & (p_shapiro2 >= sig_level) :
        w_levene, p_levene = levene(group1[column], group2[column])
        t_ind, p_ind, = ttest_ind(group1[column], group2[column], equal_var=(p_levene >= sig_level))
    else:
        t_mann, p_mann = mannwhitneyu(group1[column], group2[column])
    used = "ttest_ind(average)" if p_ind > 0 else "mannwhitneyu(rank)"
    if Conf:
        return (f"{YEL if p_ind > 0 else BLU }"
                f"*First Group* \nMean-> {g1_mean:.3f}, Normality-> {(p_shapiro1 >= sig_level)}\n"
                f"*Second Group* \nMean-> {g2_mean:.3f}, Normality-> {(p_shapiro2 >= sig_level)}\n"
                f"First Group - Second Group Variance Similarity: {(p_levene >= sig_level)}\n"
                f"Confidence Interval: {low:.3f} - {up:.3f}\n"
                f"Method Used: {used}\n"
                f"H₀: {(p_ind or p_mann) >= sig_level} \t H₁: {(p_ind or p_mann) <= sig_level}")
    else:
        return (f"{YEL if p_ind > 0 else BLU }"
                f"*First Group* \nMean-> {g1_mean:.3f}, Normality-> {(p_shapiro1 >= sig_level)}\n"
                f"*Second Group* \nMean-> {g2_mean:.3f}, Normality-> {(p_shapiro2 >= sig_level)}\n"
                f"First Group - Second Group Variance Similarity: {(p_levene >= sig_level)}\n"
                f"Method Used: {used}\n"
                f"H₀: {(p_ind or p_mann) >= sig_level} \t H₁: {(p_ind or p_mann) <= sig_level}")

print(report(control,test,column="Purchase",sig_level=0.05, Conf=False))
# *First Group*
# Mean-> 550.894, Normality-> True
# *Second Group*
# Mean-> 582.106, Normality-> True
# First Group - Second Group Variance Similarity: True
# Method Used: ttest_ind(average)
# H₀: True 	 H₁: False
