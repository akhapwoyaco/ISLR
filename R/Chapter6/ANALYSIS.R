# 6.5 Labs: Linear Models and Regularization Methods

## 6.5.1 Subset Selection Methods
### Best Subset Selection
library(ISLR2)
names(Hitters)
#
dim(Hitters)
sum(is.na(Hitters))
#
Hitters <- na.omit(Hitters)
dim(Hitters)
sum(is.na(Hitters))
#
library(leaps)
#
