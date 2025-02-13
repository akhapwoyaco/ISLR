#
library(ISLR2)
#
dim(Default)
par(mfrow = c(1,2))
plot(income~balance, col = default, data = Default)
par(mfrow = c(1,1))

par(mfcol = c(1,2))
boxplot(balance~default, data = Default)
boxplot(income~default, data = Default)
par(mfrow = c(1,1))
#
# 4.3 Logistic Regression
#
