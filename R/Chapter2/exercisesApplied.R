#
library(ISLR)
library(RColorBrewer)
# Q8 ###########################################################
# (a)
College <- read.csv("../../data/ALL CSV FILES - 2nd Edition/College.csv")
head(College)
# (b)
rownames(College) <- College[,1]
College <- College[,-1]
# (c)
# (c) i
num_cols = unlist(lapply(College, is.numeric), use.names = F)
summary(College[,num_cols])
# (c) ii
pairs(College[,num_cols[1:10]])
# (c) iii
boxplot(Outstate ~ Private, data = College)
grid()
# (c) iv
College$Elite <- "No"
College$Elite[College$Top10perc > 50] <- "Yes"
#
summary(College$Elite)
#
boxplot(Outstate ~ Elite, data = College)
# (c) v
par(mfrow = c(2,2))
hist(College$Grad.Rate, breaks = 5)
hist(College$Grad.Rate, breaks = 10)
hist(College$Grad.Rate, breaks = 15)
hist(College$Grad.Rate, breaks = 20)
par(mfrow = c(1,1))
# (c) vi
bar_plot = barplot(
  as.matrix(College[1:5,2:4]), beside = T,
  legend.text = rownames(College[1:5,]), col = brewer.pal(3, "Dark2")
)
#
# Q9 ###########################################################################
#
data("Auto")
#
str(Auto)
# Predictors
# Quantitative: 
names(Auto)[unlist(lapply(Auto, is.numeric), use.names = T)]
# Qualitative: 
names(Auto)[!unlist(lapply(Auto, is.numeric), use.names = T)]
# (b)
numeric_cols <- unlist(lapply(Auto, is.numeric), use.names = T)
numeric_cols
apply(Auto[,numeric_cols], MARGIN = 2, FUN = function(x) range(x))
# (c)
apply(Auto[,numeric_cols], MARGIN = 2, FUN = function(x) mean(x))
apply(Auto[,numeric_cols], MARGIN = 2, FUN = function(x) sd(x))
apply(Auto[,numeric_cols], MARGIN = 2, FUN = function(x) summary(x))
# (d)
apply(Auto[c(-10,-85),numeric_cols], MARGIN = 2, FUN = function(x) summary(x))
# (e)
pairs(Auto[, numeric_cols])
# (f)
# Acceleration
plot(mpg~acceleration, data = Auto)
abline(lm(mpg~acceleration, data = Auto))
#
# Q10 ##########################################################################
# (a)
library(ISLR2)
data("Boston")
#
dim(Boston)
# (b)
pairs(Boston)
# (c)
# relationship of crime to other variables not clearly defined via viz
# (d)
summary(Boston)
# (e)
colnames(Boston)
# (f)
# subset(Boston, chas == 1)
table(Boston$chas)
# 
median(Boston$ptratio)
# (g)
subset(Boston, medv == min(Boston$medv))
# high crim, per capita crime rate by town.
# low zn proportion of residential land zoned for lots over 25,000 sq.ft.
# high indus: proportion of non-retail business acres per town.
# no bounds to chas: Charles River  
# high nox: nitrogen oxides concentration (parts per 10 million).
# close to 1st qu rm:average number of rooms per dwelling.
# maximum  age: proportion of owner-occupied units built prior to 1940.
# above average dis: weighted mean of distances to five Boston employment centres
# max rad: index of accessibility to radial highways.
# 3rd Qu tax: full-value property-tax rate per $10,000.
# 3rd Qu ptratio: pupil-teacher ratio by town.
# Above 3rd Qu lstat: lower status of the population (percent).
# minimum medv: median value of owner-occupied homes in $1000s.
#
Boston[Boston$rm>7,] |> nrow()
Boston[Boston$rm>8,] |> nrow()
#




