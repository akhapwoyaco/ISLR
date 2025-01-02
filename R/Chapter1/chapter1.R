#
library(ISLR)
#
data("Wage")
#

##### FIGURE 1.1 ########################################################
## WAGES DATA
par(mfrow = c(1,3))
# plot 2 wage ~ age
plot(
  wage ~ age, data = Wage)
#add fitted regression line to scatterplot

# plot 2 wage ~ year
plot(
  wage ~ year, data = Wage)
abline(lm(wage ~ year, data = Wage), col = 'blue')

# plot 2 wage ~ education
# boxplot(
#   wage ~ education, data = Wage)
#
wage_edu_box <- boxplot(
  wage ~ education, data = Wage,
  xaxt = "n", border = "white", 
  col = c("blue","orange","green", "black", "lightblue"),
  boxwex = 0.3, medlwd = 1, whiskcol = "black", 
  staplecol = "black", outcol = "red", cex = 0.3, outpch=19, 
  main = "Wages by Education Levels")
axis(
  side = 1, at = 1:length(wage_edu_box$names), 
  labels = paste(
    # sub removes first occurence of space and gsub removes all
    # https://stackoverflow.com/questions/58196481/substitute-up-until-first-dash-with-regex?noredirect=1&lq=1
    sub(pattern = ".*? ", x = wage_edu_box$names, 
        replacement = ""),"\n(n=",wage_edu_box$n,")",sep=""), 
  mgp = c(3,2,0))
#
par(mfrow = c(1,1))
#
#
# FIGURE 1.2 ###################################################################
# STOCK MARKET DATA
data("Smarket")
#
head(Smarket)
#
par(mfrow = c(1,3))
lag1_direction_box <- boxplot(
  Lag1 ~ Direction, data = Smarket,
  xaxt = "n", border = "white", 
  col = c("blue","orange"),
  boxwex = 0.3, medlwd = 1, whiskcol = "black", 
  staplecol = "black", outcol = "red", cex = 0.3, outpch=19, 
  main = "Yesterday", ylab = "Percentage Change in S&P", 
  xlab = "Today's Direction")
axis(
  side = 1, at = 1:length(lag1_direction_box$names), 
  labels = paste(
    lag1_direction_box$names,"\n(n=",lag1_direction_box$n,")",sep=""), 
  mgp = c(3,2,0))
#
lag2_direction_box <- boxplot(
  Lag2 ~ Direction, data = Smarket,
  xaxt = "n", border = "white", 
  col = c("blue","orange"),
  boxwex = 0.3, medlwd = 1, whiskcol = "black", 
  staplecol = "black", outcol = "red", cex = 0.3, outpch=19, 
  main = "Two Days Previous", ylab = "Percentage Change in S&P", 
  xlab = "Today's Direction")
axis(
  side = 1, at = 1:length(lag2_direction_box$names), 
  labels = paste(
    lag2_direction_box$names,"\n(n=",lag2_direction_box$n,")",sep=""), 
  mgp = c(3,2,0))
#
lag3_direction_box <- boxplot(
  Lag3 ~ Direction, data = Smarket,
  xaxt = "n", border = "white", 
  col = c("blue","orange"),
  boxwex = 0.3, medlwd = 1, whiskcol = "black", 
  staplecol = "black", outcol = "red", cex = 0.3, outpch=19, 
  main = "Three Days Previous", ylab = "Percentage Change in S&P", 
  xlab = "Today's Direction")
axis(
  side = 1, at = 1:length(lag3_direction_box$names), 
  labels = paste(
    lag3_direction_box$names,"\n(n=",lag3_direction_box$n,")",sep=""), 
  mgp = c(3,2,0))
#
par(mfrow = c(1,1))
#
#
# FIGURE 1.3 ##################################################################
#
model1 = MASS::qda(
  Direction~., data = subset(Smarket, Year < 2005))
prediction_prob = predict(
  model1, newdata = subset(Smarket, Year >= 2005), 
  type = 'response')$posterior
prediction_prob
#
boxplot(
  prediction_prob, 
  ylab = "Predicted Probability", 
  xlab = "Today's Direction"
)
#

## Gene Expression Data

data("NCI60")
#
# head(NCI60)
#
nc160_pca = prcomp(NCI60$data, scale. = T, center = T)
