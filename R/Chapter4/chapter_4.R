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
table(Default$default)

## 4.3.1 The logistic model
default_balance_model <- glm(
  formula = default~balance, family = binomial(),
  data = Default)
summary(default_balance_model)
predict(
  default_balance_model, newdata = data.frame(balance = c(1000, 2000)), 
  type = "response")
#
default_student_model <- glm(
  formula = default~student, family = binomial(),
  data = Default)
summary(default_student_model)
predict(default_student_model, 
        newdata = data.frame(student = c("Yes","No")), 
        type = "response")
#
## 4.3.4
#
multiple_model <- glm(
  formula = default~balance+income+student, family = binomial(),
  data = Default)
summary(multiple_model)
#
pred_bal = predict(
  default_balance_model, newdata = data.frame(balance = c(min(Default$balance):max(Default$balance))), 
  type = "response")
pred_bal
#

balance_data = data.frame(
  balance = c(min(Default$balance):max(Default$balance))
) |> cbind(pred_bal)
head(balance_data)
#
plot(pred_bal~balance, data = balance_data)
#
# 4.3.5 
#

## 4.4 Generative Models for Classification
#


# 4.6.1 Linear Regression
Bikeshare |> head()
#
summary(lm(bikers~mnth+hr+workingday+temp+weathersit, data = Bikeshare))
#
plot(bikers~hr, data = Bikeshare)
plot(log(bikers)~hr, data = Bikeshare)
#
plot(x = 1:10, y = 1:10)
#

plot(y = Bikeshare$bikers, x = as.numeric(Bikeshare$hr), col = 'blue')
plot(y = log(Bikeshare$bikers), x = as.numeric(Bikeshare$hr), col = 'blue')
#
# 4.6.2 Poisson regression on bike share data
#










