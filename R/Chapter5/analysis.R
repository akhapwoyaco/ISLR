# 5.3

library(ISLR2)
set.seed(1)
train <- sample(392, 196)
#
lm_fit <- lm(mpg~horsepower , data = Auto , subset = train)
#
mean((Auto$mpg - predict(lm_fit , Auto))[-train]^2)
#
lm_fit2 <- lm(mpg~poly(horsepower , 2), data = Auto ,
              subset = train)
mean((Auto$mpg - predict(lm_fit2 , Auto))[-train]^2)
lm_fit3 <- lm(mpg~poly(horsepower , 3), data = Auto ,
              subset = train)
mean((Auto$mpg - predict(lm_fit3 , Auto))[-train]^2)
#
set.seed(2)
train <- sample(392, 196)
lm_fit <- lm(mpg ~ horsepower , data = Auto, subset = train)
mean((Auto$mpg - predict(lm_fit , Auto))[-train ]^2)

lm_fit2 <- lm(mpg~poly(horsepower , 2), data = Auto ,
              subset = train)
mean((Auto$mpg - predict(lm_fit2 , Auto))[-train ]^2)
lm_fit3 <- lm(mpg ~ poly(horsepower , 3), data = Auto ,
              subset = train)
mean((Auto$mpg - predict(lm_fit3 , Auto))[-train]^2)
#
#
# 5.3.2 Leave-One-Out Cross-Validation
#
glm_fit <- glm(mpg~horsepower, data = Auto)
coef(glm_fit)
#
lm_fit <- lm(mpg~horsepower, data = Auto)
coef(lm_fit)
#
library(boot)
glm_fit <- glm(mpg~horsepower , data = Auto)
cv_err <- cv.glm(Auto , glm_fit)
cv_err$delta
#
cv_error <- rep(0, 10)
for (i in 1:10) {
  glm_fit <- glm(mpg ~ poly(horsepower , i), data = Auto)
  cv_error[i] <- cv.glm(Auto , glm_fit)$delta[1]
}
cv_error
#
# 5.3.3 k-Fold Cross Validation
#
set.seed(17)
cv_error_10 <- rep(0, 10)
for (i in 1:10) {
  glm_fit <- glm(mpg ~ poly(horsepower , i), data = Auto)
  cv_error_10[i] <- cv.glm(Auto , glm_fit, K = 10)$delta[1]
}
cv_error_10
#
# 5.3.4
#
alpha_fn <- function(data , index) {
  X <- data$X[index]
  Y <- data$Y[index]
  (var(Y) - cov(X, Y)) / (var(X) + var(Y) - 2 * cov(X, Y))
}
alpha_fn(Portfolio, 1:100)
#
set.seed(7)
alpha_fn(Portfolio, sample(100, 100, replace = T))
#
boot(Portfolio, alpha_fn, R = 100000)
#
boot_fn <- function(data , index)
  coef(lm(mpg~horsepower , data = data , subset = index))
boot_fn(Auto , 1:392)
#
set.seed(1)
boot_fn(Auto, sample(392, 392, replace = T))
#
# boot_fn(Auto, sample(392, 392, replace = T))
#
boot(Auto, boot_fn, 1000)
#
summary(lm(mpg~horsepower , data = Auto))$coef
#
#
boot.fn <- function(data , index)
  coef(
    lm(mpg ~ horsepower + I(horsepower^2),
       data = data , subset = index)
  )
set.seed(1)
boot(Auto , boot.fn, 1000)

summary(
  lm(mpg~horsepower + I(horsepower^2), data = Auto)
)$coef
#

# Applied
## Q5
### a
q5a_model <- glm(default~income+balance, data = Default, 
                 family = binomial)
summary(q5a_model)
### b






















