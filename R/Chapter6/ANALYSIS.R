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
regfit_full <- regsubsets(Salary~., data = Hitters)
summary(regfit_full)
#
regfit_full <- regsubsets(Salary~., data = Hitters, 
                          nvmax = 19)
reg_summary <- summary(regfit_full)
names(reg_summary)
reg_summary$rsq
#
par(mfrow = c(2,3))
plot(
  reg_summary$rss , xlab = "Number of Variables",
  ylab = "RSS", type = "l")
plot(
  reg_summary$adjr2 , xlab = "Number of Variables",
  ylab = "Adjusted RSq", type = "l")

which.max(reg_summary$adjr2)
plot(
  reg_summary$adjr2 , xlab = "Number of Variables",
  ylab = "Adjusted RSq", type = "l")
points(
  11, reg_summary$adjr2[11], col = "red", cex = 2,
  pch = 20)

plot(
  reg_summary$cp, xlab = "Number of Variables",
  ylab = "Cp", type = "l")
which.min(reg_summary$cp)
points(
  10, reg_summary$cp[10], col = "red", cex = 2,
  pch = 20)
which.min(reg_summary$bic)
plot(
  reg_summary$bic , xlab = "Number of Variables",
  ylab = "BIC", type = "l")
points(
  6, reg_summary$bic[6], col = "red", cex = 2,
  pch = 20)

par(mfrow = c(1,1))
#
#
par(mfrow = c(2,2))
plot(regfit_full , scale = "r2")
plot(regfit_full , scale = "adjr2")
plot(regfit_full , scale = "Cp")
plot(regfit_full , scale = "bic")
par(mfrow = c(1,1))
#
# Forward and Backward Stepwise Selection
#
regfit_fwd <- regsubsets(
  Salary~., data = Hitters, 
  nvmax = 19, method = "forward")
summary(regfit_fwd)
#
regfit_bwd <- regsubsets(
  Salary~., data = Hitters, 
  nvmax = 19, method = "backward")
#
summary(regfit_bwd)
#
set.seed(1)
train <- sample(
  c(TRUE , FALSE), nrow(Hitters),
  replace = TRUE)
test <- (!train)
#
regfit_best <- regsubsets(
  Salary~., data = Hitters[train,], 
  nvmax = 19)
#
test_mat <- model.matrix(
  Salary~., 
  data = Hitters[test , ])
#
#
val_errors <- rep(NA, 19)
for (i in 1:19) {
  coefi <- coef(regfit_best , id = i)
  pred <- test_mat[, names(coefi)] %*% coefi
  val_errors[i] <- mean((Hitters$Salary[test] - pred)^2)
}
#
val_errors
#
predict_regsubsets <- function(object , newdata , id, ...) {
  form <- as.formula(object$call [[2]])
  mat <- model.matrix(form , newdata)
  coefi <- coef(object , id = id)
  xvars <- names(coefi)
  mat[, xvars] %*% coefi
}
#
regfit_best <- regsubsets(
  Salary~., data = Hitters, 
  nvmax = 19)
#
k <- 10
n <- nrow(Hitters)
set.seed(1)
folds <- sample(rep(1:k, length = n))
cv_errors <- matrix(
  NA, k, 19,
  dimnames = list(NULL , paste (1:19)))
#
for (j in 1:k) {
  best_fit <- regsubsets(
    Salary ~ .,
    data = Hitters[folds != j, ],
    nvmax = 19)
  for (i in 1:19) {
    pred <- predict_regsubsets(best_fit , Hitters[folds == j, ], id = i)
    cv_errors[j, i] <- mean((Hitters$Salary[folds == j] - pred)^2)
  }
}
#
#
mean_cv_errors <- apply(cv_errors , 2, mean)
mean_cv_errors
par(mfrow = c(1, 1))
plot(mean_cv_errors , type = "b")
reg_best <- regsubsets(
  Salary ~ ., data = Hitters ,
  nvmax = 19)
coef(reg_best, 10)
#
#
# 6.5.2 Ridge regression and lasso
#
library(glmnet)
x <- model.matrix(Salary~., Hitters)[, -1]
y <- Hitters$Salary
#
# Ridge regression
#
library(glmnet)
grid <- 10^seq(10, -2, length = 100)
ridge_mod <- glmnet(x, y, alpha = 0, lambda = grid)
#
dim(coef(ridge_mod))
#
set.seed(1)
train <- sample(1:nrow(x), nrow(x) / 2)
test <- (-train)
y_test <- y[test]
#
ridge_mod <- glmnet(x[train , ], y[train], alpha = 0,
                    lambda = grid, thresh = 1e-12)
ridge_pred <- predict(ridge_mod , s = 4, newx = x[test , ])
mean((ridge_pred - y_test)^2)
#
ridge_pred <- predict(
  ridge_mod , s = 0, newx = x[test , ],
  exact = T, x = x[train , ], y = y[train])
mean((ridge_pred - y_test)^2)
lm(y~x, subset = train)
predict(
  ridge_mod , s = 0, exact = T, type = "coefficients",
          x = x[train, ], y = y[train])[1:20, ]
#
set.seed(1)
cv_out <- cv.glmnet(x[train , ], y[train], alpha = 0)
plot(cv_out)
bestlam <- cv_out$lambda.min
bestlam
#
ridge_pred <- predict(ridge_mod , s = bestlam ,
                          newx = x[test , ])
mean((ridge_pred - y_test)^2)
out <- glmnet(x, y, alpha = 0)
predict(out , type = "coefficients", s = bestlam)[1:20, ]


#



#







#
