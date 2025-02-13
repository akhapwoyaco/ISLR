# Linear Regression
#
library(ISLR2)
#
advertising = read.csv(
  file = 'Advertising.csv', row.names = 1)
head(advertising)
#
# Simple Linear Regression
### Estimating Coefficients
# Figure 3.1
plot(sales ~ TV, data = advertising, col = 'red')
slm_model = lm(sales ~ TV, data = advertising)
# https://r-graph-gallery.com/44-polynomial-curve-fitting.html
poly_predict <- predict( slm_model ) 
ix <- sort(advertising$TV,index.return=T)$ix
lines(advertising$TV[ix], poly_predict[ix], col=4, lwd=2 ) 
#
apply(
  cbind(#https://stackoverflow.com/questions/23494232/regression-line-to-data-points-how-to-create-vertical-lines
    advertising$TV, advertising$TV, advertising$sales, 
    predict(slm_model)),1,
  function(coords){
    lines(coords[1:2],coords[3:4], col = 3)
  })
# I add the features of the model to the plot
coe_ff <- round(slm_model$coefficients , 2)
text(55, 23 , 
     paste(
       "Model : ",coe_ff[1] , " + " , 
       coe_ff[2] , "*x"  , "\n\n" , "P-value adjusted = ", 
       round(summary(slm_model)$adj.r.squared,2)), cex = 0.6)
#

#
advert_2 = advertising[c('TV', 'sales')]
x_bar = mean(advert_2$TV)
y_bar = mean(advert_2$sales)
advert_2['x-x_bar'] = advert_2$TV - x_bar
advert_2['y-y_bar'] = advert_2$sales - y_bar
head(advert_2)
beta_1 = sum(advert_2$`x-x_bar` * advert_2$`y-y_bar`
)/ sum(advert_2$`x-x_bar`^2)
beta_0 = y_bar - beta_1 * x_bar
#
beta_1; beta_0
#
### Assessing the Accuracy of the Coefficient Estimates
#
RSS = sum((advert_2$sales - beta_0 - beta_1 * advert_2$TV)^2)
RSS
#
n = nrow(advert_2)
n
RSE = sqrt(RSS/(n - 2))
RSE
#
SE_beta_0 = sqrt(
  RSE * RSE * (
    (1/n)+((x_bar^2)/sum(advert_2$`x-x_bar`^2)))
)
SE_beta_1 = sqrt(RSE * RSE / sum(advert_2$`x-x_bar`^2))
#
SE_beta_0; SE_beta_1
#
c(beta_0 - 2*SE_beta_0, beta_0 + 2*SE_beta_0)
c(beta_1 - 2*SE_beta_1, beta_1 + 2*SE_beta_1)
#
t_beta_0 = (beta_0 - 0)/SE_beta_0
t_beta_1 = (beta_1 - 0)/SE_beta_1
#
t_beta_0; t_beta_1
#
TSS = sum(advert_2$`y-y_bar`^2)
R_SQUARED = 1 - (RSS/TSS)
R_SQUARED
# 
# Regression
#
lin_model <- function(formula, dat){
  
  # Ensure data is a dataframe
  if (class(dat) != "data.frame"){
    stop("Data must be input in the data.frame format")
  }
  
  # All character to factor and integers to numeric
  dat[sapply(dat, is.character)] <- lapply(dat[sapply(dat, is.character)], 
                                           as.factor)
  dat[sapply(dat, is.integer)] <- lapply(dat[sapply(dat, is.integer)], 
                                         as.numeric)
  
  # get all independent variables and response variable from formula
  ind_resp_var <- unique(all.vars(formula))
  if (ind_resp_var[2] == ".") {ind_resp_var <- c(ind_resp_var[1], colnames(dat)[colnames(dat) != ind_resp_var[1]])}
  
  # check that independent variables are either numeric or factor
  class_pred <- sapply(ind_resp_var[-1], function(x) {class(dat[, x])}) #get classes
  class_pred_unique <- unique(class_pred) #unique clases
  
  #comparison of class and checks
  if(!any(class_pred %in% c("factor", "numeric", "single", "double"))) {
    stop("Independent variables should be of numeric or factor types")
  }
  
  # model matrix x and y variable
  X <- model.matrix(formula,dat)
  y <- ind_resp_var[1] 
  #
  n_obs = nrow(X)
  p_vars = ncol(X)
  
  #
  # check that dependent variable is numeric
  if (!is.numeric(dat[, ind_resp_var[1]])){
    stop("Dependent variable should be numeric type")}###
  #
  cl <- match.call()
  #'Create class
  lnreg <- list()
  class(lnreg) <- "linmod"
  
  # Factor Elements
  factor_indep_vars <- names(class_pred[class_pred == "factor"])
  lnreg$flev <- lapply(factor_indep_vars, function(x) {levels(dat[, x])})
  names(lnreg$flev) <- factor_indep_vars
  #
  mu_y = mean(dat[,ind_resp_var[1]], na.rm = T)
  # y element
  lnreg$y <- dat[,ind_resp_var[1]]
  # y name
  lnreg$yname <- ind_resp_var[1]
  #'Decomposing 
  qr_x_m <- qr(X)
  # Get Regression coefficient
  lnreg$beta <- solve(qr.R(qr_x_m)) %*% t(qr.Q(qr_x_m)) %*% dat[,y]
  # Alternative below provide inefficient estimates to the data
  #lnreg$beta <- (qr.solve(qr.R(qr_x_m))*(qr.R(qr_x_m))) %*% t(qr.Q(qr_x_m)) %*% dat[,y]
  # fitted values
  lnreg$mu <- qr.fitted(qr_x_m,dat[,y])
  # residual values
  lnreg$residu <- qr.resid(qr_x_m,dat[,y])
  # Degrees of freedom
  lnreg$degfre <- nrow(dat)-ncol(X)
  
  # Gives residual variance abd regression coeff vars
  lnreg$sigma <- (t(lnreg$residu) %*% lnreg$residu) /lnreg$degfre
  #'converting into a scalar quantity
  resid_var_sc <- lnreg$sigma[1,1]
  regcoemat1 <- chol2inv(qr.R(qr_x_m))
  regcoemat2 <- vector(length = nrow(regcoemat1))
  for (i in 1:nrow(regcoemat1)) {
    regcoemat2[i] <- regcoemat1[i,i]
  }
  # variance of Regression coefficients
  lnreg$V <- regcoemat2 * resid_var_sc
  
  #'T-values for each coefficient
  tvalue <- vector(length = nrow(lnreg$beta))
  for (i in 1:nrow(lnreg$beta)) {
    tvalue[i] <- lnreg$beta[i,1]/sqrt(lnreg$V[i])
  }
  lnreg$tval <- tvalue
  #'P-values
  lnreg$pval <- pt(abs(lnreg$tval),lnreg$degfre, lower.tail = FALSE)
  lnreg$formula <- cl #model formula
  # model F value .... need to look into it
  lnreg$F_value = (sum((lnreg$y - mu_y)^2) -  sum(lnreg$residu^2) )/(sum(lnreg$residu^2)/(n_obs - p_vars - 1)) 
  return(lnreg)
}
#
# Advertising
model_TV_sales <- lin_model(formula = sales ~ TV,dat = advertising)
model_radio_sales <- lin_model(formula = sales ~ radio,dat = advertising)
model_newspaper_sales <- lin_model(formula = sales ~ newspaper,dat = advertising)
model_allpred_sales <- lin_model(formula = sales ~ TV+radio+newspaper,dat = advertising)
#
model_TV_sales$beta
model_radio_sales$beta
model_newspaper_sales$beta
model_allpred_sales$beta
#
predict.lin_model <- function(x, newdata){
  call_forme = regmatches(deparse(x$formula), regexec("=\\s*(.*?)\\s*," , deparse(x$formula)))[[1]][2]
  ind_resp_var <- unique(all.vars(eval(parse(text=call_forme))))
  #
  print(names(newdata))
  #
  if (ind_resp_var[1] %in% names(newdata)){
    newdata = newdata
  } else {
    y_name <- ind_resp_var[1]
    newdata[y_name] <- rep(0, nrow(newdata))
  }
  
  X <- model.matrix(eval(parse(text=call_forme)),newdata)
  predictions <- X %*% c(x$beta)
  predictions
}

predict.lin_model(model_allpred_sales, newdata = advertising)
predict.lin_model(model_TV_sales, newdata = advertising[,1, drop=F])


# Print Function
print.lin_model <- function(x){
  cat(regmatches(deparse(x$formula), regexec("=\\s*(.*?)\\s*," , deparse(x$formula)))[[1]][2])
  cat("\n\n") 
  if(length(x$beta[,1])){
    cat("Coefficients:\n")
    # the output has 2 columns, all are numeric
    stderror <- sqrt(x$V)
    estder <- cbind(x$beta,stderror)
    colnames(estder) <- c("Estimate", "Std. Error")
    print(estder)
  }
  else cat("No coefficients\n")
  invisible(x)
}

print.lin_model(model_allpred_sales)

# Plot Funtion
plot.lin_model <- function(x){
  data1 <- cbind(x$mu,x$residu)
  data1 <- as.data.frame(data1)
  names(data1) <- c("mu","residu")
  plot(x = data1$mu, y= data1$residu, 
       xlab = "Fitted values", ylab = "Residuals",
       main = "Residuals vs Fitted")
  abline(0, 0)                  # error
  grid()
}

plot.lin_model(model_allpred_sales)
#
# multiple reg lm
model_allpred_vs_sales <- lm(formula = sales ~ TV+radio+newspaper,dat = advertising)
summary(model_allpred_vs_sales)
#
cor(advertising)
#
pairs(Credit)
#
model_credit = lm(Balance~Own, data = Credit)
summary(model_credit)
#
lin_model(Balance~Own, dat = Credit)$beta
print.lin_model(lin_model(Balance~Own, dat = Credit))
lin_model(Balance~Region, dat = Credit)$beta
#
#
#
head(advertising)
#
summary(lm(
  sales~TV+radio+TV*radio, data = advertising
))
#
#
# 3.3.2 Extension of the Linear Model

# FIGURE 3.8
plot(mpg ~ horsepower, data = Auto, main = "Regression: mpg ~ horsepower, data = Auto")
# abline(lm(mpg ~ horsepower, data = Auto), col = 'blue')
for (i in 1:5){
  poly_model = lm(mpg ~ poly(horsepower, i, raw=TRUE), data = Auto)
  print(paste("Degree: ", i, ", R Squared: ",summary(poly_model)$r.squared))
  # https://r-graph-gallery.com/44-polynomial-curve-fitting.html
  poly_predict <- predict( poly_model ) 
  ix <- sort(Auto$horsepower,index.return=T)$ix
  lines(Auto$horsepower[ix], poly_predict[ix], col=i, lwd=2 ) 
}
#
legend(
  'topright', legend = paste0("Degree: ", 1:5),
  col = 1:5, #c("black", "red","blue","green","orange"),
  lty=1,lwd=2)
#
# # 3.3.3 Extension of the Linear Model
#
### 1 Non-linearity of data
# The presence of a pattern may indicate a problem with 
# ome aspect of the linear model.
# figure 3.9
# 
model_line = lm(mpg ~ horsepower, data = Auto)
model_quad = lm(mpg ~ poly(horsepower, 2), data = Auto)
par(mfrow = c(1,2))
plot(model_line, 1)
plot(model_quad, 1)
par(mfrow = c(1,1))
#
# 2. Correlation of error terms
# The standard errors that are computed for the estimated regression 
# coefficients or the fitted values are based on the assumption of 
# uncorrelated error terms. If in fact there is 
# correlation among the error terms, then the estimated standard 
# errors will tend to underestimate the true standard errors. 
# As a result, confidence and prediction intervals will be 
# narrower than they should be

# 3. Non-constant Variance of Error Terms
# One can identify non-constant variances in the errors, or 
# heteroscedasticity, from the presence of a funnel shape in
# otheresidual plot.

# 4. Outliers
# An outlier is a point for which yi is far from the value predicted 
# by the model. Outliers can arise for a variety of reasons, such as 
# incorrect recording of an observation during data collection.
# If we believe that an outlier has occurred due to an error in data 
# collection or recording, then one solution is to simply 
# remove the observation.

# 5. High Leverage Points
#
observation_leverage <- function(x){
  n = length(x)
  mean_x = mean(x)
  h = (1/n) + ((x-mean_x)^2)/sum((x-mean_x)^2)
}
#
par(mfrow = c(1,2))
plot(Age~Limit, data = Credit)
plot(Rating~Limit, data = Credit)
par(mfrow = c(1,1))
#
# 6. Collinearity
# Collinearity refers to the situation in which two or 
# more predictor variables are closely related to one another.
#
summary(lm(Balance~Age+Limit, data = Credit))
summary(lm(Balance~Rating+Limit, data = Credit))
# A simple way to detect collinearity is to look at the 
# correlation matrix of the predictors.

# it is possible for collinearity to exist between three or 
# more variables even if no pair of variables 
# has a particularly high correlation. We call this situation 
# multicollinearity. Instead of inspecting the correlation matrix,
# a better way to assess multi- collinearity 
# collinearity is to compute the variance inflation factor (VIF).
#

# 3.4 The Marketing Plan
#
summary(lm(sales ~ TV + radio+newspaper, data = advertising))

