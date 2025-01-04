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

#
#