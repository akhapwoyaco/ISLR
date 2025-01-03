#
library(ISLR)

#

## FIGURE 2.1 ##################################################################
#
Advertising = read.csv(
  # get Advertising data
  "./data/ALL CSV FILES - 2nd Edition/Advertising.csv", row.names = 1 
)
head(Advertising)
#
par(mfrow = c(1,3))

plot(sales~TV, data = Advertising)
abline(lm(sales~TV, data = Advertising), col = 'blue')

plot(sales~radio, data = Advertising)
abline(lm(sales~radio, data = Advertising), col = 'blue')

plot(sales~newspaper, data = Advertising)
abline(lm(sales~newspaper, data = Advertising), col = 'blue')

par(mfrow = c(1,1))
#
## FIGURE 2.2 ##################################################################
#
Income1 = read.csv(
  # get Advertising data
  "./data/ALL CSV FILES - 2nd Edition/Income1.csv", row.names = 1 
)
head(Income1)
#
par(mfrow = c(1,2))

plot(Income ~ Education, data = Income1)
abline(lm(Income ~ Education, data = Income1), col = 'blue')

plot(Income ~ Education, data = Income1)
poly_model = lm(Income ~ poly(Education, 3, raw=TRUE), data = Income1)
# https://r-graph-gallery.com/44-polynomial-curve-fitting.html
poly_predict <- predict( poly_model ) 
ix <- sort(Income1$Education,index.return=T)$ix
lines(Income1$Education[ix], poly_predict[ix], col=2, lwd=2 ) 
#
apply(
  cbind(#https://stackoverflow.com/questions/23494232/regression-line-to-data-points-how-to-create-vertical-lines
    Income1$Education, Income1$Education, Income1$Income, 
    predict(poly_model)),1,function(coords){lines(coords[1:2],coords[3:4])})

# I add the features of the model to the plot
coe_ff <- round(poly_model$coefficients , 2)
text(14, 70 , 
     paste(
       "Model : ",coe_ff[1] , " + " , 
       coe_ff[2] , "*x"  , "+" , coe_ff[3] , "*x^2" , "+" , coe_ff[4] , 
       "*x^3" , "\n\n" , "P-value adjusted = ", 
       round(summary(poly_model)$adj.r.squared,2)))
#
par(mfrow = c(1,1))
#
#
## FIGURE 2.3 ################################################################
#
Income2 = read.csv(
  # get Advertising data
  "./data/ALL CSV FILES - 2nd Edition/Income2.csv", row.names = 1 
)
head(Income2)
#
library(MASS)
#Bivariate kernel density estimation with automatic bandwidth
density3d <- kde2d(x = Income2[,1], y = Income2[,3], h = Income2[,3])
#Visualize the density in 3D space
persp(density3d)
#
#

## LABS
#
# basic
(x <- c(1,3,2,3,5))
(y <- c(1,4,3))
#
length(x)
length(y)
ls()
#
set.seed(1303)
x <- rnorm(50)
y <- x + rnorm(50, mean = 50, sd = .1)
#
mean(y); var(y); sqrt(var(y)); sd(y)
#
x <- seq(-pi,pi, length = 50)
y <- x
f <- outer(x, y, function(x, y) cos(y)/(1+x^2))
contour(x,y,f)
contour(x,y,f, nlevels = 45, add = T)
fa <- (f - t(f))/2
contour(x,y,fa, nlevels = 15)
#
#
image(x,y,fa)
persp(x, y, fa)
persp(x, y, fa, theta = 30)
persp(x, y, fa, theta = 30, phi = 20)
persp(x, y, fa, theta = 30, phi = 70)
persp(x, y, fa, theta = 30, phi = 40)
