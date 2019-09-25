# LASSO on prostate data using glmnet package 
#  (THERE IS ANOTHER PACKAGE THAT DOES LASSO.  WE WILL SEE IT LATER)
# Splitting the data in half and modeling each half separately.

prostate <-  read.table("Prostate.csv", header=TRUE, sep=",", na.strings=" ")
# head(prostate)

library(glmnet)
library(tidyverse)

# 1.
lazy_lasso <- function(seed){
  set.seed(seed) 
  prostate$set <- ifelse(runif(n=nrow(prostate))>0.5, yes=2, no=1)
  
  
  y.1 <- prostate[which(prostate$set==1),10]
  x.1 <- scale(as.matrix(prostate[which(prostate$set==1),c(2:9)]))

  y.2 <- prostate[which(prostate$set==2),10]
  x.2 <- scale(as.matrix(prostate[which(prostate$set==2),c(2:9)]))

  
  # Fit LASSO by glmnet(y=, x=). Gaussian is default, but other families are available  
  #  Function produces series of fits for many values of lambda.  
  
  # First half of data 
  lasso.1 <- glmnet(y=y.1, x= x.1, family="gaussian")
  
  # cv.glmnet() uses crossvalidation to estimate optimal lambda
  #  We haven't talked about this yet, so don't worry about it.
  
  cv.lasso.1 <- cv.glmnet(y=y.1, x= x.1, family="gaussian")
  
  # Predict both halves using first-half fit
  predict.1.1 <- predict(cv.lasso.1, newx=x.1, s="lambda.min")
  predict.1.2 <- predict(cv.lasso.1, newx=x.2, s="lambda.min")
  MSPE_lasso_1_min <- mean((y.2 - predict.1.2)^2)
  sMSE_lasso_1_min <- mean((y.1 - predict.1.1)^2)
  
  predict.1.1 <- predict(cv.lasso.1, newx=x.1)
  predict.1.2 <- predict(cv.lasso.1, newx=x.2)
  MSPE_lasso_1_1se <- mean((y.2 - predict.1.2)^2)
  sMSE_lasso_1_1se <- mean((y.1 - predict.1.1)^2)
  # Repeat for second half of data
  
  lasso.2 <- glmnet(y=y.2, x= x.2, family="gaussian")
  cv.lasso.2 <- cv.glmnet(y=y.2, x= x.2, family="gaussian")
  
  predict.2.1 <- predict(cv.lasso.2, newx=x.1, s="lambda.min")
  predict.2.2 <- predict(cv.lasso.2, newx=x.2, s="lambda.min")
  MSPE_lasso_2_min <- mean((y.1 - predict.2.1)^2)
  sMSE_lasso_2_min <- mean((y.2 - predict.2.2)^2)
  
  predict.2.1 <- predict(cv.lasso.2, newx=x.1)
  predict.2.2 <- predict(cv.lasso.2, newx=x.2)
  MSPE_lasso_2_1se <- mean((y.1 - predict.2.1)^2)
  sMSE_lasso_2_1se <- mean((y.2 - predict.2.2)^2)
  
  res <- tibble(Seed = rep(seed, 4), set = rep(c("1", "2"), each = 2), lambda = rep(c("min", "1SE"), 2), 
                lambda_val = c(cv.lasso.1$lambda.min, cv.lasso.1$lambda.1se,
                               cv.lasso.2$lambda.min, cv.lasso.2$lambda.1se),
                training_error = c(sMSE_lasso_1_min, sMSE_lasso_1_1se,
                                   sMSE_lasso_2_min, sMSE_lasso_2_1se),
                testing_error = c(MSPE_lasso_1_min, MSPE_lasso_1_1se,
                                  MSPE_lasso_2_min, MSPE_lasso_2_1se))
  return(res)
}

seed1 <- 120401002

seed1_res <- lazy_lasso(seed1)
seed1_res
# It seems that we are biased low for estimating sigma^2 when using LASSO to predict on the training data.


#2. Compare results with those obtained via all subsets and BIC methods in last lecture. Is there a clear
# preference for one method or another?

# Still need to compare
#3. Repeat the entire process and tables for LASSO using new seed. Are the results consistent with the
# first run? What about the comparison with all subsets and BIC methods from last lecture?
seed2 <- 9267926
seed2_res <- lazy_lasso(seed2)
seed2_res
# Running with a new seed, there are some notable discrepancies. The new seed generates larger testing
# errors across the board. Further, the chosen lambda values are inconsistent. Seed 2 generates lambda values
# for set 2 that are larger than even the 1SE lambda for set 1. This is not the case in seed 1 where the 
# lambda values for minimum and 1se are practically the same.

library(relaxo)
set.seed(120401002)
# NEED to scale variables, or it doesn't work right.
#  Need to scale both halves of the data according to the training set 
#  in order to do prediction properly later

# Function below scales values in a single object according to its own min and max.
#   Can apply it to training data, but test set needs to be scaled

rescale_2 <- function(x1,x2){scale(x1, center = apply(x2, 2, mean), scale = apply(x2, 2, sd)) 
}

lazy_relaxo <- function(seed){
  set.seed(seed)
  
  prostate$set <- ifelse(runif(n=nrow(prostate))>0.5, yes=2, no=1)
  y.1 <- prostate[which(prostate$set==1),10]
  x.1 <- scale(as.matrix(prostate[which(prostate$set==1),c(2:9)]))
  
  y.2 <- prostate[which(prostate$set==2),10]
  x.2 <- scale(as.matrix(prostate[which(prostate$set==2),c(2:9)]))
  
  y.1s1 <- scale(y.1, mean(y.1), sd(y.1))
  y.2s2 <- scale(y.2, mean(y.2), sd(y.2))
  x.1s1 <- rescale_2(x.1, x.1)
  x.2s1 <- rescale_2(x.2, x.1)
  x.2s2 <- rescale_2(x.2, x.2)
  x.1s2 <- rescale_2(x.1, x.2)
  
  # Basic Relaxed Lasso, using default "fast" algorithm that just computes 
  #  convex combination of LASSO and OLS. 
  relaxo.1 <- relaxo(Y=y.1s1, X= x.1s1)
  relaxo.2 <- relaxo(Y=y.2s2, X= x.2s2)
  #Look atresults over values of lambda and phi
  
  
  # Use crossvalidation to select "optimal" lambda and phi
  cv.relaxo.1 <- cvrelaxo(Y=y.1s1, X= x.1s1)
  
  # Get predicted values and (important!) rescale them to original Y scale
  predrel.1.1 <- predict(relaxo.1, newX=x.1s1, lambda=cv.relaxo.1$lambda, phi=cv.relaxo.1$phi)
  predrely.1.1 <- predrel.1.1*sd(y.1) + mean(y.1)
  sMSE.relaxo.1 <- mean((y.1 - predrely.1.1)^2)
  
  # Predict other half and compute MSPE
  predrel.1.2 <- predict(relaxo.1, newX=x.2s1, lambda=cv.relaxo.1$lambda, phi=cv.relaxo.1$phi)
  predrely.1.2 <- predrel.1.2*sd(y.1) + mean(y.1)
  MSPE.relaxo.1 <- mean((y.2 - predrely.1.2)^2)
  
  cv.relaxo.2 <- cvrelaxo(Y=y.2s2, X= x.2s2)
  
  predrel.2.1 <- predict(relaxo.2, newX=x.1s2, lambda=cv.relaxo.2$lambda, phi=cv.relaxo.2$phi)
  predrely.2.1 <- predrel.2.1 *sd(y.2) + mean(y.2)
  MSPE.relaxo.2 <- mean((y.1 - predrely.2.1)^2)
  
  predrel.2.2 <- predict(relaxo.2, newX=x.2s2, lambda=cv.relaxo.2$lambda, phi=cv.relaxo.2$phi)
  predrely.2.2 <- predrel.2.2 * sd(y.2) + mean(y.2)
  sMSE.relaxo.2 <- mean((y.2 - predrely.2.2)^2)
  
  tibble(seed = rep(seed, 2), set = c(1,2),
         lambda = c(cv.relaxo.1$lambda, cv.relaxo.2$lambda),
         phi = c(cv.relaxo.1$phi, cv.relaxo.2$phi),
         sMSE = c(sMSE.relaxo.1, sMSE.relaxo.2),
         MSPE = c(MSPE.relaxo.1, MSPE.relaxo.2))
}

relaxo_res <- bind_rows(lazy_relaxo(seed1), lazy_relaxo(seed2))
# Comparing this with previous results, it seems that relaxo more readily allows us to select larger
# lambda values. This is because we are now just selecting what variables to include in the model
# and then using phi to control the estimation of the parameters for these variables.


# Revisit, result feels funky
# 5. Apply LASSO to the Abalone data using the same training test split as before. Use both the 
# lambda that produces the minimum CV error and the +1SE version to do the following
# a) Identify the variables that each LASSO is selecting and compare them to what all-subsets/BIC chose
set.seed(29003092)
abalone <- read_csv("Abalone.csv", col_types = cols())

abalone <- abalone %>% 
  mutate(male_dummy = Sex == 1,
         female_dummy = Sex == 2) %>%
  select(-Sex) %>%
  select(Rings, everything())

abalone <- abalone %>%
  filter(Height > 0 & Height < 0.5)
abalone <- abalone %>%
  mutate(train = ifelse(runif(nrow(.)) > 0.75, "test", "train"))

abalone_train <- abalone %>% filter(train %in% "train") %>% dplyr::select(-train) %>% as.data.frame()
abalone_test <- abalone %>% filter(train %in% "test") %>% dplyr::select(-train) %>% as.data.frame()

abalone_train_rescale <- rescale_2(abalone_train %>% as.matrix, abalone_train %>% as.matrix)


cv.lasso.abalone <- cv.glmnet(y=abalone_train_rescale[,1], x= abalone_train_rescale[,-1],
                             family="gaussian")
lasso.abalone.min <- glmnet(y=abalone_train_rescale[,1], x= abalone_train_rescale[,-1],
                            family="gaussian", lambda = cv.lasso.abalone$lambda.min)
lasso.abalone.1se <- glmnet(y=abalone_train_rescale[,1], x= abalone_train_rescale[,-1],
                            family="gaussian", lambda = cv.lasso.abalone$lambda.1se)

# b) Compute the sample sMSE and MSPE and compare them to each other and to what all-subsets/BIC chose.
# Which method seems to be better at producing a prediction model?
abalone_test_rescale <- rescale_2(abalone_test %>% as.matrix, abalone_train %>% as.matrix)

predictions.min <- predict(lasso.abalone.min, newx = abalone_test_rescale[,-1])
scaled.preds.min <- predictions.min * sd(abalone_test_rescale[,1]) + mean(abalone_test_rescale[,1])
min.MSPE <- mean((abalone_test_rescale[,1] - scaled.preds.min)^2)

predictions.min.self <- predict(lasso.abalone.min, newx = abalone_train_rescale[,-1])
scaled.preds.min.self <- predictions.min.self * sd(abalone_train_rescale[,1]) + mean(abalone_train_rescale[,1])
min.sMSE <- mean((abalone_train_rescale[,1] - scaled.preds.min.self)^2)

predictions.1se <- predict(lasso.abalone.1se, newx = abalone_test_rescale[,-1])
scaled.preds.1se <- predictions.1se * sd(abalone_test_rescale[,1]) + mean(abalone_test_rescale[,1])
min.MSPE <- mean((abalone_test_rescale[,1] - scaled.preds.1se)^2)

predictions.1se.self <- predict(lasso.abalone.1se, newx = abalone_train_rescale[,-1])
scaled.preds.1se.self <- predictions.1se.self * sd(abalone_train_rescale[,1]) + mean(abalone_train_rescale[,1])
min.sMSE <- mean((abalone_train_rescale[,1] - scaled.preds.1se.self)^2)

# The prediction errors are roughly an order of magnitude smaller than those generated by all subsets regression
# and stepwise selection. I would prefer to use a LASSO model for prediction based on this fact.

# 6.

R =20
set.seed (890987665)
sMSE <- matrix( NA , nrow =R , ncol =5)
colnames( sMSE ) <- c("OLS " , "ALLBIC " , "LASSOMIN " ,
                        "LASSO1SE " , "RELAX ")
MSPE <- matrix( NA , nrow =R , ncol =5)
colnames( MSPE ) <- colnames( sMSE )

library(leaps)
for( r in 1:R ){
  new <- ifelse(runif( n = nrow( abalone )) <.75 , yes =1 , no =2)
  y.1n <- abalone[ which( new ==1) , 1]
  x.1n <- as.matrix( abalone[ which( new ==1) , -1])
  y.2n <- abalone[ which( new ==2) , 1]
  x.2n <- as.matrix( abalone[ which( new ==2) , -1])
  print(dim(x.1n))
  print(dim(y.1n))
  print(dim(x.2n))
  print(dim(y.2n))
  # Lin Reg via lsfit
  olsn <- lm( y.1n ~ x.1n )
  sMSE[ r ,1] <- mean(( y.1n - cbind(1 , x.1n ) %*% olsn$coef )^2)
  MSPE[ r ,1] <- mean(( y.2n - cbind(1 , x.2n ) %*% olsn$coef )^2)

  # BIC model selection via all subsets regression
  bic_mod <- regsubsets(y.1n ~ x.1n)
  sMSE[ r ,2] <- mean(( y.1n - cbind(1 , x.1n ) %*% bic_mod$coef )^2)
  MSPE[ r ,2] <- mean(( y.2n - cbind(1 , x.2n ) %*% bic_mod$coef )^2)
}
# (a) Make a table of the mean and 95%