# Lecture 13: Gradient Boosting


# 1.
# In the Prostate data, use the training set to fit boosted trees using gbm(). Play
# with the tuning parameters until you find a combination that seems to minimize some
# measure of test error without touching the test set. Note that you can set ntrees to
# be fairly large and let the internal CV choose the optimum value for a given set of
# tuning parameters, although this could mean nested resampling if you use bootstrap
# or CV to tune combinations of the other tuning parameters.1 You may instead prefer
# to set the number of trees as another parameter in the grid and tell gbm() not to do
# crossvalidation.

library(gbm)
library(tidyverse)
library(tictoc)

prostate <- read_csv("Prostate.csv") 

set.seed(120401002) 
prostate$set <- ifelse(runif(n=nrow(prostate))>0.5, yes=2, no=1)

prostate_train <- prostate %>%
  filter(set == 1) %>%
  select(-set, -train, -ID)

# Important note: gbm uses the first n rows in train for error calculations
# Naturally it is important to randomize order, especially here because data is ordered by lpsa I believe
rand_ind <- sample.int(nrow(prostate_train), nrow(prostate_train))

prostate_train <- prostate_train[rand_ind,]

prostate_test <- prostate %>%
  filter(set == 2) %>%
  select(-set, -train, -ID)

# First model, testing run times and what not
# This will train based on the bag fraction
tic()
basic_boost_prostate <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
                 n.trees=10000, interaction.depth=2, shrinkage=0.001, 
                 bag.fraction=0.8, cv.folds=5, n.cores = 8) # Will use num_cores later
toc()

# RMSE of best fit (best number of trees)
sqrt(min(basic_boost_prostate$cv.error))

# Plot shows that there is clearly no reason to keep going past the returned number of trees in these conditions
# Would want to adjust if the green line was still decreasing at or near n.trees
gbm.perf(basic_boost_prostate, method = 'cv')

# The above sample implementation is done with the formula interface
# If using a large number of variables the matrix notation, used in gbm.fit, is more efficient

# Before diving into actual cross validation and grid search, play around a bit with some simple implementaitons
# Fix other parameters, test on a few combos of a given param

# Int depth 1, 3, 5
tic()
int_1 <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
             n.trees=10000, interaction.depth=1, shrinkage=0.001, 
             bag.fraction=0.8, cv.folds=5, n.cores = 8)
sqrt(min(int_1$cv.error))

int_3 <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
             n.trees=10000, interaction.depth=3, shrinkage=0.001, 
             bag.fraction=0.8, cv.folds=5, n.cores = 8)
sqrt(min(int_3$cv.error))

int_5 <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
             n.trees=10000, interaction.depth=5, shrinkage=0.001, 
             bag.fraction=0.8, cv.folds=5, n.cores = 8)
sqrt(min(int_5$cv.error))
toc()

# Minimum at 5, continue with that value. Note it had a moderate, inconsistent effect on rmse

# Shrinkage 0.001, 0.01, 0.1, 0.3
tic()
shrink_001 <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
             n.trees=10000, interaction.depth=5, shrinkage=0.001, 
             bag.fraction=0.8, cv.folds=5, n.cores = 8)
sqrt(min(shrink_001$cv.error))

shrink_01 <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
                  n.trees=10000, interaction.depth=5, shrinkage=0.01, 
                  bag.fraction=0.8, cv.folds=5, n.cores = 8)
sqrt(min(shrink_01$cv.error))

shrink_1 <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
             n.trees=10000, interaction.depth=5, shrinkage=0.1, 
             bag.fraction=0.8, cv.folds=5, n.cores = 8)
sqrt(min(shrink_1$cv.error))

shrink_3 <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
             n.trees=10000, interaction.depth=5, shrinkage=0.3, 
             bag.fraction=0.8, cv.folds=5, n.cores = 8)
sqrt(min(shrink_3$cv.error))
toc()

# relatively large gain in cv.error as shrinkage increases, brings error lower than the interaction tuning
# Suggests to me this is more important than interaction depth for prostate data

# nminobsnode 2, 7, 12, 17 (weird scale because small data set)
tic()
nnode2 <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
                  n.trees=10000, interaction.depth=5, shrinkage=0.3, 
                  bag.fraction=0.8, cv.folds=5, n.cores = 8,
              n.minobsinnode = 2)
sqrt(min(nnode2$cv.error))

nnode7 <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
                 n.trees=10000, interaction.depth=5, shrinkage=0.3, 
                 bag.fraction=0.8, cv.folds=5, n.cores = 8,
              n.minobsinnode = 7)
sqrt(min(nnode7$cv.error))

nnode12 <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
                n.trees=10000, interaction.depth=5, shrinkage=0.3, 
                bag.fraction=0.8, cv.folds=5, n.cores = 8,
               n.minobsinnode = 12)
sqrt(min(nnode12$cv.error))

# Too big a nodesize for the given data set
# nnode17 <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
#                 n.trees=10000, interaction.depth=5, shrinkage=0.3, 
#                 bag.fraction=0.8, cv.folds=5, n.cores = 8,
#                n.minobsinnode = 17)
# sqrt(min(nnode17$cv.error))
toc()

# The lowest error is achieved by leaving the default in place. Can check larger spacing of this param.

# Now bag.fraction 0.65, 0.8, 1
tic()
bagfrac065 <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
              n.trees=10000, interaction.depth=5, shrinkage=0.3, 
              bag.fraction=0.65, cv.folds=5, n.cores = 8)
sqrt(min(bagfrac065$cv.error))



bagfrac08 <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
              n.trees=10000, interaction.depth=5, shrinkage=0.3, 
              bag.fraction=0.8, cv.folds=5, n.cores = 8)
sqrt(min(bagfrac08$cv.error))

bagfrac1 <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
               n.trees=10000, interaction.depth=5, shrinkage=0.3, 
               bag.fraction=1, cv.folds=5, n.cores = 8)
sqrt(min(bagfrac1$cv.error))
toc()

# Bag fraction seems variable, min occurs near 0.8, leave as is

# Priority: Shrinkage, interaction depth, (minnodes and bag frac)

# Perhaps EZtune will be useful
library(EZtune)
ez_gbm = eztune(x=prostate_train[,-9], y=prostate_train$lpsa, method="gbm", optimizer="hjn", fast=FALSE)

ez_gbm
# gbm(data=AQ, Ozone~Wind+Temp, distribution="gaussian", 
#     n.trees=EZ.gbm$n.trees, interaction.depth=EZ.gbm$interaction.depth, 
#     shrinkage=EZ.gbm$shrinkage, n.minobsinnode=EZ.gbm$n.minobsinnode)

# eztune(x, y, method, optimizer, fast)
params <- expand.grid(interaction_depth = list(c(1, 2, 4)),
                      shrinkage = c(0.001, 0.01, 0.05, 0.1, 0.3, 0.5), 
                      bag_fraction = list(c(0.75, 0.8, 0.85))) %>%
  unnest(.preserve = interaction_depth) %>% 
  unnest() %>%
  mutate(tree_count = 0,
         rmse = 0)

# Train numerous models, gather testing criteria
# Implement the lazy way

for(i in 1:nrow(params)){
  
  temp_model <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
      n.trees=10000, interaction.depth=params$interaction_depth[i],
      shrinkage=params$shrinkage[i], 
      bag.fraction=params$bag_fraction[i],
      cv.folds=5, n.cores = parallel::detectCores())
  
  params$tree_count[i] <- which.min(temp_model$cv.error)
  params$rmse[i] <- sqrt(min(temp_model$cv.error))
}

# Optimal parameters on this run were:
#  shrinkage bag_fraction interaction_depth tree_count      rmse
#     0.3          0.8                 2         15       0.7358375

# Since the data set is so small, try with a greatly expanded grid
# This should train reasonably fast as the current grid took only a few minutes
params_new <- expand.grid(interaction_depth = list(c(1:8)),
                          shrinkage = c(0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75), 
                          bag_fraction = list(c(0.7, 0.75, 0.8, 0.85, 0.9))) %>%
  unnest(.preserve = interaction_depth) %>% 
  unnest() %>%
  mutate(tree_count = 0,
         rmse = 0)

n_cores <- parallel::detectCores()
# This will take roughly 8 times longer than the first set
for(i in 1:nrow(params_new)){
  
  temp_model <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
                    n.trees=10000, interaction.depth=params_new$interaction_depth[i],
                    shrinkage=params_new$shrinkage[i], 
                    bag.fraction=params_new$bag_fraction[i],
                    cv.folds=5, n.cores = n_cores)
  
  params_new$tree_count[i] <- which.min(temp_model$cv.error)
  params_new$rmse[i] <- sqrt(min(temp_model$cv.error))
}

# The best parameters here were:
# shrinkage bag_fraction interaction_depth tree_count      rmse
#     0.01          0.9                 6        741    0.7146581

# a) Describe the process and talk a little about some of the decisions I made:
# Choice of test error, combinations tried, and why I did not choose other values

# The process here was a standard iterative approach with what should be a little bit of forethought. 
# This meant that I played around with some fixed values while varying other parameters to get an idea of 
# how important the parameters each mattered. I got a rough idea that the most important parameters to
# explore were shrinkage and interaction depth. This came from large changes in RMSE and 
# small minimum values. 

# I then built an actual cross validation process in which a larger grid was used. This grid incorporated
# more shrinkage and interaction depth terms than it would have otherwise. In training, the selected model
# was chosen for its minimum RMSE value, generated by measuring prediction errors on the out of sample fold.
# Note that I did not explicitly train on trees as I can get optimal number of trees as an output in the
# training process naturally.

# b) For the final model,

## i) Report the variable importance and all of the partial dependence plots. Comment/compare to past. 
opt_model <- gbm(data=prostate_train, lpsa ~ ., distribution="gaussian", 
                 n.trees=741, interaction.depth=6,
                 shrinkage=0.01, 
                 bag.fraction=0.9,
                 cv.folds=5, n.cores = n_cores)
gbm.perf(opt_model, method="cv" ) 
summary(opt_model)

# lcavol is overwhelmingly relevant relative to the other variables in the data set. This is consistent 
# with previous modeling as lcavol frequently comes out as a highly relevant predictor.

# Partial dependence plots
plot(opt_model, i.var = 1)
plot(opt_model, i.var = 2)
plot(opt_model, i.var = 3)
plot(opt_model, i.var = 4)
plot(opt_model, i.var = 5)
plot(opt_model, i.var = 6)
plot(opt_model, i.var = 7)
plot(opt_model, i.var = 8)

# Still need to discuss these a bit

# 2, Repeat exercise 1 with xgboost and relevant complement functions
library(xgboost)

# Fit a basic xgboost model, note that it is a list format and is a bit more tedious to work with
base_xg <- xgboost(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                  max_depth=3, eta=.3, subsample=1,
                  nrounds=20, objective="reg:linear", verbose = 0)

# Try some of the built in cross validation
base_xg_cv <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                    max_depth=3, eta=.3, subsample=1,
                    nrounds=20, objective="reg:linear", nfold=5, verbose = 0)

# Evaluation log spits out train and test rmse means and std
(cv_results <- base_xg_cv$evaluation_log) 
which.min(cv_results$test_rmse_mean)

# Play around with general importance of tuning parameters, as above

# Parameters to check: max_depth, eta, subsample, and nrounds

# First max_depth: 3, 6, 9, 12
xg_d3 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                     max_depth=3, eta=.3, subsample=1,
                     nrounds=200, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d3 <- xg_d3$evaluation_log
(min_cv_d3 <- cv_results_d3[which.min(cv_results_d3$test_rmse_mean), 4])

xg_d6 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                max_depth=6, eta=.3, subsample=1,
                nrounds=200, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d6 <- xg_d6$evaluation_log
(min_cv_d6 <- cv_results_d6[which.min(cv_results_d6$test_rmse_mean), 4])

xg_d9 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                max_depth=9, eta=.3, subsample=1,
                nrounds=200, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d9 <- xg_d9$evaluation_log
(min_cv_d9 <- cv_results_d9[which.min(cv_results_d9$test_rmse_mean), 4])

xg_d12 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                max_depth=12, eta=.3, subsample=1,
                nrounds=200, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d12 <- xg_d12$evaluation_log
(min_cv_d12 <- cv_results_d12[which.min(cv_results_d12$test_rmse_mean), 4])


# Now eta: 0.01, 0.1, 0.3, 0.7

xg_d3 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                max_depth=9, eta=.01, subsample=1,
                nrounds=200, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d3 <- xg_d3$evaluation_log
(min_cv_d3 <- cv_results_d3[which.min(cv_results_d3$test_rmse_mean), 4])

xg_d6 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                max_depth=9, eta=.1, subsample=1,
                nrounds=200, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d6 <- xg_d6$evaluation_log
(min_cv_d6 <- cv_results_d6[which.min(cv_results_d6$test_rmse_mean), 4])

xg_d9 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                max_depth=9, eta=.3, subsample=1,
                nrounds=200, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d9 <- xg_d9$evaluation_log
(min_cv_d9 <- cv_results_d9[which.min(cv_results_d9$test_rmse_mean), 4])

xg_d12 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                 max_depth=9, eta=.7, subsample=1,
                 nrounds=200, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d12 <- xg_d12$evaluation_log
(min_cv_d12 <- cv_results_d12[which.min(cv_results_d12$test_rmse_mean), 4])

# A low eta (learning rate) paired with only 20 rounds suggests a problem. Increase rounds first.
# Now that I have increased the nrounds parameter, I see more clear results in that the 
# larger learning rates pair well with more rounds (naturally). Continue with an eta of 0.3 as it
# generated the best value.

# Next is subsample: 0.5, 0.75, 0.9, 1
xg_d3 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                max_depth=9, eta=.3, subsample=0.5,
                nrounds=200, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d3 <- xg_d3$evaluation_log
(min_cv_d3 <- cv_results_d3[which.min(cv_results_d3$test_rmse_mean), 4])

xg_d6 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                max_depth=9, eta=.3, subsample=0.75,
                nrounds=200, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d6 <- xg_d6$evaluation_log
(min_cv_d6 <- cv_results_d6[which.min(cv_results_d6$test_rmse_mean), 4])

xg_d9 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                max_depth=9, eta=.3, subsample=0.9,
                nrounds=200, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d9 <- xg_d9$evaluation_log
(min_cv_d9 <- cv_results_d9[which.min(cv_results_d9$test_rmse_mean), 4])

xg_d12 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                 max_depth=9, eta=.3, subsample=1,
                 nrounds=200, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d12 <- xg_d12$evaluation_log
(min_cv_d12 <- cv_results_d12[which.min(cv_results_d12$test_rmse_mean), 4])

# INterestingly this model is greatly helped by reducing the subsample rate. This may not be consistent
# in future models / datasets. Keep note of that. Otherwise this is something worth tuning a bit.

# Now nrounds: 20, 200, 500, 1000

xg_d3 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                max_depth=9, eta=.3, subsample=0.5,
                nrounds=20, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d3 <- xg_d3$evaluation_log
(min_cv_d3 <- cv_results_d3[which.min(cv_results_d3$test_rmse_mean), 4])

xg_d6 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                max_depth=9, eta=.3, subsample=0.5,
                nrounds=200, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d6 <- xg_d6$evaluation_log
(min_cv_d6 <- cv_results_d6[which.min(cv_results_d6$test_rmse_mean), 4])

xg_d9 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                max_depth=9, eta=.3, subsample=0.5,
                nrounds=500, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d9 <- xg_d9$evaluation_log
(min_cv_d9 <- cv_results_d9[which.min(cv_results_d9$test_rmse_mean), 4])

xg_d12 <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                 max_depth=9, eta=.3, subsample=0.5,
                 nrounds=1000, objective="reg:linear", nfold=5, verbose = 0)

cv_results_d12 <- xg_d12$evaluation_log
(min_cv_d12 <- cv_results_d12[which.min(cv_results_d12$test_rmse_mean), 4])

# For some reason the best test rmse is in the nrounds = 20 with a subsample of 0.5. Tune carefully to
# investigate.


# Parameters to check: max_depth, eta, subsample, and nrounds
tic()
params_xg <- expand.grid(max_depth = list(c(3, 6, 12, 20, 30)),
                      eta = c(0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75), 
                      subsample = list(c(0.5, 0.6, 0.75, 0.9, 1)),
                      nrounds = list(c(20, 200, 500, 1000))) %>%
  unnest(.preserve = c(max_depth, subsample)) %>% 
  unnest(.preserve = subsample) %>% 
  unnest() %>%
  mutate(rmse = 0)

# Train numerous models, gather testing criteria
# Implement the lazy way

for(i in 1:nrow(params_xg)){
  
  # xgb
  temp_model <- xgb.cv(data=as.matrix(prostate_train[,-9]), label=prostate_train$lpsa, 
                       max_depth=params_xg$max_depth[i],
                       eta=params_xg$eta[i],
                       subsample=params_xg$subsample[i],
                       nrounds=params_xg$subsample[i],
                       objective="reg:linear", nfold=5, verbose = 0)
  
  eval_log <- temp_model$evaluation_log
  min_res_cv <- eval_log[which.min(eval_log$test_rmse_mean), 4]
  
  params_xg$rmse[i] <- min_res_cv$test_rmse_mean
}
toc()

# Wow that was fast, only 26.83 seconds
# Best model fit with parameters:

# eta nrounds max_depth subsample      rmse
# 0.75    20         3         1   0.9940256

# COntinue with the interpretation work after setting up the last part

# 3. Set up gbm with OOB and xgboost with 5 fold CV.
set.seed(890987665)
abalone <- read_csv("Abalone.csv")

# Remove outliers in height
abalone <- abalone %>%
  filter(Height > 0 & Height < 0.5) %>%
  mutate(Sex = as.factor(Sex))

num_folds <- 20
r_vec <- sample(1:num_folds, dim(abalone)[1], replace=TRUE)

params_xg <- expand.grid(max_depth = list(c(3, 6, 12, 20, 30)),
                         eta = c(0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75), 
                         subsample = list(c(0.5, 0.6, 0.75, 0.9, 1)),
                         nrounds = list(c(20, 200, 500, 1000))) %>%
  unnest(.preserve = c(max_depth, subsample)) %>% 
  unnest(.preserve = subsample) %>% 
  unnest() %>%
  mutate(rmse = 0,
         predict = 0)


# params_new <- expand.grid(interaction_depth = list(c(1:8)),
#                           shrinkage = c(0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75), 
#                           bag_fraction = list(c(0.7, 0.75, 0.8, 0.85, 0.9))) %>%
#   unnest(.preserve = interaction_depth) %>% 
#   unnest() %>%
#   mutate(tree_count = 0,
#          oob = 0,
#          predict = 0)

# Set up stuff for easy tune saving
res_gbm_noloop <- tibble()

n_cores <- parallel::detectCores()
# This will take roughly 8 times longer than the first set

abalone <- abalone %>%
  mutate(folds = r_vec)

# Store the resulting data frames as list objects in each corresponding frame
res_xgb <- tibble(iter = 1:20, data = list(0))
res_gbm <- tibble(iter = 1:20, data = list(0))

#Need to do some thinking about how to implement this appropriately
# Storage is the key
tic()
for(i in 1:num_folds){
  
  abalone_train_split <- abalone %>%
    dplyr::filter(folds != i) %>%
    dplyr::select(-folds)
  
  abalone_test_split <- abalone %>%
    filter(folds == i) %>%
    dplyr::select(-folds)
  
  # gbm
  ez_gbm <- eztune(x=abalone_train_split[,-9],
                   y=abalone_train_split$Rings, method="gbm", optimizer="hjn", fast=0.75)
  
  ez_gbm_fit <- gbm(data=abalone_train_split, Rings ~., distribution="gaussian", 
                    n.trees=ez_gbm$n.trees, interaction.depth=ez_gbm$interaction.depth, 
                    shrinkage=ez_gbm$shrinkage, n.minobsinnode=ez_gbm$n.minobsinnode,
                    cv.folds=5, bag.fraction = 0.5) # Can use smaller because already tuned
  
  results_iter <- tibble(fold = i,
                         tree_count = which.min(ez_gbm_fit$cv.error),
                         oob = sqrt(min(ez_gbm_fit$cv.error)),
                         rmspe = sqrt(mean(
                           (abalone_test_split$Rings - predict(ez_gbm_fit, abalone_test_split))^2)))
  
  res_gbm_noloop <- res_gbm_noloop %>%
    bind_rows(results_iter)
  
  # for(j in 1:nrow(params_new)){
  #   # gbm
  #   temp_model <- gbm(data= abalone_train_split, Rings ~ ., distribution="gaussian", 
  #                     n.trees=10000, interaction.depth=params_new$interaction_depth[j],
  #                     shrinkage=params_new$shrinkage[j], 
  #                     bag.fraction=params_new$bag_fraction[j],
  #                     cv.folds=5, n.cores = n_cores)
  #   
  #   params_new$tree_count[j] <- which.min(temp_model$cv.error)
  #   params_new$oob[j] <- sqrt(min(temp_model$cv.error))
  #   params_new$predict[j] <- sqrt(mean((abalone_test_split$Rings - predict(temp_model, abalone_test_split))^2))
  # }
  
  # Save the tibble to the data frame for gbm
  #res_gbm$data[i] <- params_new %>% nest()
  print("Done the gbm step")
  
  # Need to convert Sex to numeric because it is making my matrix character data
  abalone_train_split <- abalone_train_split %>%
    mutate(Sex = as.numeric(Sex))
  
  abalone_test_split <- abalone_test_split %>%
    mutate(Sex = as.numeric(Sex))
  
  for(k in 1:nrow(params_xg)){
    # xgb stuff
    temp_model <- xgb.cv(data=as.matrix(abalone_train_split[,-9]), label=abalone_train_split$Rings, 
                         max_depth=params_xg$max_depth[k],
                         eta=params_xg$eta[k],
                         subsample=params_xg$subsample[k],
                         nrounds=params_xg$subsample[k],
                         objective="reg:linear", nfold=5, verbose = 0)
    
    act_model <- xgboost(data=as.matrix(abalone_train_split[,-9]), label=abalone_train_split$Rings, 
                     max_depth=params_xg$max_depth[k],
                     eta=params_xg$eta[k],
                     subsample=params_xg$subsample[k],
                     nrounds=params_xg$subsample[k],
                     objective="reg:linear", verbose = 0)
    
    eval_log <- temp_model$evaluation_log
    min_res_cv <- eval_log[which.min(eval_log$test_rmse_mean), 4]
    
    params_xg$rmse[k] <- min_res_cv$test_rmse_mean
    
    # Need to add in an actual model, the xgb.cv object is not appropriate for predict()
    params_xg$predict[k] <- sqrt(mean(
      (abalone_test_split$Rings - predict(act_model, as.matrix(abalone_test_split[,-9])))^2))
  }
  
  # Save the data frame to the tibble for xgb
  res_xgb$data[i] <- list(params_xg)
  print(i)
}
toc()
# Write the two data frames to csv
# gbm
#res_gbm_noloop %>% write.csv('gbm_results.csv')
# backup_gbm <- res_gbm_noloop
# xgb
#res_xgb %>% unnest() %>% write.csv('xgb_results.csv')
# backup_xgb <- res_xgb

# Forgot to do the default settings for each split
res_default <- tibble(fold = 1:num_folds, rmspe_gbm = 0, rmspe_xgb = 0)
for(i in 1:num_folds){
  
  # gbm first
  abalone_train_split <- abalone %>%
    dplyr::filter(folds != i) %>%
    dplyr::select(-folds)
  
  abalone_test_split <- abalone %>%
    filter(folds == i) %>%
    dplyr::select(-folds)
  
  # Default is 100 trees, not being detected though
  gbm_fit <- gbm(data=abalone_train_split, Rings ~., distribution="gaussian", n.trees = 100)
  
  # Default param has no cv.folds so there is no oob calculation
  # res_default$oob_gbm[i] <- sqrt(min(gbm_fit$cv.error))
  
  # Not sure why I needed to manually add here, the gbm_fit object has an n.trees element
  res_default$rmspe_gbm[i] <- sqrt(mean(
    (abalone_test_split$Rings - predict(gbm_fit, abalone_test_split, n.trees = 100))^2))
  
  # xgb second
  # Need to convert Sex to numeric because it is making my matrix character data
  abalone_train_split <- abalone_train_split %>%
    mutate(Sex = as.numeric(Sex))
  
  abalone_test_split <- abalone_test_split %>%
    mutate(Sex = as.numeric(Sex))
  
  xgb_fit <- xgboost(data=as.matrix(abalone_train_split[,-9]), label=abalone_train_split$Rings, 
          nrounds = 20, objective="reg:linear", verbose = 0)
  
  # Fit criteria to report
  res_default$rmspe_xgb[i] <- sqrt(mean(
    (abalone_test_split$Rings - predict(xgb_fit, as.matrix(abalone_test_split[,-9])))^2))
}

# Now start the proper write up, at least in preparation