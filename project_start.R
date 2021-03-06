# Stat 852 project

library(tidyverse)
library(h2o)
library(devtools)

h2o.shutdown()
h2o.init(nthreads = -1, max_mem_size = '3G') # Default is 1.77 GB of memory, think before ensembling

# Import the data
train <- read_csv('Data2019.csv')

# Sample plot
# train %>%
#   ggpairs()

# Convert X1, X2, and X15 to factors
train <- train %>% 
  mutate(X1 = as.factor(X1),
         X2 = as.factor(X2),
         X15 = as.factor(X15))

# Do some brief outlier detection things tomorrow
# Distribution of Y is roughly Gaussian
# Some X values are correlated but nothing crazy, maybe consider some correlation reduction techniques

set.seed(46676352)
train$set <- ifelse(runif(nrow(train)) > 0.75, 1, 2)

train_h2o <- as.h2o(train %>% filter(set == 2) %>% select(-set))
test_h2o <- as.h2o(train %>% filter(set == 1) %>% select(-set))

y <- "Y"
x <- setdiff(names(train_h2o), y)

nfolds <- 5

# The MSPE values for this model and onwards are now correct. Baseline is (1.41) and
# the gbm_rf ensemble (1.339). Best ensemble is currently the simple combo of one gbm and one rf for
# a MSPE of 1.289 and the ensemble with lots of random forests and gbms for a MSPE of 1.283.
base_gbm <- h2o.gbm(x = x,
                  y = y,
                  training_frame = train_h2o,
                  distribution = "gaussian",
                  ntrees = 10,
                  max_depth = 3,
                  min_rows = 2,
                  learn_rate = 0.2,
                  nfolds = nfolds,
                  fold_assignment = "Modulo",
                  keep_cross_validation_predictions = TRUE,
                  seed = 1)

base_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = train_h2o,
                          ntrees = 50,
                          nfolds = nfolds,
                          fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE,
                          seed = 1)

# Train a stacked ensemble using the GBM and RF above
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train_h2o,
                                model_id = "my_ensemble_base", # Will automatically be named otherwise
                                base_models = list(base_gbm@model_id, base_rf@model_id))

# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test_h2o)

# This basic 2 model ensemble runs a MSPE of 1.289 and a RMSE of 1.133 on the test set.
# How much better is it than the individual models?
perf_gbm_test <- h2o.performance(base_gbm, newdata = test_h2o) # 1.4588
perf_rf_test <- h2o.performance(base_rf, newdata = test_h2o) # 1.291

# The ensemble here is actually a bit better, had data problems the first time

# Instead build a larger ensemble, more automated

learn_rate_opt <- c(0.01, 0.03, 0.1, 0.3)
max_depth_opt <- c(3, 5, 6, 9)
sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
col_sample_rate_opt <- c(0.7, 0.9)
hyper_params <- list(learn_rate = learn_rate_opt,
                     max_depth = max_depth_opt,
                     sample_rate = sample_rate_opt,
                     col_sample_rate = col_sample_rate_opt) #min_rows is n.minobsinnode if I want to tune

# Search_criteria will be used to explore the grid sapace defined above (when called in h2o.grid)
# Default is Cartesian strategy which will search all combos. If using RandomDiscrete, make sure
# to specify max_models and / or max_runtime_secs. These seem to be the only 2 options currently.

search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 10,
                        seed = 1)

#search_criteria <- list(strategy = "Cartesian")

# Automated grid fitting
gbm_grid <- h2o.grid(algorithm = "gbm",
                     grid_id = "gbm_grid_binomial",
                     x = x,
                     y = y,
                     training_frame = train_h2o,
                     ntrees = 25,
                     seed = 1,
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params, # List to search
                     search_criteria = search_criteria) # How to search

ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train_h2o,
                                model_id = "ensemble_gbm_grid_binomial",
                                base_models = gbm_grid@model_ids) # Specify model_ids to include

# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test_h2o) # With random discrete, n =10, get mspe = 1.369
# MSPE of 1.368 on Cartesian search

.getmse <- function(mm) h2o.mse(h2o.performance(h2o.getModel(mm), newdata = test_h2o))
baselearner_mses <- sapply(gbm_grid@model_ids, .getmse)
baselearner_best_auc_test <- min(baselearner_mses)

max_depth_opt <- c(3, 5, 6, 9)
mtries_opt <- c(3, 5, 7, 10)
nbins_opt <- c(5, 10, 20)
min_rows_opt <- c(1, 5, 10, 20)
ntrees_opt <- c(10, 25, 50, 100, 500)
hyper_params_rf <- list(mtries = mtries_opt, 
                        max_depth = max_depth_opt,
                        nbins = nbins_opt,
                        min_rows = min_rows_opt,
                        ntrees = ntrees_opt)

search_criteria <- list(strategy = "Cartesian")

rf_grid_test <- h2o.grid(algorithm = "randomForest",
                    grid_id = "rf_grid_small_test",
                    x = x,
                    y = y,
                    training_frame = train_h2o,
                    seed = 1,
                    nfolds = nfolds,
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE,
                    hyper_params = hyper_params_rf, # List to search
                    search_criteria = search_criteria)

rf_ensemble_test <- h2o.stackedEnsemble(x = x,
                    y = y,
                    training_frame = train_h2o,
                    validation_frame = test_h2o,
                    model_id = "ensemble_gbm_rf_small",
                    base_models = rf_grid_test@model_ids,
                    metalearner_algorithm = "AUTO",
                    metalearner_nfolds = 10)

perf <- h2o.performance(rf_ensemble_test, newdata = test_h2o) # 1.283-1.298 area MSPE, 1.13 RMSE
# Consistent on both desktop and laptop
# 1.265 now by using 10 trees and validation set, has variability

# Lets just explore the data a bit first
train_split <- train %>% filter(set == 2) %>% select(-set)
test_split <- train %>% filter(set == 1) %>% select(-set)

library(GGally)
train_split %>% 
  ggpairs()

cor(train_h2o) %>% View()
# X13 and X14 are highly correlated, X11 and X12 are highly correlated, X14 and X13 are moderately correlated
# with X10

# What does this plot look like after principle components?
# Need to use a different type of pca because of factor variables
library(FactoMineR)
pca <- PCA(train_split %>% select(-Y), quali.sup = c(1,2,15)) # Not sure how to use for rotation yet
summary(pca)

# Try another package called homals, should help with non-linear PCA
library(homals)
data(senate)
pca_round2 <- homals(as.data.frame(train_split %>% select(-Y)), active = TRUE,
                     level = c("nominal", "nominal",
                               rep("numerical", 12), "nominal"),
                     ndim = 15)

# I dont know what is going on, should not use
train_split_pca <- as.data.frame(cbind(train_split$Y, pca_round2$scoremat))
names(train_split_pca) <- names(train_split)

test_split_pca <- as.data.frame(cbind(test_split$Y, predict(pca, test_split %>% select(-Y))))
names(test_split_pca) <- names(test_split)

train_split_pca %>%
  ggpairs()
# x9 seems to generate the same distribution as y, I wonder if they are linked

base_lm <- lm(Y ~ X9, data = train_split)
summary(base_lm)

# What would a default multivariate linear regression generate for mspe?
baseline <- lm(Y ~ ., train_split)
mspe_baseline <- mean((test_split$Y - predict(baseline, test_split))^2)

pca_baseline <- lm(Y ~ ., train_split_pca)
mspe_pca_baseline <- mean((test_split_pca$Y - predict(pca_baseline, test_split_pca))^2)
# Baseline is 1.41, regardless of rotation, as expected

# My current basic ensemble only generates a MSPE of 1.50
# Can do better than this, hopefully

# Try a tuned version of a gbm or neural net
library(gbm)
library(nnet)
library(randomForest)
library(tictoc)
library(multidplyr)

# Set model specific params to explore
# GBM
params <- expand.grid(n.trees = c(200, 1000, 4000),
                           interaction.depth = c(1, 4, 7),
                           n.minobsinnode = c(1, 5, 10),
                           shrinkage = c(0.01, 0.1, 0.3),
                           bag.fraction = c(0.45, 0.65, 0.85))

# RF
params_rf <- expand.grid(ntree = c(100, 500, 2000, 4000),
                         mtry = c(3, 7, 10),
                         replace = c(TRUE, FALSE),
                         nodesize = c( 5, 10, 20))

# NNet
params_nnet <- expand.grid(size = c(1, 5, 10, 20, 40),
                           decay = c(0, 0.001, 0.1, 1, 2, 4, 6),
                           maxit = c(500, 1000, 2000))

# Set up parallel where appropriate
num_cores <- parallel::detectCores()
cluster <- create_cluster(num_cores)

# Copy over the libraries, functions and data needed
cluster %>%
  cluster_library("tidyverse") %>%
  cluster_library("randomForest") %>%
  cluster_library("nnet") 

# use this cluster for all parallel stuff
set_default_cluster(cluster)

# Define function to scale data for neural nets
rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}

set.seed(12344321)
res_gbm_loop <- tibble(fold = 1:5, data = list(0))
res_rf_loop <- tibble(fold = 1:5, data = list(0))
res_nnet_loop <- tibble(fold = 1:5, data = list(0))

# Results might not even be a fair comparison, forgot to change X1, X2, and X15 to categorical
# Retraining with the variables now categorical
tic()
for(i in 1:5){
  
  train$set <- ifelse(runif(nrow(train)) > 0.75, 1, 2)
  
  this_train <- train %>%
    filter(set == 2) %>%
    select(-set)
  
  # Important note: gbm uses the first n rows in train for error calculations
  # Naturally it is important to randomize order, especially here because data is ordered by lpsa I believe
  rand_ind <- sample.int(nrow(this_train), nrow(this_train))
  
  this_train <- this_train[rand_ind,]
  
  this_test <- train %>%
    filter(set == 1) %>%
    select(-set)
  
  tic()
  # GBM
  for(j in 1:nrow(params)){

    temp_model <- gbm(data=this_train, Y ~ ., distribution="gaussian",
                      n.trees=params$n.trees[j], interaction.depth=params$interaction.depth[j],
                      shrinkage=params$shrinkage[j],
                      bag.fraction=params$bag.fraction[j],
                      n.minobsinnode = params$n.minobsinnode[j],
                      cv.folds=5, n.cores = parallel::detectCores())

    params$tree_count[j] <- which.min(temp_model$cv.error)
    params$oob_mse[j] <- min(temp_model$cv.error)
    params$mspe[j] <- mean((this_test$Y - predict(temp_model, this_test,
                                                         n.trees = params$n.trees[j],
                                                         type = 'response'))^2)
  }
  toc()
  res_gbm_loop$data[i] <- list(params)


  # # RF
  tic()
  cluster %>%
    cluster_assign_value("this_train", this_train) %>%
    cluster_assign_value("this_test", this_test)

  rf_data <- params_rf %>%
    mutate(cluster_group = rep_len(1:num_cores, length.out = nrow(params_rf))) %>%
    partition(cluster_group, cluster = cluster) %>%
    mutate(rf_obj = pmap(list(ntree, mtry, replace, nodesize),
                          ~ randomForest(x = this_train %>% select(-Y),
                                         y = this_train$Y,
                                         ntree = ..1,
                                         mtry = ..2,
                                         replace = ..3,
                                         nodesize = ..4)),
           smse = map_dbl(rf_obj, ~ mean((this_train$Y - .$predicted)^2)),
           MSPE = map_dbl(rf_obj,
                           ~ mean((this_test$Y - predict(., this_test, type = 'response'))^2))) %>%
    collect() %>%
    select(-rf_obj)

  toc()
  res_rf_loop$data[i] <- list(rf_data)


  # NNet
  # Convert back to numeric for scaling and nnet stuff
  this_train <- this_train %>% mutate_all(as.numeric)
  this_test <- this_test %>% mutate_all(as.numeric)
  
  # Not sure what to do about factor level variables
  this_train_x <- rescale(this_train[, -1], this_train[, -1]) %>% mutate(Y = this_train$Y)
  this_test_x <- rescale(this_test[, -1], this_train[, -1]) %>% mutate(Y = this_test$Y)
  
  cluster %>%
    cluster_assign_value("this_train_x", this_train_x) %>%
    cluster_assign_value("this_test_x", this_test_x)
  
  tic()
  nnet_data <- params_nnet %>%
    mutate(cluster_group = rep_len(1:num_cores, length.out = dim(params_nnet)[1])) %>%
    partition(cluster_group, cluster = cluster) %>%
    mutate(nnet_obj = pmap(list(size, decay, maxit), ~nnet(Y ~ ., data = this_train_x,
                                                           decay = ..2, size = ..1,
                                                          linout = TRUE, trace = FALSE, maxit = ..3)),
           smse = map_dbl(nnet_obj, ~ .$value / nrow(this_train_x)),
           MSPE = map_dbl(nnet_obj, ~ mean((this_test_x$Y - predict(., this_test_x))^2))) %>%
    collect() %>%
    select(-nnet_obj)
  
  toc()
  res_nnet_loop$data[i] <- list(nnet_data) 
  
  
  # Measure how far I am
  print(i)
}
toc()

# Wow the RF and NNet train so fast, its just the gbm thats slow
# Probably could do a better job of parallelizing it for fast training time
res_gbm_loop %>% unnest() %>% write.csv('gbm_project_progress.csv')
res_rf_loop %>% unnest() %>% write.csv('rf_project_progress.csv')
res_nnet_loop %>% unnest() %>% write.csv('nnet_project_progress.csv')

# Check best model params for each
res_gbm_loop %>% unnest() %>% group_by(n.trees, interaction.depth,
                                       shrinkage, 
                                       bag.fraction,
                                       n.minobsinnode) %>%
  summarize(avg_oob = mean(oob_mse),
            avg_MSPE = mean(mspe)) %>%
  arrange(avg_oob) %>% ungroup() %>% slice(1)

# The best model for gbm is fit with 
#n.trees interaction.depth n.minobsinnode shrinkage bag.fraction avg_oob avg_MSPE
#<dbl>             <dbl>          <dbl>     <dbl>        <dbl>   <dbl>  
#  1000             4             1        0.01         0.85    1.44     1.27

res_rf_loop %>% unnest() %>% group_by(ntree, mtry, replace, nodesize) %>%
  summarize(avg_smse = mean(smse),
            avg_MSPE = mean(MSPE)) %>%
  ungroup() %>%
  arrange(avg_MSPE) %>%
  slice(1)

# The best model with random forests is fit with
#ntree  mtry replace nodesize avg_smse avg_MSPE
#<dbl> <dbl> <lgl>      <dbl>    <dbl>    <dbl>
# 4000    10 FALSE         20     1.45     1.27

res_nnet_loop %>% unnest() %>% group_by(size, decay, maxit) %>%
  summarize(avg_smse = mean(smse),
            avg_MSPE = mean(MSPE)) %>%
  ungroup() %>%
  arrange(avg_MSPE) %>%
  slice(1)

# The best model with neural nets is fit with
#size decay maxit avg_smse avg_MSPE
#<dbl> <dbl> <dbl>    <dbl>    <dbl>
#   10     1   2000     1.66     1.78

# The neural net looks a little weak to me, probably need to train it more extensively
# I forgot to scale the data, running again. Didn't really fix much. Might be because
# previous work fit 100 neural nets per combo and chose the best one.

rand_ind <- sample.int(nrow(train_split), nrow(train_split))
train_split <- train_split[rand_ind,]
full_gbm <- gbm(data = train_split, Y ~., distribution="gaussian", 
                n.trees=4000, interaction.depth=7,
                shrinkage=0.01, 
                bag.fraction=0.85,
                n.minobsinnode = 5,
                cv.folds=5, n.cores = parallel::detectCores())

# 1.496 MSPE, this is currently better than my ensemble
gbm_tuned_mspe <- mean((test_split$Y - predict(full_gbm, test_split,
                            n.trees = which.min(full_gbm$cv.error),
                            type = 'response'))^2)


# Try to diversify the ensemble, just using lots of gbm may leave it weak
h2o.shutdown()
h2o.init(nthreads = -1, max_mem_size = "6g") # Default is 1.77 GB of memory, think before ensembling

set.seed(46676352)
train$set <- ifelse(runif(nrow(train)) > 0.75, 1, 2)

train_h2o <- as.h2o(train %>% filter(set == 2) %>% select(-set))
test_h2o <- as.h2o(train %>% filter(set == 1) %>% select(-set))

y <- "Y"
x <- setdiff(names(train_h2o), y)

nfolds <- 5

# GBM
learn_rate_opt <- c(0.01, 0.1, 0.3)
max_depth_opt <- c(1, 3, 5, 6)
sample_rate_opt <- c(0.7, 0.8)
col_sample_rate_opt <- c(0.7, 0.9)
hyper_params_gbm <- list(learn_rate = learn_rate_opt,
                     max_depth = max_depth_opt,
                     sample_rate = sample_rate_opt,
                     col_sample_rate = col_sample_rate_opt) #min_rows is n.minobsinnode if I want to tune

# RF
max_depth_opt <- c(3, 5, 6, 9)
mtries_opt <- c(3, 5, 7, 10)
nbins_opt <- c(5, 10, 20)
min_rows_opt <- c(1, 5, 10, 20)
hyper_params_rf <- list(mtries = mtries_opt, 
                        max_depth = max_depth_opt,
                        nbins = nbins_opt,
                        min_rows = min_rows_opt)

# GLM
lambda_opt <- list(lambda = seq(0, 1, by = 0.1))

# Deep ANN
epochs_opt <- c(5, 10, 20)
activation_opt <- c("Tanh", "Rectifier")
rate_opt <- c(0.001, 0.005, 0.01, 0.1)
momentum_start_opt <- c(0, 0.5)
l1_opt <- c(0, 0.5, 1)
l2_opt <- c(0, 0.5, 1)
hyper_params_deep <- list(epochs = epochs_opt,
                          activation = activation_opt,
                          rate = rate_opt,
                          momentum_start = momentum_start_opt,
                          l1 = l1_opt,
                          l2 = l2_opt)

# Search_criteria will be used to explore the grid sapace defined above (when called in h2o.grid)
# Default is Cartesian strategy which will search all combos. If using RandomDiscrete, make sure
# to specify max_models and / or max_runtime_secs. These seem to be the only 2 options currently.

# search_criteria <- list(strategy = "RandomDiscrete",
#                         max_models = 10,
#                         seed = 1)

search_criteria <- list(strategy = "Cartesian")

# Automated grid fitting
tic()
gbm_grid <- h2o.grid(algorithm = "gbm",
                     grid_id = "gbm_grid_small",
                     x = x,
                     y = y,
                     training_frame = train_h2o,
                     seed = 1,
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params_gbm, # List to search
                     search_criteria = search_criteria) # How to search

rf_grid <- h2o.grid(algorithm = "randomForest",
                    grid_id = "rf_grid_small",
                    x = x,
                    y = y,
                    training_frame = train_h2o,
                    ntrees = 25,
                    seed = 1,
                    nfolds = nfolds,
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE,
                    hyper_params = hyper_params_rf, # List to search
                    search_criteria = search_criteria)

glm_grid <- h2o.grid(algorithm = "glm",
                     grid_id = "glm_grid_small",
                     x = x,
                     y = y,
                     training_fram = train_h2o,
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = lambda_opt,
                     search_criteria = search_criteria)

deep_grid <- h2o.grid(algorithm = "deeplearning",
                      grid_id = "deep_grid_small",
                      x = x,
                      y = y,
                      training_fram = train_h2o,
                      nfolds = nfolds,
                      fold_assignment = "Modulo",
                      keep_cross_validation_predictions = TRUE,
                      hyper_params = hyper_params_deep,
                      search_criteria = search_criteria)
toc() # 1400 seconds to train
tic()
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train_h2o,
                                validation_frame = test_h2o,
                                model_id = "ensemble_gbm_rf_small",
                                base_models = c(gbm_grid@model_ids, rf_grid@model_ids,
                                                glm_grid@model_ids, deep_grid@model_ids),
                                metalearner_algorithm = "AUTO",
                                metalearner_nfolds = 10) # Specify model_ids to include
toc() # 220 seconds to combine

# Lets see what the performance is like on this ensesmble, now that it has some rf as well
mspe_rf_gbm_ensemble <- h2o.performance(ensemble, newdata = test_h2o) # 20 seconds ish
# 1.29 MSPE, 1.13 RMSE Now, not bad
# Does not seem to offer any improvement. Investigate why.

# Before trying another ensemble, scale variables to 0-1 and then use PCA
scaled_train <- h2o.prcomp(train_h2o, x = x,
                           k = 10, impute_missing = TRUE)
View(scaled_train)

# Figure out this model or try to make manual scaling functions