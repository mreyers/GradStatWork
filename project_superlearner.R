# New project idea: Super learnersfrom a different package

#install.packages("remotes")
#remotes::install_github("jeremyrcoyle/sl3")

library(tidyverse)
library(sl3)
library(SuperLearner)
library(origami)
library(earth)
library(bartMachine)
train <- read_csv('Data2019.csv')

# Convert X1, X2, and X15 to factors
train <- train %>% 
  mutate(X1 = as.factor(X1),
         X2 = as.factor(X2),
         X15 = as.factor(X15))

# Do some brief outlier detection things tomorrow
# Distribution of Y is roughly Gaussian
# Some X values are correlated but nothing crazy, maybe consider some correlation reduction techniques

set.seed(46676352)

# Same naming conventions as h2o

train_split <- train %>% filter(set == 2) %>% select(-set)
test_split <- train %>% filter(set == 1) %>% select(-set)

y <- "Y"
x <- setdiff(names(train_split), y)


task_train <- sl3_Task$new(train_split, covariates = x, outcome = y)
task_test <- sl3_Task$new(test_split, covariates = x, outcome = y)


# Available models, stick to ones from in class
sl3_list_learners()

# Check to see if a subset of variables is sufficient for this task
SuperLearner::listWrappers("screen")
screen_cor <- Lrnr_pkg_SuperLearner_screener$new("screen.corP") #"screen.corP"
screen_fit <- screen_cor$train(task_train)
print(screen_fit)
# corRank only chooses the factor variables
# corP chooses every variable except for X8, MSPE 1.33
# randomForest excludes X2, X7, X9, X10, MSPE 1.459

# The screen suggests that X8 is not necessary for fitting
task_train <- screen_fit$chain()

# Set up learners and super learners to test performance
# Note these are all default learners and that I can add in parameters that each model naturally takes
# The models below now have some sample parameters which are not tuned, just exampels
glm_learner <- Lrnr_glm$new()
lasso_learner <- Lrnr_glmnet$new(alpha = 1, family = 'gaussian')
ridge_learner <- Lrnr_glmnet$new(alpha = 0, family = 'gaussian')
#mars_learner <- Lrnr_earth$new()
xgb_learner <- Lrnr_xgboost$new(eta = 0.1, max_depth = 7, subsample = 0.7, nrounds = 1000, verbose = 0)
gbm_learner <- Lrnr_gbm$new(distribution = 'gaussian', n.trees = 1000, interaction.depth = 5,
                            n.minobsinnode = 5, shrinkage = 0.01)
gam_learner <- Lrnr_gam$new()
rf_learner <- Lrnr_randomForest$new(mtry = 5, maxnodes = 7)
#h2o_grid_learner <- 

# Availabe superlearners: SuperLearner::listWrappers("SL")
# Test this behaviour with individual superlearners and tuning parameters instead of a huge model
SL.glmnet_learner_1 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.earth")
#SL.glmnet_learner_2 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.biglasso")
SL.glmnet_learner_3 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.gam")
SL.glmnet_learner_4 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.nnet")
SL.glmnet_learner_5 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.gbm")
SL.glmnet_learner_6 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.glm")
SL.glmnet_learner_7 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.randomForest",
                                                 ntree= 1000, nodesize = 10)
SL.glmnet_learner_8 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.xgboost",
                                                 nrounds = 500)





# Stack learners into a model 
learner_stack <- Stack$new(SL.glmnet_learner_1, #SL.glmnet_learner_2,
                           SL.glmnet_learner_3, SL.glmnet_learner_4,
                           SL.glmnet_learner_5, SL.glmnet_learner_6,
                           SL.glmnet_learner_7, SL.glmnet_learner_8,
                           glm_learner, #mars_learner,
                           xgb_learner,
                           gbm_learner, gam_learner, rf_learner)

# Train the stack
stack_fit <- learner_stack$train(task_train)

# Can eventually do pipelines, which will be useful with principle components being used as regressors
# Alternatively can PCA prior to training / testing

test_predict <- stack_fit$predict(task_test)
#test_predict

# How do the models inthe stack compare to one another?
cv_learners <- Lrnr_cv$new(learner_stack)
cv_fit <- cv_learners$train(task_train) # Cross validate using the data I used to train
cv_preds <- cv_fit$predict()

print(cv_fit$cv_risk(loss_squared_error))

# Combine these predictions sensibly, use a super learner
metalearner <- make_learner(Lrnr_nnls)
cv_task <- cv_fit$chain()
ml_fit <- metalearner$train(cv_task)

# Combination by piping, sends the results from stack_fit to ml_fit
sl_all_fit <- make_learner(Pipeline, stack_fit, ml_fit)
sl_preds <- sl_all_fit$predict()
sl_preds_test <- sl_all_fit$predict(task_test)

# Check sMSE value from the ensemble of un-tuned models 
sMSE <- mean((sl_preds - train_split$Y)^2) # 0.71

# Check MSPE value from the ensemble of un-tuned models
MSPE <- mean((sl_preds_test - test_split$Y)^2) # 1.320 with GLM superlearner, 1.322 with nnet superlearner
# 1.368 with random parameter values in each of the models entered
# 1.335 with random parameter values after screening out X8, has variability
# 1.346 with superlearner expanded, need to actually add tuning params
# 1.295 when running all models untuned with a larger collection

# With little effort and run time, the basic models above yielded an MSPE of 1.334
# Try to tune each of the models and use the tuned values in an ensemble pipeline
# This could yield relatively large improvements over current h2o work as it allows for access
# to a larger collection of base learners.
# Also consider different super learners

# Potentially could write a function to explore a larger number of model combinations
auto_stack <- function(eta = 0.1, max_depth = 7, subsample = 0.7, nrounds = 1000,
                       distribution = 'gaussian', n.trees = 1000, interaction.depth = 5,
                       n.minobsinnode = 5, shrinkage = 0.01, alpha = 1, mtry = 5){
  # Availabe superlearners: SuperLearner::listWrappers("SL")
  SL.learner_1 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.earth")
  SL.learner_2 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.glmnet", alpha = alpha)
  SL.learner_3 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.gam")
  SL.learner_4 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.nnet")
  SL.learner_5 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.gbm", 
                                            distribution = distribution, n.trees = n.trees, 
                                            interaction.depth = interaction.depth)
  SL.learner_6 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.glm")
  SL.learner_7 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.randomForest",
                                                   ntree= n.trees, nodesize = n.minobsinnode)
  SL.learner_8 <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.xgboost",
                                                   nrounds = nrounds, eta = eta,
                                            max_depth = max_depth, subsample = subsample,
                                            verbose = 0)
  
  # Additional models for continuity
  glm_learner <- Lrnr_glm$new()
  lasso_learner <- Lrnr_glmnet$new(alpha = 1, family = 'gaussian')
  ridge_learner <- Lrnr_glmnet$new(alpha = 0, family = 'gaussian')
  #mars_learner <- Lrnr_earth$new()
  xgb_learner <- Lrnr_xgboost$new(eta = 0.1, max_depth = 7, subsample = 0.7, nrounds = 1000, verbose = 0)
  gbm_learner <- Lrnr_gbm$new(distribution = 'gaussian', n.trees = 1000, interaction.depth = 5,
                              n.minobsinnode = 5, shrinkage = 0.01)
  gam_learner <- Lrnr_gam$new()
  rf_learner <- Lrnr_randomForest$new(mtry = 5, maxnodes = 7)
  
  learner_stack <- Stack$new(SL.learner_1, SL.learner_2,
                             SL.learner_3, SL.learner_4,
                             SL.learner_5, SL.learner_6,
                             SL.learner_7, SL.learner_8,
                             glm_learner, #mars_learner,
                             xgb_learner,
                             gbm_learner, gam_learner, rf_learner)
  
  # Train the stack
  stack_fit <- learner_stack$train(task_train)
  
  # Return the stack
  return(list(stack_fit, learner_stack))
}

stack_fit_base <- auto_stack()
stack_fit <- stack_fit_base[[1]]
learner_stack <- stack_fit_base[[2]]

# Build from here with more / smarter stacking, grids, and tuned values