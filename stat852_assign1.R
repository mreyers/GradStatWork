# Matthew Reyers Homework 1 Question 1 
# Actually, was a good exercise in Tidyverse but just use his sample code here
# Actual submission will be in the Reg Bias and Var sim file

# Generating independent normal observations, n=20, p=19, sigma=6.

# Simulation of regression to compare linear models with different
#   numbers of variables.
# Model is E(Y) = 0 + 1 X1 + 1 X2 + e   with e~N(0,1)
# Three variables are measured: x1,x2,x3.  All are generated independent U(0,1)

# We fit 3 models: 
#   1: X1   2: X1, X2    3: X1, X2, X3
# We then compute predicted values at a small grid of points:
#  (.1,.3,.5,.7,.9)^3
# We compute the bias, variance, and MSE of the prediction at each X-combo
# We then summarize these results in plots and with summary statistics
# We also compute the model-based estimate of variance and estimate
#   its expectation as the average across simulations 


library(tidyverse)
set.seed(392039853)

reps <- 200 # Number of data sets
N <- 20      # Sample size

# Create test data
test <- expand.grid(x1 = c(.1,.3,.5,.7,.9), x2 = c(.1,.3,.5,.7,.9), x3=c(.1,.3,.5,.7,.9)) %>%
  as_tibble()
mu <- expand.grid(x1 = c(.1,.3,.5,.7,.9), x2 = c(.1,.3,.5,.7,.9), x3=c(.1,.3,.5,.7,.9)) %>%
  mutate(mu = x1 + x2) %>%
  pull(mu)

# Prepare for looping over reps
counter <- 1
# Matrix to save predictions: rows are replicates, 
#   columns are different X combinations times 3 (one for each model)
save.pred <- matrix(data=NA, ncol=3*nrow(test), nrow=reps)
# Matrix to save estimates of sigma^2
#   Rows are replicates, columns are different models 
save.sig <- matrix(data=NA, ncol=3, nrow=reps)

# Loop to generate data, analyze, and save results
data_set <- tibble(reps = 1:reps, N = N) %>%
  mutate(x1 = map(N, ~runif(n=.)),
         x2 = map(N, ~runif(n=.)),
         x3 = map(N, ~runif(n=.)),
         ep = map(N, ~rnorm(n=.)),
         y = pmap(list(x1, x2, ep), ~ ..1 + ..2 + ..3)
  ) %>%
  select(-ep)

reg_function <- function(data){
  lm(y ~ ., data = data)
}

data_set_2 <- data_set %>%
  mutate(data = pmap(list(x1, x2, x3, y), ~tibble( y = ..4, x1 = ..1, x2 = ..2, x3 = ..4)),
         number_predictors = list(1:3)) %>% # Fixes the naming, specifies eventual model size
  select(-x1, -x2, -x3, -y) %>% # Removes the old variables
  unnest(.preserve = data) %>% # Dont want to expand data
  mutate(to_model = map2(data, number_predictors, ~ select(.x, 1:(.y + 1))),
         model = map(to_model, reg_function))

pred_function <- function(model){
  predictions <- predict(model, newdata = test) %>% 
    as.data.frame() %>%
    mutate(id = 1) %>% 
    nest(-id) %>%
    pull(.)
}

# Generate predictions
data_set_preds <- data_set_2 %>%
  rowwise() %>%
  mutate(resid = sum(resid(model)^2),
         df = model$df.residual,
         sig1 = sum(resid(model)^2) / model$df.residual,
         predictions = pred_function(model)) # Figure out the problem here

# Estimate bias, variance, and MSE of predictions at each X-combo
data_set_summary <- data_set_preds %>%
  select(sig1, predictions, number_predictors) %>%
  unnest(.preserve = c(sig1, number_predictors)) %>%
  mutate(sample = (row_number() - 1) %% length(mu)) %>%
  group_by(sample, number_predictors) %>%
  summarize(mean.pred = mean(.),
            var.pred = var(.),
            n_pred = first(number_predictors)) %>%
  ungroup() %>%
  mutate(bias = mean.pred - rep(mu, times = 3),
         MSE = bias^2 + var.pred)

data_set_summary %>%
  ggplot(aes(x = as.factor(n_pred), y = bias)) +
  geom_jitter() + 
  ggtitle("Bias of predictions on test set")

data_set_summary %>%
  ggplot(aes(x = as.factor(n_pred), y = var.pred)) +
  geom_jitter() + 
  scale_y_continuous(limits = c(0, 0.6)) +
  ggtitle("Variance of predictions on test set")

data_set_summary %>%
  ggplot(aes(x = as.factor(n_pred), y = MSE)) +
  geom_jitter() + 
  scale_y_continuous(limits = c(0, 0.6)) +
  ggtitle("MSE of predictions on test set")

# Something is off, especially with the 3 predictor setting. Investigate

# Plots
# (May need to change "win.graph" to "x11" or "quartz")
windows(height=5, width=5.5, pointsize=15)
stripchart(bias ~ model, method="jitter", jitter=.1, vertical=TRUE, pch=20,
           main = "Bias of predictions on test set")
abline(h=0, lty="solid")

windows(height=5, width=5.5, pointsize=15)
stripchart(var ~ model, method="jitter", jitter=.1, vertical=TRUE, pch=20,
           main = "Variance of predictions on test set")

windows(height=5, width=5.5, pointsize=15)
stripchart(MSE ~ model, method="jitter", jitter=.1, vertical=TRUE, pch=20,
           main = "MSE (B^2+V) of predictions on test set")

# Summary statistics for variances and MSEs for prediction by model
mean(var[which(model==1)])
mean(var[which(model==2)])
mean(var[which(model==3)])
mean(MSE[which(model==1)])
mean(MSE[which(model==2)])
mean(MSE[which(model==3)])

# Mean of model-based estimates of variance 
apply(save.sig,MARGIN=2,FUN=mean) 


