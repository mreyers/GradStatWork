library(tidyverse)
library(emdbook)
library(bezier)
library(patchwork)
library(tictoc)
library(multidplyr)
library(Matrix)

# I need to install these two library from github
# not sure if this affects others
# devtools::install_github("thomasp85/patchwork")
# devtools::install_github("https://github.com/tidyverse/multidplyr/commit/0085ded4048d7fbe5079616c40640dbf5982faf2")

num_cores <- parallel::detectCores()
cluster <- create_cluster(num_cores)

# Copy over the libraries, functions and data needed
cluster %>%
  cluster_library("tidyverse") %>%
  cluster_library("nnet")

# use this cluster for all parallel stuff
set_default_cluster(cluster)




# 1. In the Prostate data, use the training set to run neural nets (nnet()) with all
# combinations of (1, 3, 5, 7, and 10) hidden nodes, and (0, 0.001, 0.1, 1, and 2) decay. 

# Prostate data neural net fitting using Tom's starter code
prostate <- read_csv("Prostate.csv")

set.seed(120401002) 
prostate$set <- ifelse(runif(n=nrow(prostate))>0.5, yes=2, no=1)

prostate_train <- prostate %>%
  filter(set == 1) %>%
  select(-set, -train, -ID)

prostate_test <- prostate %>%
  filter(set == 2) %>%
  select(-set, -train, -ID)

rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}

grid <- expand.grid(size = c(1, 3, 5, 7, 10), decay = c(0, 0.001, 0.1, 1, 2))

prostate_train_nest <- as.data.frame(rescale(prostate_train, prostate_train)) %>%
  nest() 
prostate_test_nest <- as.data.frame(rescale(prostate_test, prostate_train)) %>%
  nest() 

# trace mutes the function and linout brings the output units to linear rather than logistic scale
grid_test <- grid %>%
  mutate(data = prostate_train_nest$data,
         data_test = prostate_test_nest$data) %>%
  # Arbitrary grouping to maximize spread across the 8 cores
  mutate(cluster_group = rep_len(1:num_cores, length.out = dim(grid)[1]))

# Probably over engineered but this is dope
tic()
grid_test <- grid_test %>%
  partition(cluster_group, cluster = cluster) %>%
  mutate(nnet_obj = pmap(list(size, decay, data), ~nnet(lpsa ~ ., data = ..3, decay = ..2, size = ..1,
                                                        linout = TRUE, trace = FALSE, maxit = 500))) %>%
  mutate(sMSE = map2_dbl(data, nnet_obj, ~ .y$value/nrow(.x)),
         MSPE = map2_dbl(data_test, nnet_obj, ~ mean((.x$lpsa - predict(.y, .x))^2))) %>%
  collect() %>%
  as_tibble()
toc()
object.size(grid_test)

# Is there a lighter way to do the above, without storing the same data in columns?
prostate_train_data <- as.data.frame(rescale(prostate_train, prostate_train))
prostate_test_data <- as.data.frame(rescale(prostate_test, prostate_train))

cluster %>%
  cluster_library("tidyverse") %>%
  cluster_library("nnet") %>%
  cluster_assign_value("data", prostate_train_data) %>%
  cluster_assign_value("data_test",prostate_test_data)

# use this cluster for all parallel stuff
set_default_cluster(cluster)

# Newer, lighter version for memory
tic()
grid_light_version <- grid %>%
  mutate(cluster_group = rep_len(1:num_cores, length.out = dim(grid)[1])) %>%
  partition(cluster_group, cluster = cluster) %>%
  mutate(nnet_obj = map2(size, decay, ~nnet(lpsa ~ ., data = data, decay = .y, size = .x,
                                                        linout = TRUE, trace = FALSE, maxit = 500))) %>%
  mutate(sMSE = map_dbl(nnet_obj, ~ .$value/nrow(data)),
         MSPE = map_dbl(nnet_obj, ~ mean((data_test$lpsa - predict(., data_test))^2))) %>%
  collect() %>%
  as_tibble()
toc()  
object.size(grid_light_version)
# Less than half the size of the other at same run times
# This will become much more important when the data dimension expands immensely

# (a) Present a table of the in-sample MSEs and the test error MSPEs.
grid_light_version %>%
  ungroup()%>% 
  select(size, decay, sMSE, MSPE) %>%
  arrange(decay, size) # arrange in this fashion because I expand the original grid this way


# (b) Comment on the relationship between sMSE vs. size and vs. decay. Explain why
# this happens.
grid_light_version %>%
  ungroup()%>% 
  select(size, decay, sMSE, MSPE) %>%
  arrange(decay, size) %>% # arrange in this fashion because I expand the original grid this way
  gather(Metric, Value, -size, -decay) %>%
  ggplot(aes(x = size, y = Value, group = size)) +
  geom_boxplot() + 
  scale_x_discrete(breaks= unique(grid_light_version$size)) + 
  ggtitle("Criteria as a function of Size") +
  facet_wrap(~ Metric, scales = 'free') +
  xlab("Size") + theme_bw()
# Spacing is at least consistent, figure out how to add in the appropriate labels

grid_light_version %>%
  ungroup()%>% 
  select(size, decay, sMSE, MSPE) %>%
  arrange(decay, size) %>% # arrange in this fashion because I expand the original grid this way
  gather(Metric, Value, -size, -decay) %>%
  ggplot(aes(x = as.factor(decay), y = Value, group = decay)) + # factorize this one because of scale
  geom_boxplot() + 
  ggtitle("Criteria as a function of Decay") +
  facet_wrap(~ Metric, scales = 'free') +
  xlab("Decay") + theme_bw()

# Just considering the sMSE for now, the value as a function of decay increases with decay. This is 
# most reasonably explained by decay being a regularization technique, aimed at reducing overfitting. Since
# sMSE rewards overfitting the sMSE naturally gets worse.

# Just considering the sMSE for now, the value as a function of size is relatively consistent. I expect that
# as size gets arbitrarily large sMSE will continue to decrease slightly. Otherwise size does not seem
# to overly impact sMSE in this example.

# (c) Report test-set MSPEs. Do they follow the same pattern as the sMSEs? What
# seems to be the best model(s) according to this?
  
# Now considering the MSPE, the value seems to be relatively consistent with respect to size. Outliers 
# in this sample of data are growing as size increases but the overall tendency is for the MSPE to be less
# variable. These outliers are likely the values of a given size that have 0 decay. 

# As expected, the MSPE is bad at 0 decay. Any decay > 0 appears to improve MSPE value and the variability
# of the measure. The best decay for this instance of the data in terms of MSPE currently looks to be 0.1.




# 2. In the Prostate data, use bootstrapping with B = 1 rep to select tuning parameters on
# the training set. Use the same starting grid as in Exercise 1, and expand your search if
# you choose parameter combinations that are on the boundary of your current search.
# Then repeat for B = 4 and B = 20.

# B = 1
set.seed(120401002) 
prostate <- prostate %>%
  select(lcavol, lweight, age, lbph, svi, lcp, gleason, pgg45, lpsa) %>% nest()
B <- 1

boot_train <- function(raw_data, resamp){
  
  prostate_train <- raw_data %>%
    filter(row_number() %in% resamp) 
  
  prostate_test <- raw_data %>%
    filter(!(row_number() %in% resamp))
  
  # Skip y, dont scale
  prostate_train[, 1:8] <- rescale(prostate_train[, 1:8], prostate_train[, 1:8])
  return(prostate_train)
}

boot_test <- function(raw_data, resamp){
  
  prostate_train <- raw_data %>%
    filter(row_number() %in% resamp)
  
  prostate_test <- raw_data %>%
    filter(!(row_number() %in% resamp)) 
  
  # Skip y, dont scale
  prostate_test[, 1:8] <- rescale(prostate_test[, 1:8], prostate_train[, 1:8]) 
  return(prostate_test)
}

splitFun <- function(train, test, size, decay){
  # grid_start <- data.frame(train = train, test = test, size = size, decay = decay) %>%
  #   mutate(nnet_obj = pmap(list(train, size, decay) ~ nnet(lpsa ~ ., data = ..1,
  #                                                          size = ..2, decay = ..3,
  #                                                          linout = TRUE, trace = FALSE, maxit = 500))) %>%
  #   mutate(sMSE = map2_dbl(train, nnet_obj, ~ .y$value/nrow(.x)),
  #          predictions = map2(test, nnet_obj, ~ predict(.y, .x)),
  #          MSPR = map2_dbl(test, predictions, ~ mean((.x$lpsa - predictions)^2))) # Complete these two statements 
  # 
  nnet_obj <- nnet(lpsa ~ ., data = train, size = size, decay = decay, linout = TRUE, trace = FALSE,
                   maxit = 500)
  
  sMSE <- nnet_obj$value / nrow(train)
  predictions <- predict(nnet_obj, test)
  MSPR <- mean((test$lpsa - predictions)^2)
  
  measures <- data.frame(sMSE = sMSE, MSPR = MSPR)
  return(measures)
}

# Data heavy approach
boot_1_df <- tibble(B = 1:B, prostate = prostate$data) %>%
  mutate(resamp = map(prostate, ~ sample.int(n=nrow(.), size=nrow(.), replace=TRUE)),
         train = map2(prostate, resamp, ~ boot_train(.x, .y)),
         test = map2(prostate, resamp, ~ boot_test(.x, .y)))

split_res <- grid %>% nest()

boot_1_df <- boot_1_df %>%
  mutate(grid_stuff = split_res$data) %>%
  select(-resamp, -prostate) %>%
  unnest(.preserve = c(train, test)) %>%
  mutate(pred_values = pmap(list(train, test, size, decay), ~ splitFun(..1, ..2, ..3, ..4)))
# Check the above, some values feel weird
         
# Now need to test the tuning values, use contour plot
library(plotly)

# With sMSE
boot_1_df %>%
  select(size, decay, pred_values) %>%
  unnest() %>%
  plot_ly(x = ~size, y = ~decay, z = ~sMSE, type = 'contour')

# MSPE, MSPR same thing
boot_1_df %>%
  select(size, decay, pred_values) %>%
  unnest() %>%
  plot_ly(x = ~size, y = ~decay, z = ~MSPR, type = 'heatmap')
# Plot obviously looks wrong, implies something is wrong with some of the initial values

# B = 4
B_4 <- 4
boot_4_df <- tibble(B = 1:B_4, prostate = prostate$data) %>%
  mutate(resamp = map(prostate, ~ sample.int(n=nrow(.), size=nrow(.), replace=TRUE)),
         train = map2(prostate, resamp, ~ boot_train(.x, .y)),
         test = map2(prostate, resamp, ~ boot_test(.x, .y)))

# B = 20
B_20 <- 20
boot_20_df <- tibble(B = 1:B_20, prostate = prostate$data) %>%
  mutate(resamp = map(prostate, ~ sample.int(n=nrow(.), size=nrow(.), replace=TRUE)),
         train = map2(prostate, resamp, ~ boot_train(.x, .y)),
         test = map2(prostate, resamp, ~ boot_test(.x, .y)))

# (a) For B = 1 compute a (very rough) t-based confidence interval for the true prediction error of each parameter combination using the mean and standard deviation
# of the squared errors. What does this tell you about your ability to select good
# tuning parameters for this data set using this method?

# (b) For each B make boxplots relative to best in each rep (this is just one point per
# combination for B = 1). Use these to identify the best combination(s). 

# (c) For the B = 20 case, compute mean MSPEs for each combination and make
# boxplots of these means against each parameter separately. What do these plots
# suggest might be good tuning parameters?

#   (d) For the B = 20 case, make a contour plot of the means against the two parameters.
# Scale out any extreme values if necessary so that you can focus on meaningful
# results. Where do good values appear to lie?

#   (e) Choose a best model and provide plots and/or numerical evidence to support your
# decision.

# (f) Report the MSPE from the test set for the chosen model only using each B, and
# compare these three to past results. (Note that these MSPEs are also based on
# very small n, so small differences are not very meaningful.)
