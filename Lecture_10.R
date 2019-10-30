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


# Go back and run this with 100 Neural Nets or whatever number I choose

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
grid_test_good <- grid_test
rm(grid_test)
object.size(grid_test_good)

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

# Remove the partitioned data set so that R can gc() the cluster, fixes some runtime hopefully
grid_light_version_good <- grid_light_version
rm(grid_light_version)
object.size(grid_light_version_good)
# Less than half the size of the other at same run times
# This will become much more important when the data dimension expands immensely

# (a) Present a table of the in-sample MSEs and the test error MSPEs.
grid_light_version_good %>%
  ungroup()%>% 
  select(size, decay, sMSE, MSPE) %>%
  arrange(decay, size) # arrange in this fashion because I expand the original grid this way


# (b) Comment on the relationship between sMSE vs. size and vs. decay. Explain why
# this happens.
grid_light_version_good %>%
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

grid_light_version_good %>%
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

prostate <- read_csv("Prostate.csv")

prostate <- prostate %>%
  select(lcavol, lweight, age, lbph, svi, lcp, gleason, pgg45, lpsa) %>%
  mutate(set = ifelse(runif(n=nrow(prostate))>0.5, yes=2, no=1)) 

prostate_train <- prostate %>%
  filter(set == 1) %>%
  select(-set) %>%
  nest()

prostate_test <- prostate %>%
  filter(set == 2) %>%
  select(-set) %>%
  nest()

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

# Fix this, returning too many values
splitFun <- function(train, test, size, decay, true_test_x, true_test_y){
  # grid_start <- data.frame(train = train, test = test, size = size, decay = decay) %>%
  #   mutate(nnet_obj = pmap(list(train, size, decay) ~ nnet(lpsa ~ ., data = ..1,
  #                                                          size = ..2, decay = ..3,
  #                                                          linout = TRUE, trace = FALSE, maxit = 500))) %>%
  #   mutate(sMSE = map2_dbl(train, nnet_obj, ~ .y$value/nrow(.x)),
  #          predictions = map2(test, nnet_obj, ~ predict(.y, .x)),
  #          MSPR = map2_dbl(test, predictions, ~ mean((.x$lpsa - predictions)^2))) # Complete these two statements 
  # 
  MSE.final <- 9e99
  se_all <- 0
  for(i in 1:100){
    nnet_obj <- nnet(lpsa ~ ., data = train, size = size, decay = decay, linout = TRUE, trace = FALSE,
                     maxit = 500)
    MSE <- nnet_obj$value/nrow(train)
    
    if(MSE < MSE.final){ 
      MSE.final <- MSE
      nn.final <- nnet_obj
      
    }
  }
  
  
  sMSE <- nn.final$value / nrow(train)
  predictions <- predict(nn.final, true_test_x)
  MSPE_train <- mean((test$lpsa - predict(nn.final, test))^2)
  MSPR <- mean((true_test_y - predictions)^2)
  se_all <- sd((true_test_y - predictions)^2)
  measures <- tibble(sMSE = sMSE, MSPE_train = MSPE_train, MSPR = MSPR, se_all = se_all)
  return(measures)
}

# Data heavy approach
cluster %>%
  cluster_library("tidyverse") %>%
  cluster_library("nnet") %>%
  cluster_assign_value("boot_train", boot_train) %>%
  cluster_assign_value("boot_test", boot_test) %>%
  cluster_assign_value("splitFun", splitFun)


tic()
boot_1_df <- tibble(B = 1:B, prostate = prostate_train$data, prostate_true_test = prostate_test$data) %>%
  mutate(resamp = map(prostate, ~ sample.int(n=nrow(.), size=nrow(.), replace=TRUE)),
         train = map2(prostate, resamp, ~ boot_train(.x, .y)),
         test = map2(prostate, resamp, ~ boot_test(.x, .y)),
         true_test_x = map2(prostate_true_test, train, ~rescale(.x[,1:8], .y[,1:8])),
         true_test_y = map(prostate_true_test, ~.$lpsa))

split_res <- grid %>% nest()

boot_1_df <- boot_1_df %>%
  mutate(grid_stuff = split_res$data) %>%
  select(-resamp, -prostate, -prostate_true_test) %>%
  unnest(.preserve = c(train, test, true_test_x, true_test_y))

boot_1_df <- boot_1_df %>%
  mutate(cluster_group = rep_len(1:num_cores, length.out = nrow(boot_1_df))) %>%
  partition(cluster_group, cluster = cluster) %>%
  mutate(pred_values = pmap(list(train, test, size, decay, true_test_x, true_test_y),
                            ~ splitFun(..1, ..2, ..3, ..4, ..5, ..6))) %>%
  collect()

boot_1_df_good <- boot_1_df
rm(boot_1_df)
boot_1_df <- boot_1_df_good
toc()
# 36.91 seconds without parallel
# 9.5 seconds with parallel


# Now need to test the tuning values, use contour plot
library(plotly)

# With sMSE
boot_1_df %>%
  select(size, decay, pred_values) %>%
  unnest() %>%
  filter(sMSE < 1.5) %>%
  plot_ly(x = ~size, y = ~decay, z = ~sMSE, type = 'contour')

# MSPE, MSPR same thing
boot_1_df %>%
  select(size, decay, pred_values) %>%
  unnest() %>%
  filter(MSPR < 1.5) %>%
  plot_ly(x = ~size, y = ~decay, z = ~MSPR, type = 'contour')


# B = 4
tic()
B_4 <- 4
boot_4_df <- tibble(B = list(1:B_4), prostate = prostate_train$data, prostate_true_test = prostate_test$data) %>%
  unnest(.preserve = c(prostate, prostate_true_test)) %>%
  mutate(resamp = map(prostate, ~ sample.int(n=nrow(.), size=nrow(.), replace=TRUE)),
         train = map2(prostate, resamp, ~ boot_train(.x, .y)),
         test = map2(prostate, resamp, ~ boot_test(.x, .y)),
         true_test_x = map2(prostate_true_test, train, ~rescale(.x[,1:8], .y[,1:8])),
         true_test_y = map(prostate_true_test, ~.$lpsa))


split_res <- grid %>% nest()

boot_4_df <- boot_4_df %>%
  mutate(grid_stuff = split_res$data) %>%
  select(-resamp, -prostate, -prostate_true_test) %>%
  unnest(.preserve = c(train, test, true_test_x, true_test_y)) 

boot_4_df <- boot_4_df %>%
  mutate(cluster_group = rep_len(1:num_cores, length.out = nrow(boot_4_df))) %>%
  partition(cluster_group, cluster = cluster) %>%
  mutate(pred_values = pmap(list(train, test, size, decay, true_test_x, true_test_y),
                            ~ splitFun(..1, ..2, ..3, ..4, ..5, ..6))) %>%
  collect()

# It seems that I just need to delete the original partitioned object to free the cluster
boot_4_df_good <- boot_4_df
rm(boot_4_df)
boot_4_df <- boot_4_df_good
toc()


boot_4_df %>%
  select(size, decay, pred_values) %>%
  unnest() %>%
  filter(MSPR < 1.5) %>%
  plot_ly(x = ~size, y = ~decay, z = ~MSPR, type = 'contour')

# B = 20
tic()
B_20 <- 20
boot_20_df <- tibble(B = list(1:B_20), prostate = prostate_train$data,
                     prostate_true_test = prostate_test$data) %>%
  unnest(.preserve = c(prostate, prostate_true_test)) %>%
  mutate(resamp = map(prostate, ~ sample.int(n=nrow(.), size=nrow(.), replace=TRUE)),
         train = map2(prostate, resamp, ~ boot_train(.x, .y)),
         test = map2(prostate, resamp, ~ boot_test(.x, .y)),
         true_test_x = map2(prostate_true_test, train, ~rescale(.x[,1:8], .y[,1:8])),
         true_test_y = map(prostate_true_test, ~.$lpsa))


split_res <- grid %>% nest()

boot_20_df <- boot_20_df %>%
  mutate(grid_stuff = split_res$data) %>%
  select(-resamp, -prostate, -prostate_true_test) %>%
  unnest(.preserve = c(train, test, true_test_x, true_test_y)) 

boot_20_df <- boot_20_df %>%
  mutate(cluster_group = rep_len(1:num_cores, length.out = nrow(boot_20_df))) %>%
  partition(cluster_group, cluster = cluster) %>%
  mutate(pred_values = pmap(list(train, test, size, decay, true_test_x, true_test_y),
                            ~ splitFun(..1, ..2, ..3, ..4, ..5, ..6))) %>%
  collect()
toc()

# boot_20_df %>% 
#   select(size, decay, pred_values) %>%
#   unnest() %>%
#   filter(MSPR < 1.5) %>%
#   plot_ly(x = ~size, y = ~decay, z = ~MSPR, type = 'contour')

boot_20_df_good <- boot_20_df
rm(boot_20_df)
boot_20_df <- boot_20_df_good
# The above has been run, dope
# Definitely needs an expanded grid, view plots to determine changes necessary


# (a) For B = 1 compute a (very rough) t-based confidence interval for the true prediction error of each parameter combination using the mean and standard deviation
# of the squared errors. What does this tell you about your ability to select good
# tuning parameters for this data set using this method?
boot_1_df_int <- boot_1_df %>%
  unnest(.preserve = c(train, test, true_test_x, true_test_y)) %>%
  ungroup() %>%
  mutate(se = se_all) %>%
  select(-se_all) %>%
  mutate(lower_bound = MSPR - qt(0.975, 100 - 1) * se,
         upper_bound = MSPR + qt(0.975, 100 - 1) * se) %>%
  unite(combo, size, decay)
  
boot_1_df_int %>%
  ggplot(aes(x = combo, y = MSPR)) +
  geom_point() + geom_errorbar(aes(ymin = lower_bound, ymax = upper_bound)) +
  ggtitle("Interval Comparison for MSPR by Decay and Size Combination") +
  xlab("Size_Decay") +
  theme_bw() +
  theme(axis.text.x = element_text(size = rel(0.75), angle = 70, hjust = 1))

# Due to the degree of overlap between the combinations it seems likely that we will not always get the 
# truly best method due to randomness. More fits will likely help reduce this but we will always be uncertain.

# (b) For each B make boxplots relative to best in each rep (this is just one point per
# combination for B = 1). Use these to identify the best combination(s). 
minimum_mspe_train <- boot_1_df_int %>%
  ungroup() %>%
  arrange(MSPE_train) %>%
  slice(1) %>%
  pull(MSPE_train)

boot_1_df_int %>%
  mutate(MSPE_train = sqrt(MSPE_train / minimum_mspe_train)) %>%
  ggplot(aes(x = combo, y = MSPE_train)) +
  geom_point() +
  ggtitle("Relative MSPR by Decay and Size Combination") +
  xlab("Size_Decay") +
  scale_y_continuous(limits = c(1, 2)) + 
  theme_bw() +
  theme(axis.text.x = element_text(size = rel(0.75), angle = 70, hjust = 1))

# The best combination here looks like it could be any size with 1 decay or size 1 with decay 0.1

minimum_mspe_train <- boot_4_df %>%
  unnest(.preserve = c(train, test, true_test_x, true_test_y)) %>%
  ungroup() %>%
  mutate(se = se_all) %>%
  select(-se_all) %>%
  arrange(B) %>%
  group_by(B) %>%
  arrange(MSPE_train) %>%
  slice(1) %>%
  select(min_mspr = MSPE_train)
  
boot_4_df %>%
  unnest(.preserve = c(train, test, true_test_x, true_test_y)) %>%
  ungroup() %>%
  left_join(minimum_mspe_train) %>%
  unite(combo, size, decay) %>%
  mutate(se = se_all,
         MSPE_train = sqrt(MSPE_train / min_mspr)) %>%
  ggplot(aes(x = combo, y = MSPE_train)) +
  geom_boxplot() +
  ggtitle("Relative MSPR by Decay and Size Combination") +
  xlab("Size_Decay") +
  scale_y_continuous(limits = c(1, 2)) + 
  theme_bw() +
  theme(axis.text.x = element_text(size = rel(0.75), angle = 70, hjust = 1))

# Again this could be any size with decay 1 or size 1 with decay 0.1. The possibility also exists for 
# size any size with decay 0.1.

minimum_mspe_train <- boot_20_df %>%
  unnest(.preserve = c(train, test, true_test_x, true_test_y)) %>%
  ungroup() %>%
  mutate(se = se_all) %>%
  select(-se_all) %>%
  arrange(B) %>%
  group_by(B) %>%
  arrange(MSPE_train) %>%
  slice(1) %>%
  select(min_mspr = MSPE_train)

boot_20_df %>%
  unnest(.preserve = c(train, test, true_test_x, true_test_y)) %>%
  ungroup() %>%
  left_join(minimum_mspe_train) %>%
  unite(combo, size, decay) %>%
  mutate(se = se_all,
         MSPE_train = MSPE_train / min_mspr) %>%
  ggplot(aes(x = combo, y = MSPE_train)) +
  geom_boxplot() +
  ggtitle("Relative MSPR by Decay and Size Combination") +
  xlab("Size_Decay") +
  scale_y_continuous(limits = c(1,2)) +
  theme_bw() +
  theme(axis.text.x = element_text(size = rel(0.75), angle = 70, hjust = 1))

# With 20 bootstraps it is starting to become more clear that any size and decay 1 should be reasonable. 
# By increasing the number of bootstraps we get a better measure of variability in the fits and a clearer
# picture. 


# (c) For the B = 20 case, compute mean MSPEs for each combination and make
# boxplots of these means against each parameter separately. What do these plots
# suggest might be good tuning parameters?

boot_20_df %>%
  unnest(.preserve = c(train, test, true_test_x, true_test_y)) %>%
  ungroup() %>%
  left_join(minimum_mspe_train) %>%
  mutate(se = se_all,
         MSPE_train = MSPE_train / min_mspr) %>%
  ggplot(aes(x = as.factor(size), y = MSPE_train)) +
  geom_boxplot() +
  ggtitle("Relative MSPR by Decay and Size Combination") +
  xlab("Size") +
  scale_y_continuous(limits = c(1,2)) +
  theme_bw() +
  theme(axis.text.x = element_text(size = rel(0.75), angle = 70, hjust = 1))
  
boot_20_df %>%
  unnest(.preserve = c(train, test, true_test_x, true_test_y)) %>%
  ungroup() %>%
  left_join(minimum_mspe_train) %>%
  mutate(se = se_all,
         MSPE_train = MSPE_train / min_mspr) %>%
  ggplot(aes(x = as.factor(decay), y = MSPE_train)) +
  geom_boxplot() +
  ggtitle("MSPR by Decay and Size Combination") +
  xlab("Decay") +
  scale_y_continuous(limits = c(1, 3)) +
  theme_bw() +
  theme(axis.text.x = element_text(size = rel(0.75), angle = 70, hjust = 1))

#   (d) For the B = 20 case, make a contour plot of the means against the two parameters.
# Scale out any extreme values if necessary so that you can focus on meaningful
# results. Where do good values appear to lie?

boot_20_df %>%
  unnest(.preserve = c(train, test, true_test_x, true_test_y)) %>%
  ungroup() %>%
  left_join(minimum_mspe_train) %>%
  mutate(se = se_all,
         MSPE_train = MSPE_train / min_mspr) %>%
  group_by(size, decay) %>%
  summarize(MSPE_train = mean(MSPE_train)) %>%
  filter(MSPE_train < 2) %>%
  plot_ly(x = ~decay, y = ~size, z = ~MSPE_train, type = 'contour')

# Most of the good values lie on decay 1 with varying sizes. There may be reason to suggest using a size
# larger than 10 with decay 1 to get better results. Further, the minimum value actually occurs at
# a size of 1 and a decay of 0.001. In e) I will make my grid finer over the spaces of interest to 
# investigate these possibilities.


#   (e) Choose a best model and provide plots and/or numerical evidence to support your
# decision.
grid <- expand.grid(decay = c(0.001, 0.01, 0.25, 0.5, 0.75, 1, 1.25), 
                    size = c(1, 3, 5, 7, 10, 13, 17, 20))

tic()
B_20 <- 20
boot_20_df <- tibble(B = list(1:B_20), prostate = prostate_train$data,
                     prostate_true_test = prostate_test$data) %>%
  unnest(.preserve = c(prostate, prostate_true_test)) %>%
  mutate(resamp = map(prostate, ~ sample.int(n=nrow(.), size=nrow(.), replace=TRUE)),
         train = map2(prostate, resamp, ~ boot_train(.x, .y)),
         test = map2(prostate, resamp, ~ boot_test(.x, .y)),
         true_test_x = map2(prostate_true_test, train, ~rescale(.x[,1:8], .y[,1:8])),
         true_test_y = map(prostate_true_test, ~.$lpsa))


split_res <- grid %>% nest()

boot_20_df <- boot_20_df %>%
  mutate(grid_stuff = split_res$data) %>%
  select(-resamp, -prostate, -prostate_true_test) %>%
  unnest(.preserve = c(train, test, true_test_x, true_test_y)) 

boot_20_df <- boot_20_df %>%
  mutate(cluster_group = rep_len(1:num_cores, length.out = nrow(boot_20_df))) %>%
  partition(cluster_group, cluster = cluster) %>%
  mutate(pred_values = pmap(list(train, test, size, decay, true_test_x, true_test_y),
                            ~ splitFun(..1, ..2, ..3, ..4, ..5, ..6))) %>%
  collect()
toc()

minimum_mspe_train <- boot_20_df %>%
  unnest(.preserve = c(train, test, true_test_x, true_test_y)) %>%
  ungroup() %>%
  select(-se_all) %>%
  arrange(B) %>%
  group_by(B) %>%
  arrange(MSPE_train) %>%
  slice(1) %>%
  select(min_mspr = MSPE_train)

boot_20_df %>%
  unnest(.preserve = c(train, test, true_test_x, true_test_y)) %>%
  ungroup() %>%
  left_join(minimum_mspe_train) %>%
  mutate(MSPE_train = MSPE_train / min_mspr) %>%
  group_by(size, decay) %>%
  summarize(MSPE_train = mean(MSPE_train)) %>%
  filter(MSPE_train < 2) %>%
  plot_ly(x = ~decay, y = ~size, z = ~MSPE_train, type = 'contour')

# Decay of 0.5, any size 3 or greater. This grid seems conclusive.

# (f) Report the MSPE from the test set for the chosen model only using each B, and
# compare these three to past results. (Note that these MSPEs are also based on
# very small n, so small differences are not very meaningful.)


# 3. In the Abalone data, we will rerun the 20-splits problem using neural nets. The difficulty
# here is that you need to be able to tune the NNs and select a “best” set of tuning
# parameters before you assess the chosen model on a test set. That is, the test set from
# each of the 20 splits can only be used once, on the final chosen NN model for that
# split, and may not be used to help choose the model. You will therefore need to use
# some approach to tuning that can be applied to the 75% of data that is in the training
# set.

# Try to automate this code to choose best parameter values
set.seed(890987665)
abalone <- read_csv("Abalone.csv")

# Remove outliers in height
abalone <- abalone %>%
  filter(Height > 0 & Height < 0.5)


boot_train_ab <- function(raw_data, resamp){
  
  prostate_train <- raw_data %>%
    filter(row_number() %in% resamp) 
  
  prostate_test <- raw_data %>%
    filter(!(row_number() %in% resamp))
  
  # Skip y, dont scale
  prostate_train[, 1:7] <- rescale(prostate_train[, 1:7], prostate_train[, 1:7])
  return(prostate_train)
}

boot_test_ab <- function(raw_data, resamp){
  
  prostate_train <- raw_data %>%
    filter(row_number() %in% resamp)
  
  prostate_test <- raw_data %>%
    filter(!(row_number() %in% resamp)) 
  
  # Skip y, dont scale
  prostate_test[, 1:7] <- rescale(prostate_test[, 1:7], prostate_train[, 1:7]) 
  return(prostate_test)
}

splitFun_ab <- function(train, test, size, decay){
  
  MSE.final <- 9e99
  for(i in 1:10){
    nnet_obj <- nnet(Rings ~ ., data = train, size = size, decay = decay, linout = TRUE, trace = FALSE,
                     maxit = 500)
    MSE <- nnet_obj$value/nrow(train)
    if(MSE < MSE.final){ 
      MSE.final <- MSE
      nn.final <- nnet_obj
    }
  }
  
  
  sMSE <- nn.final$value / nrow(train)
  predictions <- predict(nn.final, test)
  MSPR <- mean((test$Rings - predictions)^2)
  
  measures <- data.frame(sMSE = sMSE, MSPR = MSPR)
  return(measures)
}
# Edit this to return a neural net object

# Data heavy approach
cluster %>%
  cluster_library("tidyverse") %>%
  cluster_library("nnet") %>%
  cluster_assign_value("boot_train_ab", boot_train_ab) %>%
  cluster_assign_value("boot_test_ab", boot_test_ab) %>%
  cluster_assign_value("splitFun_ab", splitFun_ab)


# Need to build a function that handles all the splitting
# Will need a number of folds and from that to create splits
k <- 20 # N-Folds

r_vec <- sample(1:k, dim(abalone)[1], replace=TRUE)
res <- tibble(test = 1:k, data = list(0))
tic()
for(i in 1:k){
  # Split the data
  abalone_splits <- abalone %>%
    mutate(folds = r_vec,
           male_dummy = Sex == 1,
           female_dummy = Sex == 2) 
  
  abalone_train_split <- abalone_splits %>%
    filter(folds != i) %>%
    dplyr::select(-folds, -Sex) 
  
  abalone_test_split <- abalone_splits %>%
    filter(folds == i) %>%
    dplyr::select(-folds, -Sex) 
  
  # Apply and time my functions, seems to run faster on laptop
  tic()
  B_20 <- 20
  grid <- expand.grid(size = c(1, 3, 5, 7, 10, 15, 20), decay = c(0, 0.001, 0.1, 1, 2, 5))
  abalone_train_nest <- abalone_train_split %>% nest()
  
  boot_20_df <- tibble(B = 1:B_20, abalone = abalone_train_nest$data) %>%
    mutate(resamp = map(abalone, ~ sample.int(n=nrow(.), size=nrow(.), replace=TRUE)),
           train = map2(abalone, resamp, ~ boot_train_ab(.x, .y)),
           test = map2(abalone, resamp, ~ boot_test_ab(.x, .y)))
  
  
  split_res <- grid %>% nest()
  
  boot_20_df <- boot_20_df %>%
    mutate(grid_stuff = split_res$data) %>%
    select(-resamp, -abalone) %>%
    unnest(.preserve = c(train, test)) 
  
  boot_20_df <- boot_20_df %>%
    mutate(cluster_group = rep_len(1:num_cores, length.out = nrow(boot_20_df))) %>%
    partition(cluster_group, cluster = cluster) %>%
    mutate(pred_values = pmap(list(train, test, size, decay), ~ splitFun_ab(..1, ..2, ..3, ..4))) %>%
    collect() %>%
    select(-train, -test)
  toc()
  
  res[i, 2] <- boot_20_df %>% ungroup() %>% nest()
}
toc()
# Need to design some grid to test parameters
# Make sure to save results from each bootstrapped sample so that values can be compared



# TODO: Set up GCE stuff for this homework and get ready to crank some dope parallel stuff
# This looks like it will work, I am content with the design
