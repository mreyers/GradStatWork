# Lecture 11

#  In the Prostate data, use the training set and run 4 random forests using m =
#  1, p/3, 2p/3, p (the last being bagging; round m where appropriate).

library(tidyverse)

prostate_data <- read_csv('prostate.csv')


set.seed(120401002) 
prostate_data$set <- ifelse(runif(n=nrow(prostate_data))>0.5, yes=2, no=1)

prostate_train <- prostate_data %>%
  filter(set == 1) %>%
  select(-set, -train, -ID)

prostate_test <- prostate_data %>%
  filter(set == 2) %>%
  select(-set, -train, -ID)

# Train the random forests
library(randomForest)
rf_1 <- randomForest(data=prostate_train, lpsa ~ ., 
             importance=TRUE, ntree=1000, mtry=1, keep.forest=TRUE)
plot(rf_1) # pretty stable

rf_p3 <- randomForest(data=prostate_train, lpsa ~ ., 
                     importance=TRUE, ntree=1000, mtry=round((dim(prostate_train)[2]-1)/3), keep.forest=TRUE)
plot(rf_p3) # Pretty stable as well, need 300+ trees, have 1000

rf_2p3 <- randomForest(data=prostate_train, lpsa ~ ., 
                     importance=TRUE, ntree=1000, mtry=round(2*(dim(prostate_train)[2]-1)/3), keep.forest=TRUE)
plot(rf_2p3) # Pretty stable after 400 trees

rf_3p3 <- randomForest(data=prostate_train, lpsa ~ ., 
                     importance=TRUE, ntree=1000, mtry=round((dim(prostate_train)[2]-1)), keep.forest=TRUE)
plot(rf_3p3) # Pretty stable after 100-150 trees

# a) Ensure I am using enough trees, provide evidence
# Each of the plots above clearly demonstrates stability after a given number of trees well less than
# the 1000 used. We frequently gain stability around 200 trees so 1000 trees is easily reasonable.

# b) Report and interpret the variable importance measures for each m and compare
# to past results.

importance(rf_1)
varImpPlot(rf_1)

importance(rf_p3)
varImpPlot(rf_p3)

importance(rf_2p3)
varImpPlot(rf_2p3)

importance(rf_3p3)
varImpPlot(rf_3p3)

# Whether viewing node purity or %IncMSE, lcavol improves in performance as the value of
# mtry increases. This matches our expectations as it has routinely been the most important variable
# in our other methods and an increasing mtry allows for lcavol to be considered for splits more 
# frequently.

# c)  Compute the OOB estimate of error and the MSPE. How do they compare to
# each other? How does the MSPE compare to other methods?

# OOB
(rf_1_oob <- mean((predict(rf_1) - prostate_train$lpsa)^2))

(rf_p3_oob <- mean((predict(rf_p3) - prostate_train$lpsa)^2))

(rf_2p3_oob <- mean((predict(rf_2p3) - prostate_train$lpsa)^2))

(rf_3p3_oob <- mean((predict(rf_3p3) - prostate_train$lpsa)^2))

# MSPE
(rf_1_MSPE <- mean((predict(rf_1, prostate_test) - prostate_test$lpsa)^2))

(rf_p3_MSPE <- mean((predict(rf_p3, prostate_test) - prostate_test$lpsa)^2))

(rf_2p3_MSPE <- mean((predict(rf_2p3, prostate_test) - prostate_test$lpsa)^2))

(rf_3p3_MSPE <- mean((predict(rf_3p3, prostate_test) - prostate_test$lpsa)^2))


# The OOB estimates suggest that these random forest methods are relatively similar to each other
# with a slight preference being given to the forests using mtry values $p/3$ or larger.
# Using MSPE, there is a much more clear preference for the forests that use mtry >= $p/3$ in comparison
# to other random forests. Comparing to other methods that have been used, these random forest MSPE
# values are reasonably similar. I expected them to be a bit lower but this may be a product of not 
# tuning the models at this step and having a small data set.



# d) Repeat the 4 model fits changing nodesize to half and double the default (you can
# use decimals; the function will not split if it is below the stated value). This is a
# gentle attempt to tune nodesize. Which nodesize(s) appear to work best here?

# Should this be OOB or MSPE as criterion?
rf_1_smallnode <- randomForest(data=prostate_train, lpsa ~ ., 
                               importance=TRUE, ntree=1000, mtry=1, keep.forest=TRUE,
                               nodesize = 2.5) # I believe default is 5

rf_p3_smallnode <- randomForest(data=prostate_train, lpsa ~ ., 
                                importance=TRUE, ntree=1000, mtry=round((dim(prostate_train)[2]-1)/3),
                                keep.forest=TRUE,
                                nodesize = 2.5) 

rf_2p3_smallnode <- randomForest(data=prostate_train, lpsa ~ ., 
                                importance=TRUE, ntree=1000, mtry=round(2 * (dim(prostate_train)[2]-1)/3),
                                keep.forest=TRUE,
                                nodesize = 2.5) 

rf_3p3_smallnode <- randomForest(data=prostate_train, lpsa ~ ., 
                                importance=TRUE, ntree=1000, mtry=round((dim(prostate_train)[2]-1)),
                                keep.forest=TRUE,
                                nodesize = 2.5) 

(rf_1_smallnode_MSPE <- mean((predict(rf_1_smallnode, prostate_test) - prostate_test$lpsa)^2))

(rf_p3_smallnode_MSPE <- mean((predict(rf_p3_smallnode, prostate_test) - prostate_test$lpsa)^2))

(rf_2p3_smallnode_MSPE <- mean((predict(rf_2p3_smallnode, prostate_test) - prostate_test$lpsa)^2))

(rf_3p3_smallnode_MSPE <- mean((predict(rf_3p3_smallnode, prostate_test) - prostate_test$lpsa)^2))

# About the same in terms of MSPE, same in terms of decision.

rf_1_largenode <- randomForest(data=prostate_train, lpsa ~ ., 
                               importance=TRUE, ntree=1000, mtry=1, keep.forest=TRUE,
                               nodesize = 10) # I believe default is 5

rf_p3_largenode <- randomForest(data=prostate_train, lpsa ~ ., 
                                importance=TRUE, ntree=1000, mtry=round((dim(prostate_train)[2]-1)/3),
                                keep.forest=TRUE,
                                nodesize = 10) 

rf_2p3_largenode <- randomForest(data=prostate_train, lpsa ~ ., 
                                 importance=TRUE, ntree=1000, mtry=round(2 * (dim(prostate_train)[2]-1)/3),
                                 keep.forest=TRUE,
                                 nodesize = 10)

rf_3p3_largenode <- randomForest(data=prostate_train, lpsa ~ ., 
                                 importance=TRUE, ntree=1000, mtry=round((dim(prostate_train)[2]-1)),
                                 keep.forest=TRUE,
                                 nodesize = 10)

(rf_1_largenode_MSPE <- mean((predict(rf_1_largenode, prostate_test) - prostate_test$lpsa)^2))

(rf_p3_largenode_MSPE <- mean((predict(rf_p3_largenode, prostate_test) - prostate_test$lpsa)^2))

(rf_2p3_largenode_MSPE <- mean((predict(rf_2p3_largenode, prostate_test) - prostate_test$lpsa)^2))

(rf_3p3_largenode_MSPE <- mean((predict(rf_3p3_largenode, prostate_test) - prostate_test$lpsa)^2))

# Using MSPE, the larger nodesize demonstrates less consistent behaviour than 

# 2. Using the Abalone data, rerun the 20 splits and make sure that you list Sex as a
# factor. Include random forests among the methods used. Tune both the number of
# variables at each split and the minimum terminal node size using OOB error. Explain
# what you are doing for the tuning and why. Add the default and the best-tuned RF
# results to the root MSPE boxplot, and to the relative root MSPE boxplot. Make
# appropriate comments.

# Grid of test values
grid_rf <- expand.grid(nodesize = c(2, 4, 6, 8, 10),
                       mtry = c(1, 2, 4, 6, 8))

# Number of iterations
  # Dont need to bootstrap because predict already operates on the OOB in a RF

# Read in Abalone
set.seed(890987665)
abalone <- read_csv("Abalone.csv")

# Remove outliers in height
abalone <- abalone %>%
  filter(Height > 0 & Height < 0.5)

# Set up splits
r_vec <- sample(1:k, dim(abalone)[1], replace=TRUE)

# Holder df for all the results
rf_data <- tibble(split = 1:20) %>%
  unnest() %>%
  mutate(grid = list(grid_rf)) %>%
  unnest()


# Can I do this, parallelized, without a loop?

# Split the data
abalone_splits <- abalone %>%
  mutate(folds = r_vec,
         male_dummy = Sex == 1,
         female_dummy = Sex == 2) %>%
  select( -Sex)


# Requires cluster environment to have both the full abalone data set and the number of boots stored
fit_rf <- function(abalone_full, split, nodesize, mtry){
  # Takes in data smarter to make lighter objects, builds and evals rf
  
  # Establish appropriate data sets to work with
  abalone_train <- abalone_full %>%
    filter(folds != split) %>%
    select(-folds)
  
  abalone_test <- abalone_full %>%
    filter(folds == split) %>%
    select(-folds)
  
  # Train a RF 
  my_rf <- randomForest(data=abalone_train, Rings ~ ., 
                        importance=TRUE, ntree=1000, mtry=mtry,
                        keep.forest=TRUE,
                        nodesize = nodesize)
  OOB <- mean((predict(my_rf) - abalone_train$Rings)^2)
  MSPE <- mean((predict(my_rf, abalone_test) - abalone_test$Rings)^2)
  
  # Return the values that are relevant
  return(tibble(OOB = OOB, MSPE = MSPE))
}

# Build an appropriate cluster 
library(multidplyr)
num_cores <- parallel::detectCores()
cluster <- create_cluster(num_cores)

# Copy over the libraries, functions and data needed
cluster %>%
  cluster_library("tidyverse") %>%
  cluster_library("randomForest") %>%
  cluster_assign_value("abalone_full", abalone_splits) %>%
  cluster_assign_value("fit_rf", fit_rf)

# use this cluster for all parallel stuff
set_default_cluster(cluster)

# Make the tibble parallel ready
tic()
rf_data <- rf_data %>%
  mutate(cluster_group = rep_len(1:num_cores, length.out = nrow(rf_data))) %>%
  partition(cluster_group, cluster = cluster) %>% 
  mutate(metrics = pmap(list(split, nodesize, mtry), ~fit_rf(abalone_full, ..1, ..2, ..3))) %>%
  collect()
toc()

backup <- rf_data

head(backup)

rf_data %>%
  ungroup() %>%
  unnest() %>%
  arrange(split, nodesize, mtry) %>%
  write.csv("random_forest_results.csv")
