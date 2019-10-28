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
# Noted by each plot

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

(rf_1_oob <- mean((predict(rf_1) - prostate_train$lpsa)^2))

(rf_p3_oob <- mean((predict(rf_p3) - prostate_train$lpsa)^2))

(rf_2p3_oob <- mean((predict(rf_2p3) - prostate_train$lpsa)^2))

(rf_3p3_oob <- mean((predict(rf_3p3) - prostate_train$lpsa)^2))

# The MSPE for the random forest is roughly the same as other methods and possibly a bit better.

# REVIEW CLAIM

# d) Repeat the 4 model fits changing nodesize to half and double the default (you can
# use decimals; the function will not split if it is below the stated value). This is a
# gentle attempt to tune nodesize. Which nodesize(s) appear to work best here?

rf_1_smallnode <- randomForest(data=prostate_train, lpsa ~ ., 
                               importance=TRUE, ntree=1000, mtry=1, keep.forest=TRUE,
                               nodesize = 2.5) # I believe default is 5

rf_p3_smallnode <- randomForest(data=prostate_train, lpsa ~ ., 
                                importance=TRUE, ntree=1000, mtry=round((dim(prostate_train)[2]-1)/3),
                                keep.forest=TRUE,
                                nodesize = 2.5) # I believe default is 5

rf_2p3_smallnode <- randomForest(data=prostate_train, lpsa ~ ., 
                                importance=TRUE, ntree=1000, mtry=round(2 * (dim(prostate_train)[2]-1)/3),
                                keep.forest=TRUE,
                                nodesize = 2.5) # I believe default is 5

rf_3p3_smallnode <- randomForest(data=prostate_train, lpsa ~ ., 
                                importance=TRUE, ntree=1000, mtry=round((dim(prostate_train)[2]-1)),
                                keep.forest=TRUE,
                                nodesize = 2.5) # I believe default is 5

(rf_1_smallnode_oob <- mean((predict(rf_1_smallnode) - prostate_train$lpsa)^2))

(rf_p3_smallnode_oob <- mean((predict(rf_p3_smallnode) - prostate_train$lpsa)^2))

(rf_2p3_smallnode_oob <- mean((predict(rf_2p3_smallnode) - prostate_train$lpsa)^2))

(rf_3p3_smallnode_oob <- mean((predict(rf_3p3_smallnode) - prostate_train$lpsa)^2))

# About the same

rf_1_largenode <- randomForest(data=prostate_train, lpsa ~ ., 
                               importance=TRUE, ntree=1000, mtry=1, keep.forest=TRUE,
                               nodesize = 2.5) # I believe default is 5

rf_p3_largenode <- randomForest(data=prostate_train, lpsa ~ ., 
                                importance=TRUE, ntree=1000, mtry=round((dim(prostate_train)[2]-1)/3),
                                keep.forest=TRUE,
                                nodesize = 2.5) # I believe default is 5

rf_2p3_largenode <- randomForest(data=prostate_train, lpsa ~ ., 
                                 importance=TRUE, ntree=1000, mtry=round(2 * (dim(prostate_train)[2]-1)/3),
                                 keep.forest=TRUE,
                                 nodesize = 2.5) # I believe default is 5

rf_3p3_largenode <- randomForest(data=prostate_train, lpsa ~ ., 
                                 importance=TRUE, ntree=1000, mtry=round((dim(prostate_train)[2]-1)),
                                 keep.forest=TRUE,
                                 nodesize = 2.5) # I believe default is 5

(rf_1_largenode_oob <- mean((predict(rf_1_largenode) - prostate_train$lpsa)^2))

(rf_p3_largenode_oob <- mean((predict(rf_p3_largenode) - prostate_train$lpsa)^2))

(rf_2p3_largenode_oob <- mean((predict(rf_2p3_largenode) - prostate_train$lpsa)^2))

(rf_3p3_largenode_oob <- mean((predict(rf_3p3_largenode) - prostate_train$lpsa)^2))

# This seems a bit better than the above. I prefer the larger node size.

# 2. Using the Abalone data, rerun the 20 splits and make sure that you list Sex as a
# factor. Include random forests among the methods used. Tune both the number of
# variables at each split and the minimum terminal node size using OOB error. Explain
# what you are doing for the tuning and why. Add the default and the best-tuned RF
# results to the root MSPE boxplot, and to the relative root MSPE boxplot. Make
# appropriate comments.
