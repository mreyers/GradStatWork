# Abalone work, still assignment 1 of 852
library(tidyverse)
abalone <- read_csv("Abalone.csv")

unique(abalone$Sex)

abalone <- abalone %>% 
  mutate(male_dummy = Sex == 1,
         female_dummy = Sex == 2) %>%
  select(-Sex) %>%
  select(Rings, everything())

pairs(~., abalone)

# Multiple variables display high correlation, frequently due to being measurements related to size and proxies
# of size. I expect multicollinearity to be a problem. 

# Remove outliers in height
abalone <- abalone %>%
  filter(Height > 0 & Height < 0.5)

set.seed(29003092)

abalone <- abalone %>%
  mutate(train = ifelse(runif(nrow(.)) > 0.75, "test", "train"))

abalone_train <- abalone %>% filter(train %in% "train") %>% dplyr::select(-train) %>% as.data.frame()
abalone_test <- abalone %>% filter(train %in% "test") %>% dplyr::select(-train) %>% as.data.frame()

# Forward stepwise
library(leaps)
library(MASS)

base_lm <- lm(Rings ~ 1, data = abalone_train)
full_lm <- lm(Rings ~ ., data = abalone_train)
summary(base_lm)

# c)
# k = log(n) for BIC calculation
forward_selec <- stepAIC(base_lm, direction = 'forward', 
                         scope = list(lower = base_lm, upper = full_lm), k = log(nrow(abalone_train)))

# Best model: 
# Step:  AIC=5043.66
# Rings ~ Shell + Shucked + Height + Whole + Viscera + Diameter + 
# male_dummy + female_dummy

# d)
stepwise_selec <- stepAIC(base_lm, direction = 'both', 
                          scope = list(lower = base_lm, upper = full_lm), k = log(nrow(abalone_train)),
                          trace = TRUE)

# Best model: 
# Step:  AIC=5043.66
# Rings ~ Shell + Shucked + Height + Whole + Viscera + Diameter + 
#  male_dummy + female_dummy

# e) No penalty
forward_selec_no_pen <- stepAIC(base_lm, direction = 'forward', 
                         scope = list(lower = base_lm, upper = full_lm), k = 0)

log_like <- c(7424.8, 5876.02, 5423.15, 5215.6, 5141.72, 5086.4, 5033.45, 5009.7, 4971.12, 4970.62) %>%
  as.data.frame() %>%
  mutate(n_pred = row_number(),
         BIC = . + n_pred * log(nrow(abalone_train))
  )
# At first look, it seems the model stopped at an appropriate time
# Best model: Rings ~ Shell + Shucked + Height + Whole + Viscera + Diameter + 
# male_dummy + female_dummy

# f) reg subsets
regsubsets_res <- regsubsets(Rings ~ ., data = abalone_train, nbest = 1)
summary(regsubsets_res)

summary(forward_selec_no_pen)

# Best models in regsubsets
# Shell
# Shell + Shucked
# SHell + shucked + Height
# Shucked + Height + whole + viscera
# Shucked + Height + whole + viscera + DIameter
# Shucked + Height + whole + viscera + DIameter + Shell
# Shucked + Height + whole + viscera + DIameter + male_dummy + female_dummy
# Shucked + Height + whole + viscera + DIameter + male_dummy + female_dummy + shell

# Best models in forward selection
# Shell
# SHell + shucked
# SHell + shucked + Height
# SHell + shucked + Height + Whole
# SHell + shucked + Height + Whole + Viscera
# SHell + shucked + Height + Whole + Viscera + DIameter
# SHell + shucked + Height + Whole + Viscera + DIameter + male_dummy
# SHell + shucked + Height + Whole + Viscera + DIameter + male_dummy + female_dummy
# SHell + shucked + Height + Whole + Viscera + DIameter + male_dummy + female_dummy + Length

# The differences become evident when we get to the model of 4 variables. Reasoning for this is that 
# the regsubsets approach allows Shell to exit the model while forward selection is greedy and cannot
# backtrack on previous decisions. Naturally differences exist in future, larger models due to the same 
# algorithmic discrepancy.

# g) Compute train and test error for each unique model discovered by variable selection methods above
model_1 <- lm(Rings ~ Shell, data = abalone_train) #regss
model_2 <- lm(Rings ~ Shell + Shucked, data = abalone_train) #regss
model_3 <- lm(Rings ~ Shell + Shucked + Height, data = abalone_train) #regss
model_4 <- lm(Rings ~ Shucked + Height + Whole + Viscera, data = abalone_train) #regss
model_5 <- lm(Rings ~ Shucked + Height + Whole + Viscera + Diameter, data = abalone_train) #regss
model_6 <- lm(Rings ~ Shucked + Height + Whole + Viscera + Diameter + Shell, data = abalone_train) #regss
model_7 <- lm(Rings ~ Shucked + Height + Whole + Viscera + Diameter + male_dummy + female_dummy,
              data = abalone_train) #regss
model_8 <- lm(Rings ~ Shucked + Height + Whole + Viscera + Diameter + male_dummy + female_dummy + Shell,
              data = abalone_train) #regss
model_9 <- lm(Rings ~ Shell + Shucked + Height + Whole, data = abalone_train)
model_10 <- lm(Rings ~ Shell + Shucked + Height + Whole + Viscera, data = abalone_train)
model_11 <- lm(Rings ~ Shell + Shucked + Height + Whole + Viscera + male_dummy, data = abalone_train)
model_12 <- lm(Rings ~ Shell + Shucked + Height + Whole + Viscera + male_dummy + female_dummy,
              data = abalone_train)

model_list_abalone <- list(model_1, model_2, model_3, model_4, model_5, model_6, model_7, 
                           model_8, model_9, model_10, model_11, model_12)
# Calculate sMSE, feeling lazy so not making super tidy at this point
sMSE_fun <- function(model){
  summary(model)$sigma^2
}
model_list_sMSE <- lapply(model_list_abalone, sMSE_fun)

unlist(model_list_sMSE)
# Minimum occurs at model 8 with sMSE of 4.823512
# Model 8: Shucked + Height + Whole + Viscera + Diameter + male_dummy + female_dummy + Shell
# Makes sense that the full model has best performance on the training data, it explains the most 
# variance in the response because all other models compared are a subset of this model

# Now to do the predictions to get MSPE
MSPE_fun <- function(model){
  y_hat <- predict(model, newdata = abalone_test)
  mean((y_hat - abalone_test$Rings)^2)
}
model_list_MSPE <- lapply(model_list_abalone, MSPE_fun)
unlist(model_list_MSPE)
# Again the minimum MSPE occurs at model 8 with MSPE of 4.544648
# Model 8: Shucked + Height + Whole + Viscera + Diameter + male_dummy + female_dummy + Shell
# I originally expected this model to be overfit and it may very well actually be overfit. Typically an 
# overfit model struggles in predictions on new data, as compared to a less complicated / more parsimonious
# model. In our model building we found the full model to have the smallest sMSE. Since our test set is a 
# random sample of the same data, the model may do as good a job of chasing errors in the training set as it 
# does in the test set. If there were critical decisions to be made based on this model, I would want to
# use a more extensive cross-validation process then just this single split. 