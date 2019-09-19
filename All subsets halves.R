prostate <-  read.table("C:/Users/mreyers/Documents/GitHub/GradStatWork/Prostate.csv", header=TRUE, sep=",", na.strings=" ")
head(prostate)
# Splitting data in half using random uniform selection to make two "set"s.

library(tidyverse)
library(magrittr)
library(leaps)

# Finally working
set.seed(120401002) 
prostate_tidy <- prostate %>%
  as.data.frame() %>%
  mutate(set = ifelse(runif(n = nrow(prostate)) > 0.5, "set1", "set2")) %>%
  nest(-set) %>%
  mutate(regsubs = map(data, ~ regsubsets(x = as.matrix(.x[, c(2:9)]), y = as.matrix(.x[, 10]),
                                          nbest = 1))) %>%
  mutate(summ_results = map(regsubs, ~ summary(.x)))

# Print summary results
print(prostate_tidy %>% slice(1) %>% pull(summ_results))
print(prostate_tidy %>% slice(2) %>% pull(summ_results))

# Print bic from summary results

# Love it, extract2 accesses lists and extract data frames, allowing for clean piping selection
prostate_tidy %>% slice(1) %>% pull(summ_results) %>% extract2(1) %>% extract('bic')
prostate_tidy %>% slice(2) %>% pull(summ_results) %>% extract2(1) %>% extract('bic')

# Cool transpose code
# mtcars %>%
#   rownames_to_column %>% 
#   gather(var, value, -rowname) %>% 
#   spread(rowname, value)
#prostate$set <- ifelse(runif(n=nrow(prostate))>0.5, yes=2, no=1)


# Fitting the models in succession from smallest to largest.  
# Fit one-var model. then update to 2-var model.  Could keep going.
# Each time computing sample-MSE (sMSE), BIC, and mean squared pred. error (MSPE). 
basic_reg <- function(data, n_vars, sizes){
  data <- data %>% select(-ID, -train)
  sizes <- sizes[, -1] # dont need intercept term manually listed
  if(n_vars == 0){
    return(lm(lpsa ~ 1, data = data))
  } else{
    return(lm(lpsa ~ ., data = data[, c(sizes[n_vars, ], TRUE)])) # Manually add in the response variable
  }
}


# Awesome, this finally works
model_sizes_1 <- prostate_tidy %>% slice(1) %>% pull(summ_results) %>% extract2(1) %>% extract('which')
model_sizes_2 <- prostate_tidy %>% slice(2) %>% pull(summ_results) %>% extract2(1) %>% extract('which')

# Stuff for newdata later
set1 <- prostate_tidy %>%
  filter(set %in% "set1")
set2 <- prostate_tidy %>%
  filter(set %in% "set2")

prostate_models_train1 <- prostate_tidy %>%
  filter(set %in% "set1") %>%
  mutate(n_vars = list(0:8)) %>% 
  unnest(.preserve = c(data, regsubs, summ_results)) %>%
  mutate(model_sizes = list(as.matrix(model_sizes_1$which)),
         reg_results = pmap(list(data, n_vars, model_sizes), ~ basic_reg(..1, ..2, ..3))) %>%
  mutate(sMSE = map(reg_results, ~ summary(.x)$sigma^2),
         BIC = pmap(list(reg_results, data), ~ extractAIC(..1, k = log(nrow(..2)))[2])) %>%
  mutate(new_data = set2$data,
         predictions = map2(reg_results, new_data, ~ predict(.x, newdata = .y)),
         MSPE = map2(predictions, new_data, ~ mean((.x - .y$lpsa)^2))) %>%
  select(n_vars, sMSE, BIC, MSPE)

prostate_models_train2 <- prostate_tidy %>%
  filter(set %in% "set2") %>%
  mutate(n_vars = list(0:8)) %>% 
  unnest(.preserve = c(data, regsubs, summ_results)) %>%
  mutate(model_sizes = list(as.matrix(model_sizes_2$which)),
         reg_results = pmap(list(data, n_vars, model_sizes), ~ basic_reg(..1, ..2, ..3))) %>%
  mutate(sMSE = map(reg_results, ~ summary(.x)$sigma^2),
         BIC = pmap(list(reg_results, data), ~ extractAIC(..1, k = log(nrow(..2)))[2])) %>%
  mutate(new_data = set1$data,
         predictions = map2(reg_results, new_data, ~ predict(.x, newdata = .y)),
         MSPE = map2(predictions, new_data, ~ mean((.x - .y$lpsa)^2))) %>%
  select(n_vars, sMSE, BIC, MSPE, reg_results)

# Discuss the results with someone, unsure if these are reasonable
# Need to do some more work, BIC and sMSE seem to function but MSPE looks wrong for Model / set 2
# TODO: Make the desired table
  # sMSE training error, MSPE testing error in table

prostate_models_train1 %>%
  gather(Metric, Value, -n_vars) %>%
  ggplot(aes(x = n_vars, y = as.numeric(Value))) +
  geom_line() +
  geom_point() +
  ggtitle("Performance by Metric for Training Set 1") +
  xlab("Best Model with given Number of Variables") + ylab("Value") +
  facet_wrap(~ Metric, scales = "free")

prostate_models_train2 %>%
  gather(Metric, Value, -n_vars) %>%
  ggplot(aes(x = n_vars, y = as.numeric(Value))) +
  geom_line() +
  geom_point() +
  ggtitle("Performance by Metric for Training Set 2") +
  xlab("Best Model with given Number of Variables") + ylab("Value") +
  facet_wrap(~ Metric, scales = "free")

# Together
prostate_models_train1 %>%
  mutate(model = "Model 1") %>%
  bind_rows(prostate_models_train2 %>% mutate(model = "Model 2")) %>%
  gather(Metric, Value, -n_vars, -model) %>%
  ggplot(aes(x = n_vars, y = as.numeric(Value), group = model, color = model)) +
  geom_line() +
  geom_point() +
  ggtitle("Performance by Metric for Both Training Sets") +
  xlab("Best Model with given Number of Variables") + ylab("Value") +
  facet_wrap(~ Metric, scales = "free")
