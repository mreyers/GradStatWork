# Simulation of regression to COMPARE MODEL SELECTION CRITERIA
#   on models with different numbers of variables
# Model is E(Y) = 0 + 1 X1 + 1 X2 + e   with e~N(0,1)
# Three variables are measured: x1,x2,x3.  All are generated U(0,1)

# We fit 4 models: 
#   0: intercept only   1: X1   2: X1, X2    3: X1, X2, X3
# We then compute the AIC, AICC, and BIC for each model
# We determine which model has the best value for each criterion.
# Finally, we make a histogram of the models.


set.seed(392039853)

reps <- 200 # Number of data sets
N <- 20      # Sample size

# Prepare for looping over reps
counter <- 1
save.ic<- matrix(data=NA, ncol=12, nrow=reps)

# Loop to generate data, analyze, and save results
for(counter in c(1:reps)){
x1 <- runif(n=N)
x2 <- runif(n=N)
x3 <- runif(n=N)
ep <- rnorm(n=N)
y <- x1 + x2 + ep

# Fit model "*" and store object in "reg*"

reg0 <- lm(y~1) # Intercept only
aic0 <- extractAIC(reg0,k=2)[2]
bic0 <- extractAIC(reg0,k=log(N))[2]
aicc0 <- aic0 + 2 * reg0$rank * (reg0$rank + 1) / (N- reg0$rank -1)

reg1 <- lm(y~x1)
aic1 <- extractAIC(reg1,k=2)[2]
bic1 <- extractAIC(reg1,k=log(N))[2]
aicc1 <- aic1 + 2 * reg1$rank * (reg1$rank + 1) / (N- reg1$rank -1)

reg2 <- lm(y~x1 + x2)
aic2 <- extractAIC(reg2,k=2)[2]
bic2 <- extractAIC(reg2,k=log(N))[2]
aicc2 <- aic2 + 2 * reg2$rank * (reg2$rank + 1) / (N- reg2$rank -1)

reg3 <- lm(y~x1 + x2 + x3)
aic3 <- extractAIC(reg3,k=2)[2]
bic3 <- extractAIC(reg3,k=log(N))[2]
aicc3 <- aic3 + 2 * reg3$rank * (reg3$rank + 1) / (N- reg3$rank -1)

save.ic[counter,] <- c(aic0, aic1, aic2, aic3, bic0, bic1, bic2, bic3, aicc0, aicc1, aicc2, aicc3)

}

# For each IC, figure out which column (model) holds the smallest value, and same model numbers
model.aic <- table(max.col(-save.ic[,1:4]) - 1)
model.bic <- table(max.col(-save.ic[,5:8]) - 1)
model.aicc <- table(max.col(-save.ic[,9:12]) - 1)

library(tidyverse)


lazy_fn <- function(N, b_1, b_2){
  # Run all the code from above but write a function to be lazy about it
  # Not necessarily efficient but more digestible
  set.seed(392039853)
  
  reps <- 200 # Number of data sets
  N <- N      # Sample size
  
  # Prepare for looping over reps
  counter <- 1
  save.ic<- matrix(data=NA, ncol=12, nrow=reps)
  
  # Loop to generate data, analyze, and save results
  for(counter in c(1:reps)){
    x1 <- runif(n=N)
    x2 <- runif(n=N)
    x3 <- runif(n=N)
    ep <- rnorm(n=N)
    y <- b_1 * x1 + b_2 * x2 + ep
    
    # Fit model "*" and store object in "reg*"
    
    reg0 <- lm(y~1) # Intercept only
    aic0 <- extractAIC(reg0,k=2)[2]
    bic0 <- extractAIC(reg0,k=log(N))[2]
    aicc0 <- aic0 + 2 * reg0$rank * (reg0$rank + 1) / (N- reg0$rank -1)
    
    reg1 <- lm(y~x1)
    aic1 <- extractAIC(reg1,k=2)[2]
    bic1 <- extractAIC(reg1,k=log(N))[2]
    aicc1 <- aic1 + 2 * reg1$rank * (reg1$rank + 1) / (N- reg1$rank -1)
    
    reg2 <- lm(y~x1 + x2)
    aic2 <- extractAIC(reg2,k=2)[2]
    bic2 <- extractAIC(reg2,k=log(N))[2]
    aicc2 <- aic2 + 2 * reg2$rank * (reg2$rank + 1) / (N- reg2$rank -1)
    
    reg3 <- lm(y~x1 + x2 + x3)
    aic3 <- extractAIC(reg3,k=2)[2]
    bic3 <- extractAIC(reg3,k=log(N))[2]
    aicc3 <- aic3 + 2 * reg3$rank * (reg3$rank + 1) / (N- reg3$rank -1)
    
    save.ic[counter,] <- c(aic0, aic1, aic2, aic3, bic0, bic1, bic2, bic3, aicc0, aicc1, aicc2, aicc3)
  }
  
  testing <- save.ic %>%
    as.data.frame()
  
  names(testing) <- c("aic0", "aic1", "aic2", "aic3", 'bic0', 'bic1', "bic2", 'bic3', 'aicc0', "aicc1", 'aicc2', "aicc3")
  
  testing_2 <- testing %>%
    gather(key = "Criterion", value = "Value") %>%
    group_by(Criterion) %>%
    mutate(rep = row_number(),
           general_method = case_when(str_detect(Criterion, "aic[0-9]$") ~ "aic",
                                      str_detect(Criterion, "bic[0-9]$") ~ "bic",
                                      TRUE ~ "aicc"),
           n_vars = str_extract(Criterion, "[0-9]$")) %>%
    group_by(rep, general_method) %>%
    arrange(Value) %>% # default is ascending
    summarize(n_vars_min = first(n_vars))
  
  # FOr some reason needed to wrap this to force display
  plot(testing_2 %>%
    ggplot(aes(x = n_vars_min)) +
    geom_bar() +
    ggtitle("Comparison of Information Criterion") + xlab("Number of Variables in Selected Model") +
    ylab("Count") +
    facet_wrap(~ general_method) +
    theme_bw())
  
  
  return(testing_2 %>% group_by(general_method, n_vars_min) %>%
           summarize(count_of_min = length(n_vars_min)))
}

# baseline
lazy_fn(20, 1, 1)
# (a)  Increase the sample size to 100.
lazy_fn(100, 1, 1)
# (b)  Decrease the sample size to 10.
lazy_fn(10, 1, 1)
# (c)  Increase β1 to 2.
lazy_fn(20, 2, 1)
# (d)  Decrease β1 to 0.5.
lazy_fn(20, 0.5, 1)
# (e)  Increase β2 to 2.
lazy_fn(20, 1, 2)
# (f)  Decrease β2 to 0.5.
lazy_fn(20, 1, 0.5)
# (g)  Increase both β1 and β2 to 2
lazy_fn(20, 2, 2)