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
model_sizes <- prostate_tidy %>% slice(1) %>% pull(summ_results) %>% extract2(1) %>% extract('which') 
prostate_models <- prostate_tidy %>%
  group_by(set) %>%
  mutate(n_vars = list(0:8)) %>% 
  unnest(.preserve = c(data, regsubs, summ_results)) %>%
  mutate(model_sizes = list(as.matrix(model_sizes$which)),
         reg_results = pmap(list(data, n_vars, model_sizes), ~ basic_reg(..1, ..2, ..3)))

results1 <- matrix(data=NA, nrow=9, ncol=4)
mod1 <- lm(lpsa ~ 1, data=prostate[which(prostate$set==1),])
sMSE <- summary(mod1)$sigma^2
BIC <- extractAIC(mod1, k=log(nrow(prostate[which(prostate$set==1),])))
pred2 <- predict(mod1, newdata=prostate[which(prostate$set==2),])
MSPE <- mean((pred2-prostate[which(prostate$set==2),]$lpsa)^2)
results1[1,] <- c(0, sMSE, BIC[2], MSPE)

#Get rid of superfluous variables so that I can call the right variables into the data set each time.
# Also move response to 1st column to be included every time below.
prostate2 <- prostate[,c(10,2:9)]


for(v in 1:8){
  mod1 <- lm(lpsa ~ ., data=prostate2[which(prostate$set==1), summ.1$which[v,]])
  sMSE <- summary(mod1)$sigma^2
  BIC <- extractAIC(mod1, k=log(nrow(prostate2[which(prostate$set==1),])))
  pred2 <- predict(mod1, newdata=prostate2[which(prostate$set==2),])
  MSPE <- mean((pred2-prostate2[which(prostate$set==2),]$lpsa)^2)
  results1[v+1,] <- c(v, sMSE, BIC[2], MSPE)
}

results1


# All 3 plots together
x11(width=10,height=5,pointsize=18)
par(mfrow=c(1,3))
plot(x=results1[,1], y=results1[,2], xlab="Vars in model", ylab="sample-MSE",
     main="SampleMSE vs Vars: 1st", type="b")
plot(x=results1[,1], y=results1[,3], xlab="Vars in model", ylab="BIC",
     main="BIC vs Vars: 1st", type="b")
plot(x=results1[,1], y=results1[,4], xlab="Vars in model", ylab="MSPE",
     main="MSPE vs Vars: 1st", type="b")

##########
# Repeat for second data set


# Fitting the models in succession from smallest to largest.  
# Fit one-var model. then update to 2-var model.  Could keep going.
# Each time computing sample-MSE (sMSE), BIC, and mean squared pred. error (MSPE). 

results2 <- matrix(data=NA, nrow=9, ncol=4)
mod1 <- lm(lpsa ~ 1, data=prostate[which(prostate$set==2),])
sMSE <- summary(mod1)$sigma^2
BIC <- extractAIC(mod1, k=log(nrow(prostate[which(prostate$set==2),])))
pred2 <- predict(mod1, newdata=prostate[which(prostate$set==1),])
MSPE <- mean((pred2-prostate[which(prostate$set==1),]$lpsa)^2)
results2[1,] <- c(0, sMSE, BIC[2], MSPE)

#Get rid of superfluous variables so that I can call the right variables into the data set each time.
# Also move response to 1st column to be included every time below.
prostate2 <- prostate[,c(10,2:9)]


for(v in 1:8){
  mod1 <- lm(lpsa ~ ., data=prostate2[which(prostate$set==2), summ.2$which[v,]])
  sMSE <- summary(mod1)$sigma^2
  BIC <- extractAIC(mod1, k=log(nrow(prostate2[which(prostate$set==2),])))
  pred2 <- predict(mod1, newdata=prostate2[which(prostate$set==1),])
  MSPE <- mean((pred2-prostate2[which(prostate$set==1),]$lpsa)^2)
  results2[v+1,] <- c(v, sMSE, BIC[2], MSPE)
}

results2


# All 3 plots together
x11(width=10,height=5,pointsize=18)
par(mfrow=c(1,3))
plot(x=results2[,1], y=results2[,2], xlab="Vars in model", ylab="sample-MSE",
     main="SampleMSE vs Vars: 2nd", type="b")
plot(x=results2[,1], y=results2[,3], xlab="Vars in model", ylab="BIC",
     main="BIC vs Vars: 2nd", type="b")
plot(x=results2[,1], y=results2[,4], xlab="Vars in model", ylab="MSPE",
     main="MSPE vs Vars: 2nd", type="b")

