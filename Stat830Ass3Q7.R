############ Assignment 3

###### Question 7

library(tidyverse)

# Load the data
y <- c(3.547, 1.228, 2.052, 1.556, 2.487, 0.469, 2.707, 0.395,
       0.770, 0.666, 4.242, 1.474, 1.277, 2.519, 0.578, 2.989,
       1.900, 1.422, 3.701, 1.278, 2.820, 0.224, 0.482, 1.426,
       2.146, 2.975, 2.792, 0.846, 3.190, 1.680, 0.686, 1.634,
       0.969, 4.010, 1.792, 1.287, 0.730, 0.849, 2.447, 2.147)

ybar <- mean(y)
yvar <- var(y)

########### d)
# Take a = 1 and find the mle of B 
# Since hat(b) = ybar / a we get the following
hatB <- ybar / 1 # as a = 1, hatB = 1.810

########### e)
# Use method of moments to get tilde(a) and tilde(b)
# E(y) = a b, Var(y) = a b^2
# So tilde(b) = var(y) / E(y) and tilde(a)= E(x) / tilde(b)

tildeB <- yvar / ybar   # 0.6394
tildea <- ybar / tildeB # 2.8303

########### f)
# Perform Newton Raphson for 2 iterations to get MLE, using MoM estimates as starting points

# Generating necessary components
ylogbar <- mean(log(y))

# First iteration
alpha0hat <- tildea
top <- log(alpha0hat/mean(y)) - digamma(alpha0hat) + ylogbar
bottom <- 1 / alpha0hat - trigamma(alpha0hat)

alpha1hat <- alpha0hat - top/bottom

# Second iteration
alpha0hat <- alpha1hat
top2 <- log(alpha0hat/mean(y)) - digamma(alpha0hat) + ylogbar
bottom2 <- 1 / alpha0hat - trigamma(alpha0hat)

alpha2hat <- alpha1hat - top2/bottom2

#Iterations
alpha0hat # 2.830324
alpha1hat # 2.4989 
alpha2hat # 2.5346


# Functions from http://bioops.info/2015/01/gamma-mme-mle/
gamma_MME<-function(x){
  n<-length(x)
  mean_x<-mean(x)
  alpha<-n*(mean_x^2)/sum((x-mean_x)^2)
  beta<-sum((x-mean_x)^2)/n/mean_x
  estimate_MME<-data.frame(alpha,beta)
  return(estimate_MME)
}

gamma_MLE<-function(x){
  n<-length(x)
  mean_x<-mean(x)
  
  # initiate the convergence and alpha value
  converg<-1000
  alpha_prev<-gamma_MME(x)$alpha
  
  # initiate two vectors to store alpha and beta in each step
  alpha_est<-alpha_prev
  beta_est<-mean_x/alpha_prev
  
  # Newton-Raphson
  while(converg>0.0000001){
    #first derivative of alpha_k-1
    der1<-n*log(alpha_prev/mean_x)-n*digamma(alpha_prev)+sum(log(x))
    #second derivative of alpha_k-1
    der2<-n/alpha_prev-n*trigamma(alpha_prev)
    #calculate next alpha
    alpha_next<-alpha_prev-der1/der2
    # get the convergence value
    converg<-abs(alpha_next-alpha_prev)
    # store estimators in each step
    alpha_est<-c(alpha_est, alpha_next)
    beta_est<-c(beta_est, mean_x/alpha_next)
    # go to next alpha
    alpha_prev<-alpha_next
  }
  
  alpha<-alpha_next
  beta<-mean_x/alpha_next
  estimate_MLE<-data.frame(alpha,beta)
  
  return(estimate_MLE)
}




# Now for beta
beta0 <- tildeB
alphaMLE <- gamma_MLE(y)$alpha
betaMLE <- gamma_MLE(y)$beta

# First iteration
topB <- -length(y)*alphaMLE/beta0 + sum(y)/(beta0^2)
bottomB <- length(y)*alphaMLE/(beta0^2) - 2*sum(y)/(beta0^3)

beta1hat <- beta0 - topB/bottomB

# Second iteration
topB2 <- -length(y)*alphaMLE/beta1hat + sum(y)/(beta1hat^2) 
bottomB2 <- length(y)*alphaMLE/(beta1hat^2) - 2*sum(y)/(beta1hat^3)

beta2hat <- beta1hat - topB2/bottomB2

#Iterations
beta0     # 0.6394
beta2hat  # 0.6998
beta1hat  # 0.7133


############ g)
# Use Fisher's scoring and redo part f)
fisherAlpha <- length(y) * trigamma(tildea)
fisherBeta <- length(y) * tildea / (tildeB^2)

# Alpha, something not working
alpha0Fish <- tildea
topAFish <- log(alpha0Fish/mean(y)) - digamma(alpha0Fish) + ylogbar
bottomAFish <- fisherAlpha - 1 / tildea

alpha1Fish <- alpha0Fish + topAFish / bottomAFish
topA2Fish <- log(alpha1Fish/mean(y)) - digamma(alpha1Fish) + ylogbar
alpha2Fish <- alpha1Fish + topA2Fish / bottomAFish

# Iterations:
alpha0Fish # 2.8303
alpha1Fish # 2.8289
alpha2Fish # 2.8275


# Beta
beta0Fish <- tildeB
topBFish <- -length(y)*alphaMLE/beta0Fish + sum(y)/(beta0Fish^2)
bottomBFish <- length(y) * alphaMLE / (beta0Fish^2)

beta1Fish <- beta0Fish + topBFish / bottomBFish
topB2Fish <- -length(y)*alphaMLE/beta1Fish + sum(y)/(beta1Fish^2)
beta2Fish <- beta1Fish + topB2Fish / bottomBFish

#Iterations:
beta0Fish #0.6394
beta1Fish # 0.7138
beta2Fish # 0.7138


############## h)
# Compute standard errors for the MLEs

# Will do via bootstrap
library(foreach)
library(doSNOW)
cl <- makeCluster(6)
registerDoSNOW(cl)


mleEst <- data.frame(alpha = alphaMLE, beta = betaMLE)
names(mleEst) <- c("alpha", "beta")
for(i in 1:100000){
  newSamp <- sample(y, size = length(y), replace = TRUE)
  temp <- gamma_MLE(newSamp)
  mleEst[i,1] <- temp$alpha
  mleEst[i,2] <- temp$beta
}

mleEst <- foreach(i= 1:500000, .combine=rbind) %dopar% {
  newSamp <- sample(y, size = length(y), replace = TRUE)
  mleEst <- gamma_MLE(newSamp)

  
}

stopCluster(cl)
sd(mleEst[, 1]) # 0.51557
sd(mleEst[, 2]) # 0.1250


