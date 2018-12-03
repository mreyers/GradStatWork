# Stat 830 Ass 4 Q 2

# X1 ,.. , Xn follow N(mu, 1)
# Theta = e^mu, thetahat = e^xbar
library(tidyverse)
# Create a data set using mu = 5, 100 observations
set.seed(1)
X <- rnorm(100) + 5
theta <- exp(5)
thetaHat <- exp(mean(X))

B <- 10000

# Bootstrap
n <- 100
thetaHatEst <- rep(0, B)
hatVec <- rep(0, B)
for(i in 1:B){
  # Parametric
  xhatDist <- rnorm(n, mean = mean(X), sd = sd(X))
  thetaHatEst[i] <- exp(mean(xhatDist))
  
  # Non-Parametric
  newX <- sample(X, size = 100, replace = T)
  hatVec[i] <- exp(mean(newX))
}

quantile(thetaHatEst, prob = c(0.025, 0.975))
quantile(hatVec, prob = c(0.025, 0.975))


# b) plot the histograms
ggplot() +
  theme_bw() +
  geom_density(aes(thetaHatEst, colour = "Parametric")) +
  geom_density(aes(hatVec, colour = "Non-parametric")) +
  stat_function(aes(x = X, colour = "Delta Method"), fun = dnorm, n = 1000, args = c(mean = exp(mean(X)), # Add in Delta Function to plotting
                                                                                     sd = exp(mean(X)) / sqrt(100))) +
  scale_x_continuous(limits = c(100, 225), expand = c(0, 0)) +
  stat_function(aes(x = X, colour = "True Distribution"), fun = dnorm, n = 1000,
                args = c(mean = theta, sd = theta / sqrt(100))) +
  ggtitle("Comparison of bootstrap methods to actual distribution") 

# Based on the plots and my given seed for randomization, I find the parametric bootstrap to be most closely 
# related to the true distribution in terms of shape and proximity to the center.


