# Stat 830 Ass 4 Q 2

# X1 ,.. , Xn follow N(mu, 1)
# Theta = e^mu, thetahat = e^xbar

# Create a data set using mu = 5, 100 observations
set.seed(1)
X <- rnorm(100) + 5
theta <- exp(5)
thetaHat <- exp(mean(X))

# Parametric bootstrap: Generate thetaHat from the theoretical distribution to assign bounds
n <- 10000
thetaHatDist <- rnorm(n, mean = theta, sd = theta)
thetaIntervalP <- c(mean(thetaHatDist) - 1.96*sd(thetaHatDist) / sqrt(n), mean(thetaHatDist) + 1.96 * sd(thetaHatDist)/sqrt(n))


# Nonparametric bootstrap: Generate new evaluations of thetaHat and describe behaviour accordingly
hatVec <- rep(0, n)
for(i in 1:n){
  newX <- sample(X, size = 100, replace = T)
  hatVec[i] <- exp(mean(newX))
}
  # Now 2.5% quantile and 97.5%
thetaIntervalNP <- quantile(hatVec, prob = c(0.025, 0.975))

rbind(thetaIntervalP, thetaIntervalNP)
# Interval for parametric bootstrap is much tighter, as expected. Sampling from actual distribution allows for more accurate results

# b) plot the histograms
