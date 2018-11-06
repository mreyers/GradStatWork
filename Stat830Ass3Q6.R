########### Assignment 3
# Question 6

# e) 5 doses, 40 mice exposed to each, then given suriving number
dose <- c(-3.204, -2.903, -2.602, -2.301, -2.000)
n <- rep(40, n = 5)
live <- c(7, 18, 32, 35, 38)

propLive <- live / n

# Can do prelim estimates of MLE for h(d) against dose via linear regression
hdiHat <- propLive # As MLE for p in a binomial(n, p) is xbar


logodds <- log(hdiHat / (1 - hdiHat))

testLM <- lm(logodds ~ dose)
summary(testLM)

# Based on the model given, my estimates for the parameters are
  # alpha = 10.5322
  # beta = 3.6999

# Now to calculate the large sample confidence limits
alpha <- summary(testLM)$coefficients[1,1]
beta  <- summary(testLM)$coefficients[2,1]
U <- c(-1 / beta, -alpha / (beta^2))
FisherCol1 <- c(sum(n*hdiHat*(1-hdiHat)), sum(n*dose*hdiHat*(1-hdiHat)))
FisherCol2 <- c(sum(n*dose*hdiHat*(1-hdiHat)), sum(n*dose^2*hdiHat*(1-hdiHat)))
Fisher <- cbind(FisherCol1, FisherCol2)

se <- t(U) %*% solve(Fisher) %*% U

bands <- 2 * sqrt(value)

# Actual CI
interval <- c(-alpha / beta - bands, -alpha / beta + bands)
