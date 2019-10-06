# Lecture 5 stuff

library(datasets)
library(tidyverse)
library(splines)

plot(UKDriverDeaths)

data.frame(Y=as.matrix(dat), date=time(dat))
mod_df <- data.frame(deaths = as.matrix(UKDriverDeaths), date = time(UKDriverDeaths))

# 1. Plot. Comment on trends
mod_df %>%
  ggplot(aes(x = date, y = deaths)) +
  geom_point() + geom_line() +
  ggtitle("Deaths by Month of Year in British Auto Accidents") +
  xlab("Month and Year") + ylab("Deaths") +
  theme_bw()

# The cyclic behavior of driving deaths is evident in the data set. It appears to peak in the Summer months
# and decline towards the Winter months. In terms of overall trend, there seems to be an immediate reduction
# in Auto Accident Deaths leading into 1984. I am unsure as to whether this change is real or not as it seems 
# driving deaths are climbing back up towards the end of 1985. Perhaps it is coincidence, though I expect not.

# 2. Fit natural splines with varying DF until fit is good. Comment on the process
basic_mod <- lm(deaths ~ ns(date, df = 1), data = mod_df)
three_mod <- lm(deaths ~ ns(date, df = 3), data = mod_df)
five_mod <- lm(deaths ~ ns(date, df = 5), data = mod_df)
seven_mod <- lm(deaths ~ ns(date, df = 7), data = mod_df)
nine_mod <- lm(deaths ~ ns(date, df = 9), data = mod_df)
sixteen_mod <- lm(deaths ~ ns(date, df = 16), data = mod_df)
thirtytwo_mod <- lm(deaths ~ ns(date, df = 32), data = mod_df)

mod_df %>%
  mutate(one_df = basic_mod$fitted.values,
         three_df = three_mod$fitted.values,
         five_df = five_mod$fitted.values,
         seven_df = seven_mod$fitted.values,
         nine_df = nine_mod$fitted.values,
         sixteen_df = sixteen_mod$fitted.values,
         thirtytwo_df = thirtytwo_mod$fitted.values) %>%
  ggplot(aes(x = date, y = deaths)) +
  geom_point() + geom_line() +
  geom_line(aes(y = one_df, col = '1 DF'), size = 1.5) +
  geom_line(aes(y = three_df, col = '3 DF'), size = 1.5) +
  geom_line(aes(y = five_df, col = '5 DF'), size = 1.5) +
  geom_line(aes(y = seven_df, col = '7 DF'), size = 1.5) +
  geom_line(aes(y = nine_df, col = '9 DF'), size = 1.5) +
  geom_line(aes(y = sixteen_df, col = '16 DF'), size = 1.5) +
  geom_line(aes(y = thirtytwo_df, col = '32 DF'), size = 1.5) +
  theme_bw()
