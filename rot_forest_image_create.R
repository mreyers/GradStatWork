# Show fake data set of tree issues
library(tidyverse)
x1 <- round(rnorm(100, mean = 90, sd = 10))            
x2 <- x1 + rnorm(100, mean = 3, sd = 7)            

fake_frame <- data.frame(pts_home = x1, pts_away = x2) %>%
  mutate(Winner = as.factor(ifelse(pts_home > pts_away, "Home", "Away")))

fake_frame %>%
  ggplot(aes(x = pts_home, y = pts_away)) +
  geom_point(aes(shape = Winner, color = Winner, size = 1.5)) +
  ggtitle("Raw data comparing points scored by home and away basketball teams") +
  theme_bw()

library(rpart)
basic_cart <- rpart(Winner ~ pts_home + pts_away, fake_frame,
                    parms = list(split = 'information'),
                    control = rpart.control(cp = 0))

library(rpart.plot)
prp(basic_cart)

library(klaR)
partimat(Winner ~ pts_home + pts_away, data = fake_frame, method = 'rpart',
         parms = list(split = 'information'),
         control = rpart.control(cp = 0))

# Now with transformed data
fake_frame_x <- prcomp(~ pts_home + pts_away, data = fake_frame)
x <- fake_frame_x$x
y <- fake_frame$Winner

data.frame(x_1 = x[,1], x_2 = x[,2], y = y) %>%
  ggplot(aes(x = x_1, y = x_2), group = y) +
  geom_point(aes(shape = y, color = y)) + theme_bw()

holder <- data.frame(PC1 = x[,1], PC2 = x[,2], Winner = y)
partimat(Winner ~ PC1 + PC2, data = holder, method = 'rpart',
         parms = list(split = 'information'),
         control = rpart.control(cp = 0))

pca_cart <- rpart(Winner ~ PC1 + PC2, holder,
                  parms = list(split = 'information'),
                  control = rpart.control(cp = 0))
prp(pca_cart)


# Now just a basic rotation forest classification for the wheat data set
wheat <- read_csv('wheat.csv') %>%
  select(-id, -class) %>%
  mutate(type = as.factor(ifelse(type %in% 'Healthy', 1, 0)))

train_wheat <- wheat[1:200,]
test_wheat <- wheat[201:275,]

x_train <- train_wheat %>% select(-type)
y_train <- train_wheat %>% pull(type)

x_test <- test_wheat %>% select(-type)
y_test <- test_wheat %>% pull(type)

library(rotationForest)
base_rotf <- rotationForest(x_train, y_train, L = 500)

predictions <- predict(object = base_rotf, newdata = x_test)

predictions_f <- as.data.frame(predictions) %>%
  mutate(type = ifelse(predictions > 0.5, 1, 0))

acc <- mean(predictions_f$type == y_test)


library(randomForest)
quick_comp <- randomForest(type ~ ., data = train_wheat)
rand_preds <- predict(quick_comp, newdata = test_wheat)
# Wonder what performance is like with more correlated data
mean(rand_preds == y_test)
