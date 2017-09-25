rm(list = ls())

#loading libraries
library(ggplot2)
library("lubridate")
library("dplyr")
library(caret)
library(psych)
library(rpart)
library(randomForest)
library(xgboost)
library(RPushbullet)


#loading datasets
train <- read.csv("E:/competitions/AV/time_series_2/train.csv")
test <- read.csv("E:/competitions/AV/time_series_2/test.csv")
samp_sub <- read.csv("E:/competitions/AV/time_series_2/samp_sub.csv")

train$Datetime = dmy_hm(train$Datetime)
test$Datetime = dmy_hm(test$Datetime)


#data profiling
train_ts <- ts(train)
plot.ts(train_ts)
start(train)
start(train_ts)
end(train_ts)
frequency(train_ts)
ggplot(data = train, aes(Datetime, Count)) +  geom_line()


#converting datetime object to different columns
train$wday <- wday(train$Datetime)
train$month <- month(train$Datetime)
train$year <- year(train$Datetime)
train$hour <- hour(train$Datetime)
train$Datetime <- NULL
str(train)
train <- train %>% select(ID,hour,wday,month,year,Count)

summary(train)
pairs.panels(train[,-1])

test$wday <- wday(test$Datetime)
test$month <- month(test$Datetime)
test$year <- year(test$Datetime)
test$hour <- hour(test$Datetime)
test$Datetime <- NULL
str(test)
test <- test %>% select(ID,hour,wday,month,year)

#linear regression model
lm_model <- lm(data = train, Count ~ . - ID)
summary(lm_model)
lm_model2 <- train(data = train, Count ~ . - ID, method="lm")
summary(lm_model2)

#prediction
pred<- predict(lm_model2, newdata = test)
pred <- round(pred)
pred_df <- data.frame(ID = test$ID, Count = pred)
write.csv(pred_df, file = 'time_series_2_baseline.csv', row.names = FALSE)


#decision tree model
dt_model <- rpart(data = train, Count ~ . - ID, method = "anova")
summary(dt_model)
pred_dt<- predict(dt_model, newdata = test)
pred_dt <- round(pred_dt)
pred_dt_df <- data.frame(ID = test$ID, Count = pred_dt)
write.csv(pred_dt_df, file = 'time_series_2_dt.csv', row.names = FALSE)


#xgboost
#tune settings
xgb_grid <- expand.grid(
  nrounds = c(250,500,1000),
  max_depth = c(1,2,4),
  eta = c(0.001, 0.003, 0.01),
  gamma = c(0,1,2),
  min_child_weight = c(1, 2),
  colsample_bytree = c(1,0.5,0.25),
  subsample = 1
)

xgb_custom_grid <- expand.grid(
  nrounds = 1000,
  max_depth = c(9,12,15),
  eta = 0.05,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 2,
  subsample = c(0.6,0.8,1)
)

grid_ix <- sample(1:nrow(xgb_grid), 100)
xgb_grid1 <- xgb_grid[grid_ix,]

xgb_train_control <- trainControl(
  method = "repeatedcv",
  number  = 5,
  repeats = 2,
  verboseIter = FALSE,
  returnData = FALSE,
  allowParallel = TRUE
)

xgb_train <- train(
  x = as.matrix(train[,!names(train) %in% c("Count")]),
  y = train$Count,
  objective = "reg:linear",
  trControl = xgb_train_control,
  tuneGrid = xgb_grid,
  method = "xgbTree"
)

xgb_train_custom <- train(
  x = as.matrix(train[,!names(train) %in% c("Count")]),
  y = train$Count,
  objective = "reg:linear",
  trControl = xgb_train_control,
  tuneGrid = xgb_custom_grid,
  method = "xgbTree"
)

xgb.save(xgb_train_custom, "time_series_2_xgb_custom_full_2")
pbPost("note","xgb_train_custom done","xgboost run completed")
xgb_train_custom

xgb.save(xgb_train, "time_series_2_xgb_full_model")
xgb_train

#predicting
pred_xgb <- predict(xgb_train, newdata = test)
pred_xgb <- round(pred_xgb)
pred_xgb_df <- data.frame(ID = test$ID, Count = pred_xgb)
write.csv(pred_xgb_df, file = 'time_series_2_xgb_full.csv', row.names = FALSE)


#predicting xgb_custom
pred_xgb_custom <- predict(xgb_train_custom, newdata = test)
pred_xgb_custom <- round(pred_xgb_custom)
pred_xgb_custom_df <- data.frame(ID = test$ID, Count = pred_xgb_custom)
write.csv(pred_xgb_custom_df, file = 'time_series_2_xgb_custom_full3.csv', row.names = FALSE)




#averaging linear regression and xgb predictions
pred_comb <- ((2*pred)+pred_xgb)/2
pred_comb <- round(pred_comb)
pred_comb_df <- data.frame(ID = test$ID, Count = pred_comb)
write.csv(pred_comb_df, file = 'time_series_2_comb1.csv', row.names = FALSE)


#regression vs just ID
plot(train$ID, train$Count)
fit2 <- lm(data = train,Count ~ poly(ID, 2))
summary(fit2)
lines(train$ID, fitted(fit2), col='red', type='b', lwd=1) 
pred_lm2 <- predict(fit2, newdata = test)
pred_lm2 <- round(pred_lm2)
pred_lm2_df <- data.frame(ID = test$ID, Count = pred_lm2)
write.csv(pred_lm2_df, file = 'time_series_2_lm2.csv', row.names = FALSE)


#averaging lm2 and xgb predictions
pred_comb2 <- ((1.2*pred_lm2)+pred_xgb)/2
pred_comb2 <- round(pred_comb2)
pred_comb2_df <- data.frame(ID = test$ID, Count = pred_comb2)
write.csv(pred_comb2_df, file = 'time_series_2_comb2.csv', row.names = FALSE)
