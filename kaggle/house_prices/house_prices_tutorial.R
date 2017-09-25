rm(list = ls())


#LOADING DATASETS
train <- read.csv("C:/Users/manish.sharma/Downloads/AV/house_prices/train.csv")
test <- read.csv("C:/Users/manish.sharma/Downloads/AV/house_prices/test.csv")


#loading libraries
library("lars") #lars
library("ggplot2") #ggplot
library("caret") #createDataPartition
library("hydroGOF") #rmse
library("randomForest") #randomForest
library("xgboost")


#cleansing train data
train <- subset(train,select = -c(PoolQC,MiscFeature,Alley,Fence))

num <- sapply(train, is.numeric)
num <- train[,num]

for(i in 1:77)
{
  if(is.factor(train[,i]))
  {
    train[,i] <- as.integer(train[,i])
  }
}

train[is.na(train)] <- 0
num[is.na(num)] <- 0
ggplot(train, aes(x = YearBuilt, y = SalePrice))+geom_point() + geom_smooth()


#cleansing test data
test <- subset(test,select = -c(PoolQC,MiscFeature,Alley,Fence))

for(i in 1:76)
{
  if(is.factor(test[,i]))
  {
    test[,i] <- as.integer(test[,i])
  }
}

test[is.na(test)] <- 0




#splitting training and testing sets
index <- createDataPartition(train$SalePrice, p=0.75, list=FALSE)
train_set <- train[index,]
test_set <- train[-index,]


#linear regression
model_lm <- lm(SalePrice ~ .,train_set)
summary(model_lm)


#step regression
model_step <- step(model_lm, direction = "backward", trace = TRUE)
summary(model_step)
model_step_both <- step(model_lm, direction = "both", trace = TRUE)
summary(model_step_both)


#lasso regression
predictors <- as.matrix(train_set[,1:76])
outcome <- as.matrix(train_set[,77])
model_lasso <- lars(predictors,outcome,  type = 'lasso')
plot(model_lasso)
summary(model_lasso)
best_step <- model_lasso[which.min(model_lasso$Cp)]


#random forest
model_RF <- randomForest(data = train_set,SalePrice ~ .)


#xgboost
train_mat<- as.matrix(train, rownames.force=NA)
test_mat<- as.matrix(test, rownames.force=NA)
train_mat <- as(train_mat, "sparseMatrix")
test_mat <- as(test_mat, "sparseMatrix")
# Never forget to exclude objective variable in 'data option'
train_Data <- xgb.DMatrix(data = train_mat[,2:76], label = train_mat[,"SalePrice"])

param<-list(
  objective = "reg:linear",
  eval_metric = "rmse",
  booster = "gbtree",
  max_depth = 8,
  eta = 0.123,
  gamma = 0.0385, 
  subsample = 0.734,
  colsample_bytree = 0.512
)
model_xgb <- xgb.train(
              params = param,
              data = train_Data,
              nrounds = 600,
              watchlist = list(train = train_Data),
              verbose = TRUE,
              print_every_n = 50,
              nthread = 6
            )

test_data <- xgb.DMatrix(data = test_mat[,2:76])

#prediction
predict_lm <- predict(model_lm, newdata = test_set)
predict_step_both <- predict(model_step_both, newdata = test_set)
predict_step <- predict(model_step, newdata = test_set)
#predict_lasso <- predict.lars(model_lasso, newx = as.matrix(test_set[,1:76]), s = best_step, type = 'fit')
predict_RF <- predict(model_RF, newdata = test_set)
predict_xgb <- predict(model_xgb, test_data)




#rmse calculation of different models
rmse(log(test_set$SalePrice), log(predict_lm))
rmse(log(test_set$SalePrice), log(predict_step_both))
rmse(log(test_set$SalePrice), log(predict_step))
#rmse(log(test_set$SalePrice), log(predict_lasso))
rmse(log(test_set$SalePrice), log(predict_RF))



#final choosen model - randomForest
model_RF_final <- randomForest(data = train,SalePrice ~ .)
predict_RF_final <- predict(model_RF_final, newdata = test)
sol_RF_DF <- data.frame(test$Id,predict_RF_final)
colnames(sol_RF_DF) <- c("Id","SalePrice")
write.csv(sol_RF_DF, file = "HousePrices_RF_final.csv", row.names = FALSE)


#final choosen model - xgb
sol_xgb_DF <- data.frame(test$Id,predict_xgb)
colnames(sol_xgb_DF) <- c("Id","SalePrice")
write.csv(sol_xgb_DF, file = "HousePrices_xgb.csv", row.names = FALSE)
