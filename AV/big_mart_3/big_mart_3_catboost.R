rm(list = ls())


#LOADING DATASETS
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test


#loading libraries
library("xgboost")


#data profiling
str(train)
str(test)
dim(train$data)
dim(test$data)
class(train$data)
class(train$label)


#basic training
bstSparse <- xgboost(data = train$data, label = train$label, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose  = 0)
bstDense <- xgboost(data = as.matrix(train$data), label = train$label, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose = 1)

dtrain <- xgb.DMatrix(data = train$data, label = train$label)
bstDMatrix <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose = 2)


#prediction
pred <- predict(bstDMatrix, test$data)
print(length(pred))
head(pred)
prediction <- as.numeric(pred > 0.5)
head(prediction)


#evaluation
err <- mean(prediction != test$label)
err*100

