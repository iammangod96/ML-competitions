rm(list = ls())


#LOADING DATASETS
train <- read.csv("C:/Users/manish.sharma/Downloads/AV/house_prices/train.csv")
test <- read.csv("C:/Users/manish.sharma/Downloads/AV/house_prices/test.csv")
test1 <- read.csv("C:/Users/manish.sharma/Downloads/AV/house_prices/test.csv")


#loading libraries
library("mlr") #impute
library("corrplot") #corrplot
library("glmnet") #glmnet
library("randomForest")
library("rpart")
library("lars")
library("gbm")


#data profiling
str(train)
cor(train$SalePrice, train$OpenPorchSF*train$WoodDeckSF)
summary(train)


#treating missing values
sort(colSums(is.na(train)), decreasing = TRUE)
train <- subset(train,select = -c(LotFrontage,PoolQC,MiscFeature,Alley,Fence,FireplaceQu))
test <- subset(test,select = -c(LotFrontage,PoolQC,MiscFeature,Alley,Fence,FireplaceQu))

imputed_train_data <- impute(train, cols = 
  list(
  Electrical = imputeMode(), 
  MasVnrArea = imputeMedian(),
  MasVnrType = imputeMode(),
  BsmtFinType1 = imputeMode(),
  BsmtFinType2 = imputeMode(),
  BsmtCond = imputeMode(),
  BsmtQual = imputeMode(),
  BsmtExposure = imputeMode(),
  GarageCond = imputeMode(),
  GarageQual = imputeMode(),
  GarageFinish = imputeMode(),
  #GarageYrBlt = 
  GarageType = imputeMode()
  )
)

train <- imputed_train_data$data
train$GarageYrBlt[is.na(train$GarageYrBlt)] <- sample(1900:2010, size = sum(is.na(train$GarageYrBlt)), replace = F )

sort(colSums(is.na(test)), decreasing = TRUE)
imputed_test_data <- impute(test, cols = 
                         list(
                           MasVnrArea = imputeMedian(),
                           MasVnrType = imputeMode(),
                           BsmtFinType1 = imputeMode(),
                           BsmtFinType2 = imputeMode(),
                           BsmtCond = imputeMode(),
                           BsmtQual = imputeMode(),
                           BsmtExposure = imputeMode(),
                           GarageCond = imputeMode(),
                           GarageQual = imputeMode(),
                           GarageFinish = imputeMode(),
                           #GarageYrBlt = 
                           GarageType = imputeMode(),
                           MSZoning = imputeMode(),
                           Utilities = imputeMode(),
                           BsmtFullBath = imputeMedian(),
                           Functional = imputeMode(),
                           Exterior1st = imputeMode(),
                           Exterior2nd = imputeMode(),
                           BsmtFinSF1 = imputeMedian(),
                           BsmtUnfSF = imputeMedian(),
                           TotalBsmtSF = imputeMedian(),
                           KitchenQual = imputeMode(),
                           GarageCars = imputeMode(),
                           GarageArea = imputeMedian(),
                           SaleType = imputeMode()
                         )
)

test <- imputed_test_data$data
test$GarageYrBlt[is.na(test$GarageYrBlt)] <- sample(1900:2010, size = sum(is.na(test$GarageYrBlt)), replace = F )


#data profiling after treating missing values
nums <- sapply(train, is.numeric)
cr <- cor(train[,nums])
par(mfrow = c(1,1))
corrplot(cr)


#deciding predictors and target
outcome <- 75
not_predictors <- c(1:2,4,17,35,44,47,51,64:72)
train <- subset(train,select = -not_predictors)
test <- subset(test,select = -not_predictors)


#elasticnet
#fit_elnet <- glmnet(as.matrix(train[,1:57]), train$SalePrice, family="gaussian", alpha=.5)


#linear reg
model_lm <- lm(data = train, SalePrice ~ .)
predicted_lm <- predict(model_lm, newdata = test)
sol_DF <- data.frame(test1$Id,predicted_lm)
colnames(sol_DF) <- c("Id","SalePrice")
imputed_sol <- impute(sol_DF, cols = list(SalePrice = imputeMean()))
sol_DF <- imputed_sol$data
write.csv(sol_DF, file = "HousePrices_lm.csv", row.names = FALSE)


#decision tree model
model_DT <- rpart(data = train, SalePrice ~ .)
summary(model_DT)
predicted_DT <- predict(model_DT, newdata = test)
sol_DT_DF <- data.frame(test1$Id,predicted_DT)
colnames(sol_DT_DF) <- c("Id","SalePrice")
write.csv(sol_DT_DF, file = "HousePrices_DT.csv", row.names = FALSE)


#random forest model
model_RF <- randomForest(data = train, SalePrice ~ ., ntree= 500)
summary(model_RF)
predicted_RF <- predict(model_RF, newdata = test)
sol_RF_DF <- data.frame(test1$Id,predicted_RF)
colnames(sol_RF_DF) <- c("Id","SalePrice")
write.csv(sol_RF_DF, file = "HousePrices_RF.csv", row.names = FALSE)

#lasso
#training lasso model
model_lasso <- lars(x = as.matrix(train[,-58]),y =  train[,58], type = "lasso", trace = T)
summary(lasso_model)


#gbm
model_gbm <- gbm(data = train, SalePrice ~ ., n.trees = 5000)
summary(model_gbm)
predicted_gbm <- predict(model_gbm, newdata = test, n.trees = 2500)
sol_gbm_DF <- data.frame(test1$Id,predicted_gbm)
colnames(sol_gbm_DF) <- c("Id","SalePrice")
write.csv(sol_gbm_DF, file = "HousePrices_gbm.csv", row.names = FALSE)
