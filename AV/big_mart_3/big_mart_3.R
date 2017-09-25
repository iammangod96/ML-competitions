rm(list = ls())


#LOADING DATASETS
train <- read.csv("C:/Users/manish.sharma/Downloads/AV/big_mart_3/train.csv")
test <- read.csv("C:/Users/manish.sharma/Downloads/AV/big_mart_3/test.csv")


#loading libraries
library("psych")
library(ggplot2)
library("mlr")
library("car")
library("rpart")
library("C50")
library("randomForest")
library("gbm")


#data profiling
str(train)
summary(train)
#pairs.panels(train)
cor(train[,c(2,4,6,8,12)])*100
plot(Item_Outlet_Sales ~ Item_MRP, data = train)
abline(lm(Item_Outlet_Sales ~ Item_MRP, data = train),col = "red", pch = 2)
hist(train$Outlet_Establishment_Year)

#treating missing values
colSums(is.na(train))
colSums(is.na(test))

train$Outlet_Size <- ifelse(train$Outlet_Size == "", NA, train$Outlet_Size)
test$Outlet_Size <- ifelse(test$Outlet_Size == "", NA, test$Outlet_Size)
imputed_data_train <- impute(train, cols = list(Outlet_Size = imputeMode(), Item_Weight = imputeMean()))
train <- imputed_data_train$data
imputed_data_test <- impute(test, cols = list(Outlet_Size = imputeMode(), Item_Weight = imputeMean()))
test <- imputed_data_test$data


#treating outliers
ggplot(train, aes(Item_MRP,Item_Outlet_Sales)) + geom_jitter()


#variable transformation
str(train)

table(train$Item_Fat_Content)
train$Item_Fat_Content <- recode(train$Item_Fat_Content, "c('LF','low fat','Low Fat') = 'Low Fat'; 'reg' = 'Regular'")
test$Item_Fat_Content <- recode(test$Item_Fat_Content, "c('LF','low fat','Low Fat') = 'Low Fat'; 'reg' = 'Regular'")


#training a linear model
lr_model <- lm(data = train, Item_Outlet_Sales ~ . -Item_Identifier -Outlet_Identifier)
summary(lr_model)


#predictiction of linear model
prediction_test <- predict(lr_model , newdata = test)
sol_DF <- data.frame(test$Item_Identifier, test$Outlet_Identifier, prediction_test)
colnames(sol_DF) <- c('Item_Identifier','Outlet_Identifier','Item_Outlet_Sales')
write.csv(sol_DF, file = "BM3_solution.csv", row.names = FALSE)


#training decision tree
fit <- rpart(data = train, Item_Outlet_Sales ~ Item_Weight+Item_Fat_Content+Item_Visibility+Item_Type + Item_MRP+  Outlet_Establishment_Year+ Outlet_Size + Outlet_Location_Type+ Outlet_Type, method = "anova")
summary(fit)


#prediction using decision tree
prediction_DT <- predict(fit, newdata = test)
sol_DT_DF <- data.frame(test$Item_Identifier, test$Outlet_Identifier, prediction_DT)
colnames(sol_DT_DF) <- c('Item_Identifier','Outlet_Identifier','Item_Outlet_Sales')
write.csv(sol_DT_DF, file = "BM3_DT_solution.csv", row.names = FALSE)


#training random forest
fit_RF <- randomForest(data = train, Item_Outlet_Sales ~ Item_Weight+Item_Fat_Content+Item_Visibility+Item_Type + Item_MRP+  Outlet_Establishment_Year+ Outlet_Size + Outlet_Location_Type+ Outlet_Type, method = "anova", ntree = 500)
summary(fit_RF)


#prediction using random forest
prediction_RF <- predict(fit_RF, newdata = test)
sol_RF_DF <- data.frame(test$Item_Identifier, test$Outlet_Identifier, prediction_RF)
colnames(sol_RF_DF) <- c('Item_Identifier','Outlet_Identifier','Item_Outlet_Sales')
write.csv(sol_RF_DF, file = "BM3_RF_solution.csv", row.names = FALSE)


#training GBM
fit_GBM <- gbm(data = train, Item_Outlet_Sales ~ Item_Weight+Item_Fat_Content+Item_Visibility+Item_Type + Item_MRP+  Outlet_Establishment_Year+ Outlet_Size + Outlet_Location_Type+ Outlet_Type, n.trees = 2500)
summary(fit_GBM)


#prediction using random forest
prediction_GBM <- predict.gbm(fit_GBM, newdata = test, n.trees = 1500)
sol_GBM_DF <- data.frame(test$Item_Identifier, test$Outlet_Identifier, prediction_GBM)
colnames(sol_GBM_DF) <- c('Item_Identifier','Outlet_Identifier','Item_Outlet_Sales')
write.csv(sol_GBM_DF, file = "BM3_GBM_solution.csv", row.names = FALSE)


#training decision tree
fit_c50 <- loess(data = train, Item_Outlet_Sales ~ Item_Weight+Item_Fat_Content+Item_Visibility+Item_Type + Item_MRP+  Outlet_Establishment_Year+ Outlet_Size + Outlet_Location_Type+ Outlet_Type)
summary(fit_c50)


#prediction using decision tree
prediction_DT <- predict(fit, newdata = test)
sol_DT_DF <- data.frame(test$Item_Identifier, test$Outlet_Identifier, prediction_DT)
colnames(sol_DT_DF) <- c('Item_Identifier','Outlet_Identifier','Item_Outlet_Sales')
write.csv(sol_DT_DF, file = "BM3_DT_solution.csv", row.names = FALSE)
