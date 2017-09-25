rm(list = ls())

#loading libraries
library("data.table")
library("tibble")
library("psych")
library("rpart")
library("randomForest")
library(RPushbullet)
library(caret) #multinom
library(e1071) #multinom


#loading datasets
train <- as.tibble(fread("E:/competitions/drivendata/bloodDonations/train.csv"))
test <- as.tibble(fread("E:/competitions/drivendata/bloodDonations/test.csv"))
sub_samp <- as.tibble(fread("E:/competitions/drivendata/bloodDonations/sub_samp.csv"))


#data profiling
glimpse(train)
summary(train)
pairs.panels(train,scale=2)


#univariate nalysis
hist(train$`Months since Last Donation`)
hist(train$`Number of Donations`)
hist(train$`Total Volume Donated (c.c.)`)
hist(train$`Months since First Donation`)
hist(train$`Made Donation in March 2007`)

#bivariate analysis
plot(train$`Number of Donations`,train$`Total Volume Donated (c.c.)`)


#missing values
colSums(is.na(train))
colSums(is.na(test))


#correlations
cor(train[,c(2,3,4,5,6)])*100
str(train)
new_train <- subset(train,select = c(2,3,5,6))
new_test <- subset(test,select = c(2,3,5))
pairs.panels(new_train)


#baseline logistic model
lm_model <- glm(data = new_train, new_train$`Made Donation in March 2007` ~ . )
summary(lm_model)
lm_model

#baseline prediction
pred <- predict(lm_model, new_test)
pred_df <- data.frame(test$V1, pred)
colnames(pred_df) <- c('V1','Made Donation in March 2007')
write.csv(pred_df, file = "blood_baseline.csv",row.names = FALSE)


#decision tree
dt_model <- rpart(data = new_train,new_train$`Made Donation in March 2007` ~ . )
summary(dt_model)
pred_dt <- predict(dt_model, new_test)
pred_dt_df <- data.frame(test$V1, pred_dt)
colnames(pred_dt_df) <- c('V1','Made Donation in March 2007')
write.csv(pred_dt_df, file = "blood_dt.csv",row.names = FALSE)


#random forest
pred_rg <- predict(rg_model, new_test)
pred_rg_df <- data.frame(test$V1, pred_rg)
colnames(pred_rg_df) <- c('V1','Made Donation in March 2007')
write.csv(pred_rg_df, file = "blood_rg.csv",row.names = FALSE)


####################################################################
#using caret package

#random forest
new_train$`Made Donation in March 2007` <- as.factor(new_train$`Made Donation in March 2007`)
gbm_model <- train(data = new_train,`Made Donation in March 2007` ~ . , method = "gbm")
summary(gbm_model)
pred_gbm <- predict(gbm_model, new_test, type = "prob")
pred_gbm_df <- data.frame(test$V1, pred_gbm)
pred_gbm_df$X0 <- NULL
colnames(pred_gbm_df) <- c('V1','Made Donation in March 2007')
write.csv(pred_gbm_df, file = "blood_gbm2.csv",row.names = FALSE)

