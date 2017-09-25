rm(list = ls())
set.seed(0)


#loading libraries
library(data.table)
library(corrplot)
library(glmnet)
library(RPushbullet)
library(caret)


#loading datasets
train <- as.data.frame(fread("E:/competitions/numerai/numerai_training_data.csv", head=T))
test <- as.data.frame(fread("E:/competitions/numerai/numerai_tournament_data.csv", head=T))
samp_sub <- as.data.frame(fread("E:/competitions/numerai/example_predictions.csv", head=T))


train <- subset(train, select = -c(1,2,3))
new_test <- subset(test, select = -c(1,2,3))
comb <- rbind(train,new_test)
comp_test <- subset(new_test, select = -c(22))


#baseline logistic model
lm_model <- glm(data = comb, comb$target ~ .)
summary(lm_model)
lm_model
lm_model1 <- train(data = new_comb, target ~ ., method = "regLogistic")

#prediction using baseline model
pred <- predict(lm_model, newdata = comp_test)
pred_df <- data.frame(test$id, pred)
colnames(pred_df) <- c('id','probability')
write.csv(pred_df, file = "numerai_baseline.csv", row.names = FALSE)

#check correlation
corrplot(cor(train),type="lower")


#step_lm_model
step_lm_model <- step(lm_model)
summary(step_lm_model)
pred_step <- predict(step_lm_model, newdata = comp_test)
pred_step_df <- data.frame(test$id, pred_step)
colnames(pred_step_df) <- c('id','probability')
write.csv(pred_step_df, file = "numerai_step.csv", row.names = FALSE)


#lasso
new_comb <- comb[!is.na(comb$target),]
predictors <- c(1:14,16:21)
target <- 22
cv_lasso_model <- cv.glmnet(as.matrix(new_comb[,predictors]), new_comb$target)
summary(cv_lasso_model)
plot(cv_lasso_model)
cv_lasso_model$lambda.min
comp_test_temp <- subset(comp_test,select = -c(15))
pred_lasso <- predict(cv_lasso_model, newx = as.matrix(comp_test), s = "lambda.min")
pred_lasso_df <- data.frame(test$id, pred_lasso)
colnames(pred_lasso_df) <- c('id','probability')
write.csv(pred_lasso_df, file = "numerai_lasso1.csv", row.names = FALSE)
