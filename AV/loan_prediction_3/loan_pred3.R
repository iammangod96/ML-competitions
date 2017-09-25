rm(list = ls())


#LOADING DATASETS
train <- read.csv("E:/AV/loan_prediction_3/train.csv")
test <- read.csv("E://AV/loan_prediction_3/test.csv")


#loading libraries
library("mlr") #impute
library("caret") #createDataPartition
library("gbm")
library("ggplot2")
library("gmodels") #CrossTable
library(RPushbullet)


#data profiling
str(train)
summary(train)
hist(train$ApplicantIncome, breaks=30)
hist(train$CoapplicantIncome, breaks=30)
hist(train$Loan_Amount_Term, breaks=20)
table(train$Credit_History)
table(train$Education,train$Loan_Status) #for cat-cat relationships
CrossTable(train$Education, train$Loan_Status, prop.chisq = F,prop.r = F,prop.t = F ,dnn=c('Education','Loan Status'))
CrossTable(train$Property_Area, train$Loan_Status, prop.chisq = F,prop.r = F,prop.t = F ,dnn=c('Property area','Loan Status'))
#CrossTable(train$Gender, train$Loan_Status, prop.chisq = F,prop.r = F,prop.t = F ,dnn=c('Gender','Loan Status')) #no role
CrossTable(train$Married, train$Loan_Status, prop.chisq = F,prop.r = F,prop.t = F ,dnn=c('Married','Loan Status'))
CrossTable(train$Dependents, train$Loan_Status, prop.chisq = F,prop.r = F,prop.t = F ,dnn=c('Dependents','Loan Status'))
CrossTable(train$Self_Employed, train$Loan_Status, prop.chisq = F,prop.r = F,prop.t = F ,dnn=c('Self Employed','Loan Status'))


#treating missing values
colSums(is.na(train))
imputed_data_train <- impute(train, cols = list(LoanAmount = imputeMedian(), Loan_Amount_Term = imputeMedian(), Credit_History = imputeMedian()))
train <- imputed_data_train$data
imputed_data_test <- impute(test, cols = list(LoanAmount = imputeMedian(), Loan_Amount_Term = imputeMedian(), Credit_History = imputeMedian()))
test <- imputed_data_test$data


#splitting training and testing sets
index <- createDataPartition(train$Loan_Status, p=0.75, list=FALSE)
train_set <- train[index,]
test_set <- train[-index,]


#defining predictors and outcomes
predictors <- c('ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History')
outcome <- 'Loan_Status'


#XGBoost
xgb_grid <- expand.grid(
  nrounds = c(250,500,1000),
  max_depth = c(5,8,12),
  eta = c(0.025,0.05,0.1),
  gamma = c(0.5,0.7,1),
  min_child_weight = c(3,5,7),
  colsample_bytree = c(0.6,0.8,1),
  subsample = c(0.6,0.8,1)
)

grid_ix <- sample(1:nrow(xgb_grid), 1000)
xgb_grid1 <- xgb_grid[grid_ix,]

xgb_train_control <- trainControl(
  method = "repeatedcv",
  number  = 4,
  repeats = 2,
  verboseIter = FALSE,
  returnData = FALSE,
  allowParallel = TRUE
)

train$Loan_Status = ifelse(train$Loan_Status == "Y",1,0)



xgb_train <- caret::train(
  x = as.matrix(train[,predictors]),
  y = train$Loan_Status,
  objective = "binary:logistic",
  trControl = xgb_train_control,
  tuneGrid = xgb_grid1,
  method = "xgbTree"
)
pbPost("note","load pred xgb done","xgb done")

#get best parameters
head(xgb_train$results[with(xgb_train$results, order(RMSE)),],1)

pred <- predict(xgb_train, newdata = test[,predictors])
prediction <- as.numeric(pred > 0.5)

pred_xgb_DF <- data.frame(test$Loan_ID, prediction)
colnames(pred_xgb_DF) <- c('Loan_ID','Loan_Status')
pred_xgb_DF$Loan_Status = ifelse(pred_xgb_DF$Loan_Status == 1,"Y","N")
write.csv(pred_xgb_DF, file = 'LP3_xgb_solution.csv', row.names = FALSE)




#defining train control parameters
fitControl <- trainControl(
  method = 'cv',
  number = 5,
  savePredictions = 'final',
  classProbs = T
)


#baseline prediction
sol_DF <- data.frame(test$Loan_ID, "Y")
colnames(sol_DF) <- c('Loan_ID','Loan_Status')
write.csv(sol_DF, file = "LP3_solution.csv", row.names = FALSE)


#training and testing random forest model
model_RF <- train(train_set[,predictors], train_set[,outcome], method = 'rf', trControl = fitControl, tuneLength = 3)
pred_RF <- predict(model_RF, test_set[,predictors])
confusionMatrix(test_set$Loan_Status, pred_RF)

model_RF1 <- train(train[,predictors], train[,outcome], method = 'rf', trControl = fitControl, tuneLength = 3)
pred_RF1 <- predict(model_RF1, test[,predictors], type = 'prob')


#training and testing knn
model_knn <- train(train_set[,predictors], train_set[,outcome], method = 'knn', trControl = fitControl, tuneLength = 3)
pred_knn <- predict(model_knn, test_set[,predictors])
confusionMatrix(test_set$Loan_Status, pred_knn)

model_knn1 <- train(train[,predictors], train[,outcome], method = 'knn', trControl = fitControl, tuneLength = 3)
pred_knn1 <- predict(model_knn1, test[,predictors], type = 'prob')


#training and testing logistic regression
model_glm <- train(train_set[,predictors], train_set[,outcome], method = 'glm', trControl = fitControl, tuneLength = 3)
pred_glm <- predict(model_glm, test_set[,predictors])
confusionMatrix(test_set$Loan_Status, pred_glm)

model_glm1 <- train(train[,predictors], train[,outcome], method = 'glm', trControl = fitControl, tuneLength = 3)
pred_glm1 <- predict(model_glm1, test[,predictors], type = 'prob')


#Taking weighted average of predictions
pred<-(pred_RF1$Y*0.35)+(pred_knn1$Y*0.29)+(pred_glm1$Y*0.36)

#Splitting into binary classes at 0.5
pred<-as.factor(ifelse(pred>0.5,'Y','N'))

pred_DF <- data.frame(test$Loan_ID, pred)
colnames(pred_DF) <- c('Loan_ID','Loan_Status')
write.csv(pred_DF, file = 'LP3_avg_solution.csv', row.names = FALSE)


#training and testing gbm
model_gbm <- train(train_set[,predictors], train_set[,outcome], method = 'gbm', trControl = fitControl, tuneLength = 3)
pred_gbm <- predict(model_gbm, test_set[,predictors])
confusionMatrix(test_set$Loan_Status, pred_gbm)

#model_knn <- train(train[,predictors], train[,outcome], method = 'knn', trControl = fitControl, tuneLength = 3)
#pred_knn <- predict(model_knn, test[,predictors])
#pred_knn_DF <- data.frame(test$Loan_ID, pred_knn)
#colnames(pred_knn_DF) <- c('Loan_ID','Loan_Status')
#write.csv(pred_knn_DF, file = 'LP3_knn_solution.csv', row.names = FALSE)

train_set$Loan_Status <- ifelse(train_set$Loan_Status == "Y",1,0)
test_set$Loan_Status <- ifelse(test_set$Loan_Status == "Y",1,0)
modeGBM <- gbm(Loan_Status ~  ApplicantIncome+CoapplicantIncome+LoanAmount+Loan_Amount_Term+Credit_History , data = train_set, n.trees = 100)
predGBM <- predict(modeGBM, test_set, n.trees = 100)
test_set$Loan_Status <- ifelse(test_set$Loan_Status == 1,"Y","N")
test_set$Loan_Status <- as.factor(test_set$Loan_Status)
confusionMatrix(test_set$Loan_Status, predGBM)
