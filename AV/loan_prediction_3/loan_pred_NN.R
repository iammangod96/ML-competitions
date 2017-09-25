rm(list = ls())


#loading libraries
library("mlr") #impute
library("caret") #createDataPartition
library(randomForest)
library(neuralnet)
library(glmnet)
library(RPushbullet)


#LOADING DATASETS
train <- read.csv("E:/competitions/AV/loan_prediction_3/train.csv")
test <- read.csv("E:/competitions/AV/loan_prediction_3/test.csv")


#data profiling
str(train)
summary(train)
str(test)
summary(test)


#treating missing values
colSums(is.na(train))
imputed_data_train <- impute(train, cols = list(LoanAmount = imputeMedian(), Loan_Amount_Term = imputeMedian(), Credit_History = imputeMedian()))
train <- imputed_data_train$data
imputed_data_test <- impute(test, cols = list(LoanAmount = imputeMedian(), Loan_Amount_Term = imputeMedian(), Credit_History = imputeMedian()))
test <- imputed_data_test$data


#treating factor missing values
train$Married[train$Married == ""] <- as.factor("Yes")
train$Married <- droplevels(train$Married)

train$Gender[train$Gender == ""] <- as.factor("Male")
train$Gender <- droplevels(train$Gender)
test$Gender[test$Gender == ""] <- as.factor("Male")
test$Gender <- droplevels(test$Gender)

train$Dependents[train$Dependents == ""] <- as.factor("0")
train$Dependents <- droplevels(train$Dependents)
test$Dependents[test$Dependents == ""] <- as.factor("0")
test$Dependents <- droplevels(test$Dependents)

train$Self_Employed[train$Self_Employed == ""] <- as.factor("No")
train$Self_Employed <- droplevels(train$Self_Employed)
test$Self_Employed[test$Self_Employed == ""] <- as.factor("No")
test$Self_Employed <- droplevels(test$Self_Employed)

train[,c(2,3,4,5,6)] <- as.numeric(unlist(train[,c(2:6)]))
train$Property_Area <- as.numeric(train$Property_Area)
test[,c(2,3,4,5,6)] <- as.numeric(unlist(test[,c(2:6)]))
test$Property_Area <- as.numeric(test$Property_Area)





#random forest model
rf_model <- randomForest(data = train,Loan_Status ~ . -Loan_ID)
varImpPlot(rf_model)
rf_model1 <- train(Loan_Status ~ . , data = train[! names(train) %in% c("Loan_ID")], method = "rf")
plot(varImp(rf_model1))
pred <- predict(rf_model, newdata = test)
pred_df <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = pred)
write.csv(pred_df, file = "loan_pred3_rf.csv", row.names = FALSE)
pred1 <- predict(rf_model1, test)
pred1_df <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = pred1)
write.csv(pred1_df, file = "loan_pred3_rf1.csv", row.names = FALSE)


#xgboost
xgb_grid <- expand.grid(
  nrounds = c(250,500,1000),
  max_depth = c(5,8,12),
  eta = c(0.025,0.05,0.1),
  gamma = c(0.5,0.7,1),
  min_child_weight = c(3,5,7),
  colsample_bytree = c(0.6,0.8,1),
  subsample = c(0.6,0.8,1)
)

grid_ix <- sample(1:nrow(xgb_grid), 100)
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
  x = as.matrix(train[! names(train) %in% c("Loan_ID","Loan_Status")]),
  y = train$Loan_Status,
  objective = "binary:logistic",
  trControl = xgb_train_control,
  tuneGrid = xgb_grid1,
  method = "xgbTree"
)
xgb.save(xgb_train, "loan_pred3_grid1_100")
pbPost("note","load pred xgb 100 done","xgb done")

pred_xgb <- predict(xgb_train, test)
pred_xgb <- as.numeric(pred_xgb > 0.5)
pred_xgb_df <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = pred_xgb)
pred_xgb_df$Loan_Status <- ifelse(pred_xgb_df$Loan_Status == 1,"Y","N")
write.csv(pred_xgb_df, file = "loan_pred3_xgb100.csv", row.names = FALSE)



#logistic regression
lr_model <- train( Loan_Status ~ . , data = train[! names(train) %in% c("Loan_ID")], method = "glm")
pred_lr <- predict(lr_model, newdata = test)
pred_lr_df <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = pred_lr)
write.csv(pred_lr_df, file = "loan_pred3_lr.csv", row.names = FALSE)


#avg of logistic regression and random forest

pred_avg <- (as.numeric(pred_lr) + as.numeric(pred))/2
pred_avg <- pred_avg - 1
pred_avg <- round(pred_avg)
table(pred_avg)
pred_avg_df <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = pred_avg)
pred_avg_df$Loan_Status <- ifelse(pred_avg_df$Loan_Status == 1,"Y","N")
write.csv(pred_avg_df, file = "loan_pred3_avg.csv", row.names = FALSE)



###########################################################################################

#Neural network from scratch

sigmoid <- function(z)
{
  return(1/(1+exp(-1*z)))
}

init_params <- function(dim)
{
  w <- matrix(0,dim,1)
  b <- 0
  list("w" = w,"b" = b)
}

propagate <- function(w, b, X, Y)
{
  m <- dim(X)[1]
  
  #forward propagation
  A <- sigmoid(t( w )%*%X + b)
  cost <- (-1/m)*sum(Y*log(A) + (1-Y)*log(1-A))
  s
  #backward propagation
  dw <- (X%*%(t(A - Y)))/m
  db <- sum(A-Y)/m
  list("A" = A,"dw" = dw, "db" = db, "cost" = cost)
}


# w <- matrix(c(1,2),nrow = 2, ncol = 1)
# b <- 0
# X <- matrix(c(1,2,3,4), nrow = 2, ncol = 2)
# m <- dim(X)[1]
# Y <- matrix(c(1,0), nrow = 1, ncol = 2)
# fb <- propagate(w,b,X,Y)
# fb$dw
# fb$db
# fb$cost
# fb$A


optimize <- function(w,b,X,Y,num_iters,learning_rate)
{
  costs = c()
  for(i in 1:num_iters)
  {
    props <- propagate(w,b,X,Y)
    dw <- props$dw
    db <- props$db
    w <- w - learning_rate*dw
    b <- b - learning_rate*db
    cost <- props$cost
    
    if(i %% 100 == 0)
    {
      paste("Cost after iteration",i,":",cost)      
    }
  }
  list("w" = w, "b" = b, "dw" = dw, "db" = db, "costs" = costs)
}

# fb_opt <- optimize(w,b,X,Y,100,0.09)
# fb_opt$w
# fb_opt$b
# fb_opt$dw
# fb_opt$db


predict1 <- function(w,b,X)
{
  m <- dim(X)[1]
  Y_prediction <- matrix(0,1,m)
  A <- sigmoid(t( w )%*%X + b)
  for(i in 1:dim(A)[2])
  {
    if(A[1,i] <= 0.5)
    {
      Y_prediction[1,i] = 0
    }
    else
    {
      Y_prediction[1,i] = 1
    }
  }
  Y_prediction
}

# predict1(w,b,X)

#function,        params,                           returns
#sigmoid      <-  z                                 (1/(1+exp(-1*z)))
#init_params  <-  dim                               w,b
#propagate    <-  w, b, X, Y                        A,dw,db,cost
#optimize     <-  w,b,X,Y,num_iters,learning_rate   w,b,dw,db,cost
#predict1     <-  w,b,X                             Y_prediction

model1 <- function(X_train, Y_train, X_test, num_iters = 2000, learning_rate  = 0.5)
{
  init <- init_params(dim(X_train)[1])
  w <- init$w
  b <- init$b
  opt <- optimize(w,b,X_train, Y_train, num_iters, learning_rate)
  w <- opt$w
  b <- opt$b
  dw <- opt$dw
  db <- opt$db
  Y_prediction_test <- predict1(w,b,X_test)
  Y_prediction_test
}

NN_model <- model1(as.matrix(train[! names(train) %in% c("Loan_ID","Loan_Status")]), train$Loan_Status, as.matrix(test[! names(test) %in% c("Loan_ID")])) 
