rm(list = ls())


#loading libraries
library(randomForest)
library(data.table)


#loading datasets
train <- fread("E:/competitions/kaggle/digitrecognizer/train.csv")
test <- fread("E:/competitions/kaggle/digitrecognizer/test.csv")
samp_sub <- fread("E:/competitions/kaggle/digitrecognizer/sample_submission.csv")


#making RF model
train$label <- factor(train$label)
rf_model <- randomForest(data = train, label ~ .)


#prediction
pred <- predict(rf_model, newdata = test)
pred_df <- data.frame(ImageId = 1:28000, Label = pred )
write.csv(pred_df, file = "digits.csv", row.names = FALSE)
