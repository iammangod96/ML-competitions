rm(list = ls())


#loading libraries
library(readr)
library(caret)
library(data.table)


#loading datasets
train <- fread("E:/competitions/kaggle/digitrecognizer/train.csv")
test <- fread("E:/competitions/kaggle/digitrecognizer/test.csv")
samp_sub <- fread("E:/competitions/kaggle/digitrecognizer/sample_submission.csv")


#data profiling
table(as.factor(train$label))


#image buildup from train row
img <- matrix(0,28,28)
tot <- 0
for(i in 1:28){
  for(j in 1:28){
    tot <- tot + 1
    img[i,j] <- train[1,tot+1]
  }
}
img
image(img)

