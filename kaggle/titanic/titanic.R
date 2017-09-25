rm(list = ls())


#loading datasets
train1 <- read.csv("C:/Users/manish.sharma/Downloads/kaggle/titanic/train.csv")
test1 <- read.csv("C:/Users/manish.sharma/Downloads/kaggle/titanic/test.csv")


#data profiling
str(train1)
summary(train1)
hist(train1$Fare, breaks = 50)
colSums(is.na(train1))


#remove unnecessary columns
train <- subset(train1, select = -c(Name,Ticket,Cabin))
test <- subset(test1, select = -c(Name,Ticket,Cabin))


#converting factor columns to numeric
for(i in 1:9)
{
  if(is.factor(train[,i]))
  {
    train[,i] = as.numeric(train[,i])
  }
}
for(i in 1:8)
{
  if(is.factor(test[,i]))
  {
    test[,i] = as.numeric(test[,i])
  }
}
cor(train)*100
