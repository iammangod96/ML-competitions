rm(list=ls())



train <- as.tibble(read.csv("E:/kaggle/animalshelter/train.csv.gz"))
test <- read.csv("E:/kaggle/animalshelter/test.csv.gz")
sam_sub <- read.csv("E:/kaggle/animalshelter/sample_submission.csv.gz")


write.csv(sam_sub, file="animalshelter_baseline.csv",row.names = FALSE)                                                                                                                                                                                                                                                                                                                                                                                                                                                  