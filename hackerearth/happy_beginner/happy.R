rm(list = ls())

#loading libraries
library(caret)
library(data.table)
library(tm)
library(SnowballC)
library(wordcloud)
library(e1071) #naivebayes
library(gmodels) #crosstable


#loading datasets
train <- fread("E:/competitions/hackerearth/happy_beginner/train.csv") 
test <- fread("E:/competitions/hackerearth/happy_beginner/test.csv") 
samp_sub <- fread("E:/competitions/hackerearth/happy_beginner/sample_submission.csv") 


#data profiling
table(train$Browser_Used, train$Is_Response)
str(train)
train$Is_Response <- factor(train$Is_Response)


#corpus building
desc_corpus <- VCorpus(VectorSource(train$Description))
inspect(desc_corpus[[1]])
inspect(desc_corpus[[2]])


#cleaning corpus
desc_corpus_clean <- tm_map(desc_corpus, content_transformer(tolower))
desc_corpus_clean <- tm_map(desc_corpus_clean, removeNumbers)
desc_corpus_clean <- tm_map(desc_corpus_clean, removeWords, stopwords())
desc_corpus_clean <- tm_map(desc_corpus_clean, removePunctuation)
desc_corpus_clean <- tm_map(desc_corpus_clean, stemDocument)
desc_corpus_clean <- tm_map(desc_corpus_clean, stripWhitespace)


#document term matrix
desc_dtm <- DocumentTermMatrix(desc_corpus_clean)


#train dtm
train_dtm <- desc_dtm
train_labels <- train$Is_Response


#word cloud
wordcloud(desc_corpus_clean, min.freq = 2000, random.order = FALSE)


#description frequent words
desc_freq_words <- findFreqTerms(train_dtm, 2000)
train_dtm_freq <- train_dtm[, desc_freq_words]


#function to convert numeric data to categorical
convert_counts <- function(x)
{
  x <- ifelse(x > 0, "Yes", "No")
}

train_set <- apply(train_dtm_freq, MARGIN = 2, convert_counts)



#training
train_classifier <- naiveBayes(train_set, train_labels)

#predict
#pred <- predict(train_classifier, test_set)

#evaluation
#CrossTable(pred, test_labels, prop.chisq = F, prop.t = F, dnn = c('predicted','actual'))
#76%




##########################################################################################################
#test data preparation like train data
##########################################################################################################


#corpus building
desc_corpus_test <- VCorpus(VectorSource(test$Description))
inspect(desc_corpus_test[[1]])
inspect(desc_corpus_test[[2]])


#cleaning corpus
desc_corpus_clean_test <- tm_map(desc_corpus_test, content_transformer(tolower))
desc_corpus_clean_test <- tm_map(desc_corpus_clean_test, removeNumbers)
desc_corpus_clean_test <- tm_map(desc_corpus_clean_test, removeWords, stopwords())
desc_corpus_clean_test <- tm_map(desc_corpus_clean_test, removePunctuation)
desc_corpus_clean_test <- tm_map(desc_corpus_clean_test, stemDocument)
desc_corpus_clean_test <- tm_map(desc_corpus_clean_test, stripWhitespace)


#document term matrix
desc_dtm_test <- DocumentTermMatrix(desc_corpus_clean_test)


#test dtm
test_dtm <- desc_dtm_test


#description frequent words
desc_freq_words_test <- findFreqTerms(test_dtm, 2000)
test_dtm_freq <- test_dtm[, desc_freq_words_test]


#function to convert numeric data to categorical
test_set <- apply(test_dtm_freq, MARGIN = 2, convert_counts)


#predict
pred <- predict(train_classifier, test_set)
pred_df <- data.frame(User_ID = test$User_ID, Is_Response = pred)
write.csv(pred_df, file = "happy_naiveBayes.csv", row.names = FALSE)




###############################################################################################
#Second naive bayes classifier for browser and device used featres
###############################################################################################


second_train_set <- train[,c(3,4)]
second_test_set <- test[,c(3,4)]

second_train_set$Browser_Used <- factor(second_train_set$Browser_Used)
second_train_set$Device_Used <- factor(second_train_set$Device_Used)

second_test_set$Browser_Used <- factor(second_test_set$Browser_Used)
second_test_set$Device_Used <- factor(second_test_set$Device_Used)

#model
second_train_classifier <- naiveBayes(second_train_set, train_labels)

#predict
second_pred <- predict(second_train_classifier, second_test_set)
second_pred_df <- data.frame(User_ID = test$User_ID, Is_Response = second_pred)
write.csv(second_pred_df, file = "happy_naiveBayes_second.csv", row.names = FALSE)


###############################################################################################
#third naive bayes classifier average
###############################################################################################

pred1 <- predict(train_classifier, test_set, "raw")
pred2 <- predict(second_train_classifier, second_test_set, "raw")

pred3 <- (pred1 + pred2)/2
pred3_df <- data.frame(User_ID = test$User_ID, Is_Response = pred3)
pred3_df$Is_Response <- ifelse(pred3_df$Is_Response.happy >= pred3_df$Is_Response.not.happy, "happy", "not happy")
pred3_df$Is_Response.happy <- NULL
pred3_df$Is_Response.not.happy <- NULL

write.csv(pred3_df, file = "happy_naiveBayes_avg.csv", row.names = FALSE)
