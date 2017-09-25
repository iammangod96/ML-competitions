rm(list = ls())


#loading libraries
library("data.table") #fread
library("tibble") #as.tibble
library("ggplot2") #ggplot
library("dplyr") #%>%,bind_rows,summarise
library("lubridate") #ymd_hms
library("Rmisc")
library("rpart")
library("randomForest")
library("jsonlite")
library("RPushbullet")
library("geosphere")
library("xgboost")
library("caret")


#useful functions

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


#loading datasets
train <- as.tibble(fread("E:/competitions/kaggle/newyorktaxi/train.csv"))
test <- as.tibble(fread("E:/competitions/kaggle/newyorktaxi/test.csv"))
sam_sub <- as.tibble(fread("E:/competitions/kaggle/newyorktaxi/sample_submission.csv"))


#data profiling
summary(train)
hist(train$trip_duration[train$trip_duration < 3000], breaks  = 100)
glimpse(train)
colSums(is.na(train))
summary(test)
glimpse(test)
colSums(is.na(test))


#combining train and test data
comb <- bind_rows(
  train %>% mutate(dset="train"),
  test  %>% mutate(dset="test", dropoff_datetime=NA, trip_duration = NA)
)
str(comb)
comb <- comb %>% mutate(dset = factor(dset))


#datetime columns and factorize some other columns
comb <- comb %>% mutate(
  pickup_datetime = ymd_hms(pickup_datetime),
  dropoff_datetime = ymd_hms(dropoff_datetime),
  vendor_id = factor(vendor_id),
  passenger_count = factor(passenger_count)
)
str(comb)


#univariate analysis visualizations
train %>% ggplot(aes(trip_duration)) + geom_histogram(fill = 'blue',bins=200) + scale_x_log10() + scale_y_sqrt()
train %>% arrange(desc(trip_duration)) %>% select(trip_duration,pickup_datetime,dropoff_datetime,everything()) %>%head(15)

p1 <- comb %>% ggplot(aes(pickup_datetime)) + geom_histogram(fill="red",bins=200) + xlab("Pickup date")
p2 <- comb %>% ggplot(aes(dropoff_datetime)) + geom_histogram(fill="blue",bins=200) + xlab("Dropoff date")
layout <- matrix(c(1,2),2,1,byrow = FALSE)
multiplot(p1,p2, layout = layout)

p1 <- comb %>% ggplot(aes(vendor_id,fill = vendor_id))+geom_bar()
p2 <- comb %>% ggplot(aes(store_and_fwd_flag, fill = store_and_fwd_flag)) + geom_bar()+scale_y_log10()
p3 <- comb %>%
  mutate(wday = wday(pickup_datetime, label = TRUE)) %>%
  group_by(wday, vendor_id) %>%
  count() %>%
  ggplot(aes(wday, n, colour = vendor_id)) +
  geom_point(size = 4) +
  labs(x = "Day of the week", y = "Total number of pickups") 
p4 <- comb %>%
  mutate(hour = hour(pickup_datetime)) %>%
  group_by(hour, vendor_id) %>%
  count() %>%
  ggplot(aes(hour,n,colour = vendor_id)) + 
  geom_point(size=4)
layout <- matrix(c(1,2,3,4),2,2,byrow = FALSE)
multiplot(p1,p2,p3,p4,layout = layout)

p1 <- comb %>%
  group_by(passenger_count) %>%
  count() %>%
  ggplot(aes(passenger_count,n,fill = passenger_count)) +
  geom_col() +
  scale_y_log10()

p2 <- comb %>%
  mutate(hour = hour(pickup_datetime), month = factor(month(pickup_datetime, label = TRUE))) %>%
  group_by(hour,month) %>%
  count() %>%
  ggplot(aes(hour, n,color = month)) +
  geom_line(size = 2)

layout = matrix(c(1,2),2,1,byrow = FALSE)
multiplot(p1,p2,layout = layout)

#bivariate analysis using visualization
p1 <- comb %>%
  mutate(wday = factor(wday(pickup_datetime, label = TRUE))) %>%
  group_by(wday, vendor_id) %>%
  summarise(med = median(trip_duration)) %>%
  ggplot(aes(wday,med,color = vendor_id)) +
  geom_point(size=4)


p2 <- train %>%
  mutate(hour = hour(pickup_datetime)) %>%
  group_by(hour,vendor_id) %>%
  summarise(med = median(trip_duration)) %>%
  ggplot(aes(hour,med,color = vendor_id)) +
  geom_point(size = 4)

layout = matrix(c(1,2),2,1,byrow = FALSE)
multiplot(p1,p2,layout = layout)

#new baseline model with simple feature engineering
new_train <- train %>%
  mutate(wday = wday(pickup_datetime), month = month(pickup_datetime), hour = hour(pickup_datetime)) %>%
  select(-c(id,pickup_datetime,dropoff_datetime))


pick_coord <- new_train %>%
  select(pickup_longitude, pickup_latitude)
drop_coord <- new_train %>%
  select(dropoff_longitude, dropoff_latitude)
new_train$dist <- distCosine(pick_coord, drop_coord)
new_train$bearing = bearing(pick_coord, drop_coord)


new_train <- new_train %>%
  select(-c(pickup_longitude,dropoff_longitude,pickup_latitude,dropoff_latitude))

new_train <- new_train[new_train$trip_duration < 4000,]
new_train$store_and_fwd_flag <- ifelse(new_train$store_and_fwd_flag == "N",0,1)

#linear model
lr_model <- lm(data = new_train, trip_duration ~ .)
summary(lr_model)
try_train <- subset(new_train, select = -c(vendor_id,store_and_fwd_flag))
try_lr_model <- lm(data = try_train, trip_duration ~ .)
summary(try_lr_model)




#decision tree model
dt_model <- rpart(data = new_train, trip_duration ~ .)
summary(dt_model)
try_dt_model <- rpart(data = try_train, trip_duration ~ .)
summary(try_dt_model)



#converting test data like as new_train
new_test <- test %>%
  mutate(wday = wday(pickup_datetime), month = month(pickup_datetime), hour = hour(pickup_datetime)) %>%
  select(-c(id,pickup_datetime))


pick_coord_test <- new_test %>%
  select(pickup_longitude, pickup_latitude)
drop_coord_test <- new_test %>%
  select(dropoff_longitude, dropoff_latitude)
new_test$dist <- distCosine(pick_coord_test, drop_coord_test)
new_test$bearing = bearing(pick_coord_test, drop_coord_test)



new_test <- new_test %>%
  select(-c(pickup_longitude,dropoff_longitude,pickup_latitude,dropoff_latitude))
new_test$store_and_fwd_flag <- ifelse(new_test$store_and_fwd_flag == "N",0,1)







#prediction using lr_model
pred_lr_baseline <- predict(lr_model, newdata = new_test)
pred_lr_baseline_df <- data.frame(test$id, pred_lr_baseline)
colnames(pred_lr_baseline_df) <- c('id','trip_duration')
write.csv(pred_lr_baseline_df, file='nytaxi_lr.csv',row.names = FALSE)

#prediction using try_lr_model
try_test <- subset(new_test, select = -c(vendor_id,store_and_fwd_flag))
pred_trylr_baseline <- predict(try_lr_model, newdata = new_test)
pred_trylr_baseline_df <- data.frame(test$id, pred_trylr_baseline)
colnames(pred_trylr_baseline_df) <- c('id','trip_duration')
write.csv(pred_trylr_baseline_df, file='nytaxi_trylr_baseline.csv',row.names = FALSE)


#prediction using dt_model
pred_dt_baseline <- predict(dt_model, newdata = new_test)
pred_dt_baseline_df <- data.frame(test$id, pred_dt_baseline)
colnames(pred_dt_baseline_df) <- c('id','trip_duration')
write.csv(pred_dt_baseline_df, file='nytaxi_dt.csv',row.names = FALSE)

#prediction using try_dt_model
pred_trydt_baseline <- predict(try_dt_model, newdata = try_test)
pred_trydt_baseline_df <- data.frame(test$id, pred_trydt_baseline)
colnames(pred_trydt_baseline_df) <- c('id','trip_duration')
write.csv(pred_trydt_baseline_df, file='nytaxi_trydt_baseline.csv',row.names = FALSE)



##########################################
#usin caret package

#decision tree
dt_model2 <- train(data = new_train, trip_duration ~ . , method = "rpart")
#prediction
pred_dt2 <- predict(dt_model2, newdata = new_test)
pred_dt2_df <- data.frame(test$id, pred_dt2)
colnames(pred_dt2_df) <- c('id','trip_duration')
write.csv(pred_dt2_df, file='nytaxi_dt2.csv',row.names = FALSE)


#XGBOOST
#tune settings
xgb_grid <- expand.grid(
  nrounds = c(250,500,1000),
  max_depth = c(1,2,4),
  eta = c(0.001, 0.003, 0.01),
  gamma = c(0,1,2),
  min_child_weight = c(1, 2),
  colsample_bytree = c(1,0.5,0.25),
  subsample = 1
)

grid_ix <- sample(1:nrow(xgb_grid), 100)
xgb_grid1 <- xgb_grid[grid_ix,]

xgb_train_control <- trainControl(
  method = "repeatedcv",
  number  = 5,
  repeats = 2,
  verboseIter = FALSE,
  returnData = FALSE,
  allowParallel = TRUE
)

xgb_train <- train(
  x = as.matrix(new_train[,!names(new_train) %in% c("trip_duration")]),
  y = new_train$trip_duration,
  objective = "reg:linear",
  trControl = xgb_train_control,
  tuneGrid = xgb_grid1,
  method = "xgbTree"
)

xgb.save(xgb_train, "nytaxi_grid1_10")
pbPost("note","xgb_train nytaxi done","xgboost run completed")

#prediction
pred_xgb <- predict(xgb_train, newdata = new_test)
pred_xgb_df <- data.frame(test$id, pred_xgb)
colnames(pred_xgb_df) <- c('id','trip_duration')
write.csv(pred_xgb_df, file='nytaxi_xgb.csv',row.names = FALSE)
pbPost("note","xgb csv file written","xgboost prediction done")

xgb_train

summary(pred_xgb_df)
pred_xgb_df$trip_duration[pred_xgb_df$trip_duration <= 0] <- 0

names <- names(new_train)[! names(new_train) %in% c("trip_duration")]
importanceMatrix <- xgb.importance(names, model = xgb_train$finalModel)
xgb.plot.importance(importanceMatrix[1:10,])


plot(new_train$dist, new_train$trip_duration)
summary(new_train$dist)
hist(new_train$dist, breaks = 1000)
new_train <- new_train[!(new_train$dist > 10000),]
summary(new_test$dist)
summary(new_train$dist)
