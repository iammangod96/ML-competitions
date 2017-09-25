rm(list = ls())


#loading libraries
library(forecast)
library(zoo) #rollmean, rollapply
library(ggplot2) #ggplot
library(TTR) #EMA
library(xts)
library(tseries) #adf.test


#loading datasets
train <- read.csv("E:/competitions/AV/time_series_2/train.csv")
test <- read.csv("E:/competitions/AV/time_series_2/test.csv")
samp_sub <- read.csv("E:/competitions/AV/time_series_2/samp_sub.csv")

train$Datetime = dmy_hm(train$Datetime)
test$Datetime = dmy_hm(test$Datetime)


#creating time series object
train_ts <- xts(train[,-1]$Count, order.by = train$Datetime)
plot.ts(train_ts)


#moving average
rolmean <- rollmean(x = train_ts, k = 24)
ggplot()+theme(panel.background = element_rect(fill = "grey"))+
  geom_line(data = train_ts, aes(time(train_ts), train_ts, color = "train_ts"), color = "white", lwd = 1.1)+
  geom_line(data = rolmean, aes(time(rolmean), rolmean, color = "roll mean"),color = "red", lwd = 1.2)


#dickey-fuller test
adf.test(train_ts , alternative  = "stationary")


#decomposing
exp_wmean <- EMA(x = train_ts, n = 24)
ggplot()+theme(panel.background = element_rect(fill = "grey"))+
  geom_line(data = train_ts, aes(time(train_ts), train_ts, color = "train_ts"), color = "white", lwd = 1.1)+
  geom_line(data = exp_wmean, aes(time(exp_wmean), exp_wmean, color = "roll mean"),color = "red", lwd = 1.2)
train_e_diff <- train_ts - exp_wmean
ggplot()+theme(panel.background = element_rect(fill = "grey"))+
  geom_line(data = train_e_diff, aes(time(train_e_diff), train_e_diff, color = "train_e_diff"), color = "white", lwd = 1.1)
  

#train_sqrt
train_sqrt <- (train_e_diff)**(1/2)
ggplot()+theme(panel.background = element_rect(fill = "grey"))+
  geom_line(data = train_sqrt, aes(time(train_sqrt), train_sqrt, color = "train_sqrt"), color = "red")


#ACF AND PACF plots
par(mfrow=c(2,1))
Acf(train_sqrt)
Pacf(train_sqrt)


#arima model
model <- arima(train_sqrt, order=c(1,1,1))
fcast <- forecast(model, h = 3000)
plot(fcast)
