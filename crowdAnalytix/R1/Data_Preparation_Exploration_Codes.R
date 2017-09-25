rm(list = ls())

startup<- read.csv(file="E:/crowdAnalytix/CAX_Startup_Data.csv", header=TRUE,as.is=T)
Dict_startup<- read.csv(file="E:/crowdAnalytix/CAX_Startup_Data_Dictionary.csv", header=TRUE,as.is=T)

summary(startup)

# replacing 'No Info' and 'blanks' with NA
startup[startup=="No Info"]<- NA
startup[startup==""]<- NA

# converting column as date
startup$Est..Founding.Date <- as.Date(startup$Est..Founding.Date, "%m/%d/%Y")
startup$Last.Funding.Date <- as.Date(startup$Last.Funding.Date, "%m/%d/%Y")

# R code for converting character vector to numeric
# display column header of data
colnames(startup) 
str(startup)

# noting columns that needs to be converted to numeric
col<- c(3:5,10,11,18:23,25,61,66,68:70,72,74,88,92,94:96,98,99,102:116)

# using for loop to convert column as numeric
for(i in col)
  {
   startup[,i]<-as.numeric(startup[,i])
  }

str(startup)

# Percent missing value for each variable
mis_val<-sapply(startup, function(x) sum(is.na(x)))
percent_mis<-as.data.frame(round((mis_val/nrow(startup))*100,1))

name<-row.names(percent_mis)
pcnt_mis_var<-cbind(name,percent_mis)
row.names(pcnt_mis_var)<-NULL
colnames(pcnt_mis_var)<-c("variable","Percent.Missing")

# keeping only variables with less than 40% missing
new_var<-as.character(pcnt_mis_var$variable[which(pcnt_mis_var$Percent.Missing<=40)])
new_startup<-startup[new_var]

# separate data frame for more than 40% missing
other_var<-as.character(pcnt_mis_var$variable[which(pcnt_mis_var$Percent.Missing>40)])
other_data<-startup[other_var]


# writing new data as csv file
write.csv(new_startup,"filtered_data.csv",row.names=F)

# Separate data frame for numeric variables
cnt_df<-new_startup[,c(3:5,10,12:14,17:22,24,60,65,67:69,71,73,85,89,91:93,
                       95,96,99:113)]

# separate data frame for character variables
cnt_var<-colnames(cnt_df)
var <- colnames(new_startup) %in% cnt_var 
char_df <- new_startup[!var]

# checking distribution of continuous variable for outlier detection and missing values
summary(cnt_df$Team.size.all.employees)
quantile(cnt_df$Team.size.all.employees, probs = seq(0, 1, by= 0.05),na.rm=T)

# further exploration to determine cutoff for capping
quantile(cnt_df$Team.size.all.employees, probs = seq(.9, 1, by= 0.01),na.rm=T)

# capping values
cnt_df$Team.size.all.employees[cnt_df$Team.size.all.employees>103.8]<-103.8
                                 
# checking distribution of categorical variable for missing values
table(char_df$Local.or.global.player,useNA="always")

# convert a variable to uppercase
char_df$Local.or.global.player<-toupper(char_df$Local.or.global.player)

# trimming whitespaces
char_df$Local.or.global.player<-trimws(char_df$Local.or.global.player)

# Recoding variable levels
#char_df$Local.or.global.player[char_df$Local.or.global.player=='LOCAL']<-0
#char_df$Local.or.global.player[char_df$Local.or.global.player=='GLOBAL']<-1

char_df$Local.or.global.player<- as.factor(char_df$Local.or.global.player)

# Create additional features like counting number of investors for company
char_df$Investor.count<-length(strsplit(char_df$Investors, "|",fixed=T))
for (i in (1:length(char_df$Investors)))
{
  if(is.na(char_df$Investors[i])==T){
    char_df$Investor.count[i]<- NA}
  else{
    lst<-strsplit(char_df$Investors[i], "|", fixed=T)
    char_df$Investor.count[i]<-length(lst[[1]])
  } }

# adding dependent variable to numeric data frame
cnt_df$Dependent.Company.Status<-char_df$Dependent.Company.Status

# boxplot of employee count
boxplot(cnt_df$Employee.Count, main="box plot of employee count", 
        ylab="Employee count")

# histogram with black outline, white fill and median line
library(ggplot2)
ggplot(cnt_df, aes(x=Employee.Count))+
  geom_histogram(binwidth=5, colour="black", fill="white")+
  geom_vline(aes(xintercept=median(Employee.Count, na.rm=T)),   
             color="red", linetype="dashed", size=1)+
  ggtitle("Histogram of Employee count")+
  xlab("Employee Count") + 
  ylab("Frequency")+
  theme_light()

# box plot to see differnce in mean of team size w.r.t two categories of dependent
ggplot(cnt_df, aes(x=Dependent.Company.Status,y=Team.size.all.employees, 
                   fill=Dependent.Company.Status)) + 
  geom_boxplot()

# data preparation for bar chart
avg_emp<-aggregate(as.numeric(cnt_df$Team.size.all.employees), 
                   by=list(as.factor(cnt_df$Dependent.Company.Status)), 
                           FUN=mean, na.rm=TRUE)
colnames(avg_emp)<-c("company.status","Avg.Employee.size")

# bar chart to check for difference in mean
ggplot(avg_emp, aes(x = company.status, y = Avg.Employee.size)) +
  geom_bar(stat = "identity")

# Missing value treatemnt using median
cnt_df$Team.size.all.employees[is.na(cnt_df$Team.size.all.employees)]<-median(cnt_df$Team.size.all.employees,na.rm=T)

# function to calculate mode
Mode <- function(x) {
  u <- unique(x)
  u[which.max(tabulate(match(x, u)))]
  }
# Filling missing values with mode
char_df$Local.or.global.player[is.na(char_df$Local.or.global.player)]<-Mode(char_df$Local.or.global.player)

# t-test for checking difference in mean
t.test(Team.size.all.employees~Dependent.Company.Status, data=cnt_df)

# tabulating data for chi-sq test
tab<- table(char_df$Dependent.Company.Status,char_df$Local.or.global.player)

# chi-sq test
chisq.test(tab)

