library(gridExtra)
library(grid)
library(ggplot2)
#install.packages("lattice")
library(lattice)
#install.packages("usdm")
library(usdm)
#install.packages("pROC")
library(pROC)
#install.packages("caret")
library(caret)
#install.packages("rpart")
library(rpart)
#install.packages("DataCombine")
library(DataCombine)
#install.packages("ROSE")
library(ROSE)
#install.packages("e1071")
library(e1071)
#install.packages("xgboost")
library(xgboost)

setwd(choose.dir())
#Reading test and train data frame
train =read.csv('train.csv')
test =read.csv('test.csv')
#checking dimension of train dataset
dim(train)
#checking dimension of test dataset
dim(test)

#basic Descriptive stats 
summary(train)
summary(test)
################### obsevations ############################################
####most of the distribution mean and median are almost same


#storing ID_code  of test train data 
train_ID_code_orignal = train$ID_code
test_Id_code_orignal  = test$ID_code

#removing Idcode from orginal dataset 
train$ID_code=NULL
test$ID_code=NULL

#check dimension of dataset after removing column
print(dim(train))
print(dim(test))

#count of target variable 
table(train$target)

#Missing value analysis
# this function takes dataframe as input and calulate precentage of missing values in each 
# columns and returns that dataframe 

findMissingValue =function(df){
  missing_val =data.frame(apply(df,2,function(x){sum(is.na(x))}))
  missing_val$Columns = row.names(missing_val)
  names(missing_val)[1] =  "Missing_percentage"
  missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(train)) * 100
  missing_val = missing_val[order(-missing_val$Missing_percentage),]
  row.names(missing_val) = NULL
  missing_val = missing_val[,c(2,1)]
  return (missing_val)
}

#check missing value in train dataset
head(findMissingValue(train))
#check missing value in test dataset
head(findMissingValue(test))

############ No missing value in test and train data #########################

# creating target and independent variable from train dataset
independent_var= (colnames(train)!='target')
X=train[,independent_var]
Y=train$target


#Multicolinearity Analysis
#checking is variable are correlated
cor=vifcor(X)
print(cor)
################### No varible are correlated ############

# Distribution plot
#This function plots distribution plot from given data set
plot_distribution =function(X)
{
  variblename =colnames(X)
  temp=1
  for(i in seq(10,dim(X)[2],10))
  {
    plot_helper(temp,i ,variblename)
    temp=i+1
  }
}
# helper function takes start and stop index to print subset distribution plot
plot_helper =function(start ,stop, variblename)
{ 
  par(mar=c(2,2,2,2))
  par(mfrow=c(4,3))
  for (i in variblename[start:stop])
  {
    plot(density(X[[i]]) ,main=i )
  }
}


# plot density plot for trainig data 
plot_distribution(X)


###################Observation distribution  train dataset  #########
#  Allmost all Distributions of variables are normal

#plot density plot for testing data
plot_distribution(test)

###########Observation  distribution Test data #############
# Allmost all Distributions of variables are normal
# Test data is very similar to train data in terms of distribution

#########################################################################
##########################  Outliers ###############

#This function plots boxplot plot from given data set
#X =dataframe
plot_boxplot =function(X)
{
  variblename =colnames(X)
  temp=1
  for(i in seq(10,dim(X)[2],10))
  {
    plot_helper(temp,i ,variblename)
    temp=i+1
  }
}

# helper function takes start and stop index to print subset distribution plot
plot_helper =function(start ,stop, variblename)
{ 
  par(mar=c(2,2,2,2))
  par(mfrow=c(4,3))
  for (i in variblename[start:stop])
  {
    boxplot(X[[i]] ,main=i)
  }
}

#boxplot for training data
plot_boxplot(X)

#box plot for testing data 
plot_boxplot(test)


# This function takes dataframe as input and fill outliers with null and return modified dataframe
# df = dataframe input 
fill_outlier_with_na=function(df)
{
  cnames=colnames(df)
  for(i in cnames)
  {
    
    val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
    df[,i][df[,i] %in% val] = NA
  }
  return (df)
}

##################################### Standardisation ##########################################################

# This function takes data frame as input and standardize dataframe
# df =data frame 
# formula=(x=mean(x))/sd(x)
standardizing=function(df)
{
  cnames =colnames(df)
  for( i in   cnames ){
    df[,i]=(df[,i] -mean(df[,i] ,na.rm=T))/sd(df[,i])
  }
  return(df)
  
}

#standardize train data 
X=standardizing(X)

#standardise test data
test =standardizing(test)

# combine independent and dependent variables
std_train =cbind(X,Y)

# spilt in test train set 
# create stratified sampling
# 70% data in train in training set 
set.seed(123)
train.index =createDataPartition(std_train$Y , p=.70 ,list=FALSE)
train = std_train[train.index,]
test  = std_train[-train.index,]
#random over sampling keeping 0's and 1's (50 :50 ) sample 
over= ovun.sample(Y~. ,data =train  , method='over' )$data

# print dim of data afer partition 
print("dim train data")
dim(train)
print("dim test data ")
dim(test)

getmodel_accuracy=function(conf_matrix)
{
  model_parm =list()
  tn =conf_matrix[1,1]
  tp =conf_matrix[2,2]
  fp =conf_matrix[1,2]
  fn =conf_matrix[2,1]
  p =(tp)/(tp+fp)
  r =(fp)/(fp+tn)
  f1=2*((p*r)/(p+r))
  print(paste("accuracy",round((tp+tn)/(tp+tn+fp+fn),2)))
  print(paste("precision",round(p ,2)))
  print(paste("recall",round(r,2)))
  print(paste("fpr",round((fp)/(fp+tn),2)))
  print(paste("fnr",round((fn)/(fn+tp),2)))
  print(paste("f1",round(f1,2)))
  
}

###############################################################################
############################ Model training ###################################

# this function takes confusion matrix as input and print various classification  metrics


############################################################################
################################# LOGISTIC REGRESSION ####################### 



#fitting logistic model BASE
over_logit =glm(formula = Y~. ,data =train ,family='binomial')
# model summary  
summary(over_logit)
#get model predicted  probality 
y_prob =predict(over_logit , test[-201] ,type = 'response' )
# convert   probality to class according to thresshold
y_pred = ifelse(y_prob >0.5, 1, 0)
#create confusion matrix 
conf_matrix= table(test[,201] , y_pred)
#print model accuracy
getmodel_accuracy(conf_matrix)
# get auc 
roc=roc(test[,201], y_prob)
print(roc )
# plot roc _auc plot 
plot(roc ,main ="Logistic Regression base Roc ")
################## model 1 ########################################### 
################## test data prediction   ############################
#[1] "accuracy 0.92"
#[2] "precision 0.68"
#[3] "recall 0.01"
#[4] "fpr 0.01"
#[5] "fnr 0.73"
#[6] "f1 0.03"
#Area under the curve: 0.8585

################################### Naive Bayes #############################



# coverting target to factor 
train$Y = factor(train$Y ,levels = c(0,1))
# train model 
nb_model  =naiveBayes(Y~.  , data =train )  

# model summary  

#PREDICTING PROBALITY
y_prob =predict(nb_model , test[-201]  ,type='raw')
# convert   probality to class according to thresshold
y_pred = ifelse(y_prob[,2] >0.5, 1, 0)
#create confusion matrix 
conf_matrix= table(test[,201] , y_pred)
#print model accuracy
getmodel_accuracy(conf_matrix)
################## test data prediction  model #########################
# [1] "accuracy 0.92"
# [1] "precision 0.72"
# [1] "recall 0.02"
# [1] "fpr 0.02"
# [1] "fnr 0.64"
# [1] "f1 0.03"
#Area under the curve: 0.8866

# get Auc 
roc=roc(test[,201], y_prob[,2] )
print(roc)
# plot Roc_Auc curve 
plot(roc ,main="Roc _ auc  Naive Bayes model 1 ")

################################################################################
########################## xgboost #############################################

# convertion target varible to factor 
train$Y <- as.numeric(as.factor(train$Y)) - 1 
test$Y <- as.numeric(as.factor(test$Y)) - 1 
over$Y <- as.numeric(as.factor(over$Y)) - 1 

# coverting data into dmatrix as it required in xgboost 
trainD =xgb.DMatrix(data =as.matrix(train[,-201]),label= train$Y)
testD =xgb.DMatrix(data=as.matrix(test[,-201]) ,label  =test$Y)
overD =xgb.DMatrix(data =as.matrix(over[,-201]) ,label=over$Y)

#################### using train data without oversampling #####################


###prameters  used 
# max.depth : max depth tree is allowed to grow
# eta: similar to learing rate 
# nrounds: maximum round algorithmis allowed to run 
# scale_pos_weight: make postive weight 11 times more than neagtive 

#train model  
xgb1 = xgb.train(
  data = trainD,
  max.depth = 3,
  eta = 0.1,
  nrounds = 500,
  scale_pos_weight =11,
  objective = "binary:logistic"
)

#PREDICTING PROBALITY
y_prob =predict(xgb1 , as.matrix(test[,-201] ) )
# convert   probality to class according to thresshold
y_pred = ifelse(y_prob >0.5, 1, 0)
#create confusion matrix 
conf_matrix= table(test[,201] , y_pred)
#print model accuracy
getmodel_accuracy(conf_matrix)
# get roc 
roc=roc(test[,201], y_prob )
print(roc)
# plot roc
plot(roc ,main="Roc _ auc  xgboost model 1 ")
########## test data prediction ##################################
# [1] "accuracy 0.81"
# [1] "precision 0.32"
# [1] "recall 0.18"
# [1] "fpr 0.18"
# [1] "fnr 0.21"
# [1] "f1 0.23"
# Area under the curve: 0.8856

