library(readxl)
library(leaps)
library(ggplot2)
library(corrplot)
library(ggpubr)
library(randomForest)
library(caret)
library(olsrr)
library(tidyverse)
library(caret)


Data = read_excel("C:/Users/Desktop/insurance.xlsx")

## Visualization
View(Data)
plot(Data)

A = ggplot(Data, aes(x=sex ,y=expenses))+geom_point()
B = ggplot(Data, aes(x=children, y=expenses))+geom_point()
C = ggplot(Data, aes(x=bmi, y=expenses))+geom_point()
D = ggplot(Data, aes(x= region ,y=expenses))+geom_point()
E = ggplot(Data, aes(x=smoker, y=expenses))+geom_point()
F = ggplot(Data, aes(x=age , y=expenses))+geom_point()

ggarrange(A, B, C, D, E, F, ncol = 3, nrow = 2)

## check if there has null data cell (Result : There is no null data)
is.null(Data)

## One-hot Encoding
Data1 = Data
Data1$sex = ifelse ( Data$sex == "male", 1, 0 )
Data1$smoker = ifelse ( Data$smoker == "yes", 1, 0 )
Data1$NW = ifelse ( Data$region == "northwest", 1, 0 )
Data1$NE = ifelse ( Data$region == "northeast", 1, 0 )
Data1$SW = ifelse ( Data$region == "southwest", 1, 0 )
Data1 = Data1[,c(-6)]


## Plot the correlation with R package
## Result: We can notice expenses and smoker Seemly have significant positive relation
cor = cor(Data1)
corrplot(cor)



## Spliting smoking or not into two parts
ggplot(Data, aes(x=sex, y=expenses,fill = region)) +
  geom_boxplot() + facet_wrap(~smoker) 
## It is true that smokers have significantly higher expenses than non-smokers
## the difference in the region does not seem to affect the expenses

## Divided into 6 charts according to the number of children, 
## we can find that the number of children has a tendency to lower the expenses.
ggplot(Data, aes(x= age, y=expenses, col=bmi)) + 
  geom_point() + facet_wrap(~children)


## We draw these two variables and the expense,
## Result: It is very obvious that smoking will increase the 
## expense on average (red and blue)
ggplot(Data, aes(x= bmi, y=expenses, col=smoker))+ 
  geom_point() 

plot(Data$expenses,xlab = "",ylab="Expenses",main="Histogram of expenses",pch=20,col = "red")





## Check influential points
lm = lm(expenses~.,Data)
critical_number = 3*sqrt((ncol(Data)-1)/nrow(Data))
Influence = summary(influence.measures(S))
Influence[order(Influence[,10],decreasing=TRUE)[1:13],13] 
## Observe top10 largest DFFIT distance, they are not too significant(low cook'd value) 
## Therefore, we will NOT delete any row of data (Total 1388 samples)
Data[c(891,253,795,820),]
Data_small = Data_small[-c(order(Influence[,10],decreasing=TRUE)[1:13]),]




## Random Forest
ind <- sample(2, nrow(Data), replace = TRUE, prob = c(0.8, 0.2))
train <- Data[ind==1,]
test <- Data[ind==2,]

rf <- randomForest(x = train[,-7],y = train$expenses,
                   proximity=TRUE, xtest = test[,-7], ytest = test$expenses)
varImpPlot(rf, sort = TRUE)
plot(rf)



## Result on traning Data of Random Forest: 
pred<-rf$predicted
meanY<-sum(train$expenses)/length(train$expenses)
varpY<-sum((train$expenses-meanY)^2)/length(train$expenses)
mseY<-sum((train$expenses-pred)^2)/length(train$expenses)
maeY = sum(abs(train$expenses-pred))/length(train$expenses)
r2<-(1-(mseY/varpY))
RmseY = sqrt(mseY)
r2
mseY
RmseY
maeY


## Result on testing Data of Random Forest: 
pred_2<-rf$test$predicted
meanY_2<-sum(test$expenses)/length(test$expenses)
varpY_2<-sum((test$expenses-meanY_2)^2)/length(test$expenses)
mseY_2<-sum((test$expenses-pred_2)^2)/length(test$expenses)
maeY_2 = sum(abs(test$expenses-pred_2))/length(test$expenses)
r2_2<-(1-(mseY_2/varpY_2))
RmseY_2 = sqrt(mseY_2)
r2_2
mseY_2
RmseY_2
maeY_2

## Plot the predict VS actual
plot(pred_2,pch=20,col="red",main = "predict VS actual",ylab = "",xlab = "")
points(test$expenses,pch=20,col="green")
legend("bottom",inset = c(0, -0.5),legend=c("predict", "actual"),
       col=c("red", "green"),cex=0.8,pch=20,xpd = TRUE)





## (Simple linear regression)

## Using two method to select variables
## Variables selection(1)
LM = lm(expenses~.,Data)
plot(ols_step_both_p(LM,details = TRUE))
LM_modify = lm(expenses~.-sex,Data)
LM_modify2 = update(LM_modify,.~.-region)
anova(LM_modify,LM_modify2)

## select variabals(2)
X = as.matrix(Data1[,-6])
y = as.matrix(Data1[,6])

L2 = leaps(X,y,method="adjr2",nbest=3)
L2M = cbind(L2$which+0,L2$adjr2)
colnames(L2M) = c(colnames(X),"adjr2")
A = L2M[order(L2$adjr2,decreasing = T),]
A[1:5,]

L3 = leaps(X,y,method="Cp",nbest=3)
L3M = cbind(L3$which+0,L3$size,L3$Cp)
colnames(L3M) = c(colnames(X),"size","Cp")
B = L3M[order(L3$Cp,decreasing = FALSE),]
B[1:5,]

## We can see the matrix A,B variable selection by adjr2 & Cp,
## I would remove the variable "sex", "region", which is identical to the above method 
## by using package "olsrr"


## model construction(fesidual analysis, transformation... 
## Simple_model = lm(expenses~age+bmi+children+smoker,Da1)
car::vif(Simple_model)    ## VIF: multicollinearitysummary(Simple_model)
anova(Simple_model)       ## variable selection seems reasonable(mfrow=c(2,2))
plot(step(Simple_model,direction="both"))  ## rstandard(Simple_model)
y_hat = Simple_model$fitted.values

## QQPLOT (Check if residuals are ally dist.)
par(mfrow = c(1,2))
qqnorm(es,ylab="Standardized Residuals",xlab="Normal Scores", main="QQ Plot")+
  abline(a=0,b=1,col="red")

## (Residuals vs Fitted value)
plot(y_hat, es ,xlab="Fitted Values",ylab="Errors",ylim=c(-4,5))+
  abline(0,0,col="red")

## MSE
SSE = sum((Simple_model$residuals)^2)
RMSE = sqrt(SSE / (1388-5-1))

## MAE 
SAE = sum(abs(Simple_model$residuals))
MAE = SAE/1388


## Transformation
bc = car::boxCox(Simple_model)
lambda <- bc$x[which.max(bc$y)]
lambda
## (By using boxcox to find the lambda of lambda ~=0,
## it means we should use log transformation
modelrans = lm(log(expenses)~age+bmi+children+smoker, data=Da1)
summary(model_trans)

SSE = sum((model_trans$residuals)^2)
RMSE = sqrt(SSE / (1388-5-1))

## MAE 
SAE = sum(abs(model_trans$residuals))
MAE = SAE/1388


es_trans = rstandard(model_trans)
y_hat_trans = model_trans$fitted.values

qqnorm(es_trans,ylab="Standardized Residuals",xlab="Normal Scores", main="QQ Plot")+
  abline(a=0,b=1,col="red")

plot(y_hat_trans, es_trans ,xlab="Fitted Values",ylab="Errors",ylim=c(-4,5))+
  abline(0,0,col="red")

## It is awful even after transformation, 
## We will try polynomial regression in next section

## Lack of Fit Test
lof = lm(expenses~factor(age)+factor(bmi)+factor(children)+factor(smoker),Da1)
anova(Simple_model,lof)
## (LOF test)
## Leave one out Cross validation
train.control <- trainControl(method = "LOOCV")

reg_tran = train(log(expenses)~age+bmi+children+smoker, data = Da1,
                 method = "lm",trControl = train.control)

reg = train(expenses~age+bmi+children+smoker, data = Da1, method = "lm",
            trControl = train.control)

reg_tran
reg

## Conclusion:
Simple_model   ## Without transformation
model_trans    ## With transformation




## Self-defined generate interaction variables
## (Polynomial regression)
## (i) 
## Generate variables nametion(data){
Xppend = function(data){
  n = length(data)
  x = c(colnames(data))
  for (i in 1:n){
    for (j in i:n){
      x = append(x,paste(as.character(x[i]),as.character(x[j]),sep=" "))
    }
  }
  return(x)
}

span = function(data){
  n = length(data)
  for (i in 1:n){
    for (j in i:n){
      data = cbind(data,data[,i]*data[,j])
    }
  }
  return(data)
}

## Remove redundant variables: poly_Data
poly_Data = Data1[,c(1,3,4,5,6)]   

## Generating corresponding ind(poly_Data[,5],span(poly_Data[,-5]))
Xppend(generate(poly_Data[,-5]),"y",0)   
set.seed(validation = sample(1:length(Data2[,1]),1000)  ## Random generating training, testing setngth(Data2[,1]),train)
         Training = Data2[train,]
         Test = Data2[test,]
         
         
         ## Building model
         all = lm(y~variables selectionng)
         step(all,direction = "both")  ## ?????ܼ?
         
         Training =variables selectiong[c('y',"age","bmi","smoker",'children',"age age","bmi bmi",'bmi smoker')]
         Test = Test[c('y',"age","bmi","smoker",'children',"age age","bmi bmi",'bmi smoker')]
         
         
         
         lm_polyraining)
summary(lm_poly) ## Radj_2 = 0.842
anova(lm_poly)

## Wald test(m_poly$residuals)^2)
MSE = SSE/(1000-1-8) 
RMSE= sqrt(MSE)
SAE =  sum(abs(lm_poly$residuals))
MAE = SAE/1000

plot(lm_poly$fitted.values,rstandard(lm_poly))+abline(0,0,col='red')
qqnorm(rstandard(lm_poly))+abline(0,1,col='red')

## Residual analysis
## Still seems problematic but it indeed improves after adding interacting variables
weights1 = 1 / lm(abs(lm_poly$residuals) ~ lm_poly$fitted.values)$fitted.values^2
lm_poly_weight = lm(y~.,Training, weights = weights1)

plot(lm_poly_weight$fitted.values,rstandard(lm_poly_weight))+abline(0,0,col='red')
qqnorm(rstandard(lm_poly_weight))+abline(0,1,col='red')

## (Method 2) transformation
bc2 = car::boxCox(lm_poly)
lambda2 <- bc2$x[which.max(bc2$y)]
lambda2 
## lambda2 = 0.2626263 ~= 0.25
## lm_poly_trans transformation
plot(lm_poly_trans$fitted.values,rstandard(lm_poly_trans))+abline(0,0,col='red')

summary(lm_poly_trans)


## Result :
SSE = sum((lm_poly_trans$residuals)^2)
MSE = SSE/(1000-1-8) 
RMSE= sqrt(MSE)
SAE =  sum(abs(lm_poly_trans$residuals))
MAE = SAE/1000

qqnorm(rstandard(lm_poly_trans))+abline(0,1,col='red')

summary(lm_poly_trans)
anova(lm_poly_trans)

car::avPlots(lm_poly)



## ypre = using testing set to validateTest

R2(ypre,Test$y)      ## R^2 = 0.841
MAE( we get ,Tes~t$y)
RMpre,Test$y)



lof2 = lm(y~factor(age)+factor(bmi)+factor(smoker)+factor(children)+factor(I(age*age))+factor(I(bmi*bmi))+factor(I(bmi*smoker)),Training)
anova(lm_poly,lof2)
## (The p-value LOF test is very high.)




## (Cross validation)
poly_cv <- train(expenses~age+bmi+smoker+children+I(age*age)+I(bmi*bmi)+I(bmi*smoker), 
                 data = Da1, method = "lm",trControl = train.control)

poly_cv_tran <- train((expenses^0.25)~age+bmi+smoker+children+
                        I(age*age)+I(bmi*bmi)+I(bmi*smoker), 
                      data = Da1, method = "lm",trControl = train.control)

poly_cv
poly_cv_tran


## Conclusion:
(lm_poly)
(lm_poly_trs)



## Replicating modeling after Redution on Data 
Data_small = Data1[which(Data1$expenses <= 15000),]
S = lm(expenses~.,Data_small)

set.seed(123)
train = sample(1:980,650)
Training = Data_small[train,]
Test = Data_small[test,]

ols_step_both_p(S,details = TRUE)

LM_modify = lm(expenses~.-bmi,Data_small)
Data_small = Data_small[,-3]


X = as.matrix(Data_small[,-5])
y = as.matrix(Data_small[,5])

L2 = leaps(X,y,method="adjr2",nbest=3)
L2M = cbind(L2$which+0,L2$adjr2)
colnames(L2M) = c(colnames(X),"adjr2")
A = L2M[order(L2$adjr2,decreasing = T),]
A[1:5,]

L3 = leaps(X,y,method="Cp",nbest=3)
L3M = cbind(L3$which+0,L3$size,L3$Cp)
colnames(L3M) = c(colnames(X),"size","Cp")
B = L3M[order(L3$Cp,decreasing = FALSE),]
B[1:5,]


Simple_model = lm(expenses~.-NW-SW-NE,Data_small)
summary(Simple_model)
anova(Simple_model)


pred_2 = predict(rf,Test)

plot(pred_2,pch=20,col="red",main = "predict VS actual",ylab = "",xlab = "")
points(Test$y,pch=20,col="green")
legend("bottom",inset = c(0,-0.2),legend=c("predict", "actual"),
       col=c("red", "green"),cex=0.3,pch=20,xpd = TRUE)



SSE = sum((Simple_model$residuals)^2)
RMSE = sqrt(SSE / (980-5-1))

## MAE 
SAE = sum(abs(Simple_model$residuals))
MAE = SAE/980

es = rstandard(Simple_model)
y_hat = Simple_model$fitted.values

## QQPLOT (Normally dist.) Check if resuduals are w = c(1,2))
qqnorm(es,ylab="Standardized Residuals",xlab="Normal Scores", main="QQ Plot")+
  abline(a=0,b=1,col="red")

## (Eesiduals vs Fitted value)
plot(y_hat, es ,xlab="Fitted Values",ylab="Errors",ylim=c(-3.5,3.5))+
  abline(0,0,col="red")

bc = car::boxCox(Simple_model)
lambda <- bc$x[which.max(bc$y)]
lambda

## boxcox lambda??=0.4  We take transformation of (y)
model_trans = lm(sqrt(expenses)~.-NE-NW-SW,data=Data_small)
summary(model_trans)

SSE = sum((model_trans$residuals)^2)
RMSE = sqrt(SSE / (980-5-1))

## MAE 
SAE = sum(abs(model_trans$residuals))
MAE = SAE/980

es_trans = rstandard(model_trans)
y_hat_trans = model_trans$fitted.values

qqnorm(es_trans,ylab="Standardized Residuals",xlab="Normal Scores", main="QQ Plot",ylim=c(-1,1))+
  abline(a=0,b=1,col="red")

plot(y_hat_trans, es_trans ,xlab="Fitted Values",ylab="Errors",ylim=c(-3,2.8))+
  abline(0,0,col="red")


reg_tran = train(sqrt(expenses)~.-NE-NW-SW, data = Data_small,
                 method = "lm",trControl = train.control)

reg = train(expenses~.-NE-NW-SW, data = Data_small, method = "lm",
            trControl = train.control)

predict(Simple_model,Data_small[c(14,65,12,45),])



poly_Da = Da_small[,c(-7,-8,-9)]  ## Remove variables: sex, generating corresponding(poly_Da[,-6]),"y",0)

set.seed(123)
train = sample(1:length(Da2[,1]),700)
Training = Da2[train,]
Test = Da2[test,]


## Building Model
all = lm(y~.,Trainivariables selection(all,method = "both")
         
         Training = Training[c('y',"age","sex","children","smoker","age age","bmi children",'children smoker')]
         Test = Test[c('y',"age","sex","children","smoker","age age","bmi children",'children smoker')]
         
         
         lm_poly = lm(n ,Training)
         
         SSE = sum((lm_poly$residuals)^2)
         RMSE = sqrt(SSE / (980-6-1))
         
         ## MAE 
         SAE = sum(abs(lm_poly$residuals))
         MAE = SAE/980
         
         
         
         car::vif(lm_poly)
         summary(lm_poly) ##Radj_2 = 0.9016
         anova(lm_poly)
         ## Wald test qqnorm(rstandard(lm_poly),ylim=c(-1,1))+abline(0,0,col='red')
         qqnorm(rstandard(lm_poly))+abline(0,1,col='red')
         
         ## Residual analysis? = 1 / lm(abs(lm_poly$residuals) ~ lm_poly$fitted.values)$fitted.values^2
         lm_poly_weight = lm(y~.,Training, weights = weights1)
         
         plot(lm_poly_weight$fitted.values,rstandard(lm_poly_weight))+abline(0,0,col='red')
         qqnorm(rstandard(lm_poly_weight))+abline(0,1,col='red')
         
         ## (Method 2) transformation
         bc2 = car::boxCox(lm_poly)
         lambda2 <- bc2$x[which.max(bc2$y)]
         lambda2 
         
         ## lambda2 = 0.74747 ~= 0.75, y_trans take transformation (y^(0.75)Training)
         summary(lm_poly_trans)
         plot(lm_poly_trans$fitted.values,rstandard(lm_poly_trans),ylim=c(-2,2))+abline(0,0,col='red')
         
         SSE = sum((lm_poly_trans$residuals)^2)
         RMSE = sqrt(SSE / (980-6-1))
         
         ## MAE 
         SAE = sum(abs(lm_poly_trans$residuals))
         MAE = SAE/980
         
         qqnorm(rstandard(lm_poly_trans))+abline(0,1,col='red')
         
         summary(lm_poly_trans)
         anova(lm_poly_trans)
         
         car::avPlots(lm_poly)
         
         
         
         poly_cv <- train(expenses~sex+smoker+I(age*age)+I(bmi*children)+I(children*smoker), 
                          data = poly_Da, method = "lm",trControl = train.control)
         
         poly_cv_tran <- train((expenses^0.75)~sex+smoker+I(age*age)+
                                 I(bmi*children)+I(children*smoker), 
                               data = poly_Da, method = "lm",trControl = train.control)
         
         poly_cv_tran
         poly_cv
         
         ypre = prediypre,Test$y)     ## R^2 = 0.924???k
MAE(ypre,Test$y)
RMSE(ypre,Test$y)

predict(lm_poly,Da2[c(14,65,12,45),])


ind <- sample(2, nrow(Da_small), replace = TRUE, prob = c(0.8, 0.2))
train <- Da_small[ind==1,]
test <- Da_small[ind==2,]

rf <- randomForest(x = train[,-6],y = train$expenses,
                   proximity=TRUE, xtest = test[,-6], ytest = test$expenses)
varImpPlot(rf, sort = TRUE)
plot(rf)


## Result in training Data:  
pred<-rf$predicted
meanY<-sum(train$expenses)/length(train$expenses)
varpY<-sum((train$expenses-meanY)^2)/length(train$expenses)
mseY<-sum((train$expenses-pred)^2)/length(train$expenses)
maeY = sum(abs(train$expenses-pred))/length(train$expenses)

r2<-(1-(mseY/varpY))
RmseY = sqrt(mseY)
r2
mseY
RmseY


## Result in testing Data:  test$predicted
meanY_2<-sum(test$expenses)/length(test$expenses)
varpY_2<-sum((test$expenses-meanY_2)^2)/length(test$expenses)
mseY_2<-sum((test$expenses-pred_2)^2)/length(test$expenses)
maeY_2 = sum(abs(test$expenses-pred_2))/length(test$expenses)
r2_2<-(1-(mseY_2/varpY_2))
RmseY_2 = sqrt(mseY_2)
r2_2
mseY_2
RmseY_2
