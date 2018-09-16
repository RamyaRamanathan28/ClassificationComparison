
library("VIM")
library(tidyverse)
#loading
mushroom <-read.csv("C://Users//User//Desktop//sem_6//mushroom//mushrooms.csv", TRUE, ",")
head(mushroom)


#Preprocessing
mushroom$veil.type = NULL
df<-data.frame(mushroom)
df[df=='?']<-NA
#m=kNN(df, variable=c("stalk.root"), k=6)
#write.csv(m,"New2.csv")
library(mice)
#Table for count of missing values
md.pattern(df)
#Visualising missing values
aggr_plot <- aggr(df, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(df), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
#marginplot(df[c(4,11)]) - compare two coulmns for missing values
#Imputing the missing values
sum(is.na(df))
tempData <- mice(df,m=5,maxit=50,meth='pmm',seed=500)
m<-complete(tempData,1)
#Check the number of missing values
sum(is.na(m))
#Visualising missing values after imputation
aggr_plot <- aggr(m, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(train.complete), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
table(m$stalk.root)
library(caret)
library(rpart)
#Visualizing data
library(ggplot2)
ggplot(m,aes(x=stalk.root,y=gill.size,col=class))+geom_jitter(alpha=0.5)+scale_color_manual(breaks=c("e","p"),values=c("blue","black"))


#splitting data
train=createDataPartition(y=m$class, p=0.80, list=FALSE)
train_set=m[train,]
test_set=m[-train,]
#Decision tree
require(rpart)
library(rattle)	
library(rpart.plot)
tree= rpart(class~., data=train_set, control=rpart.control(cp=.0005))
#Plotting
fancyRpartPlot(tree)
tree_pred=predict(tree, test_set, type='class')
#Accuracy
mean(tree_pred==test_set$class)
#Confusion matrix
table(tree_pred, test_set$class)

#MODEL 2: SVM
require(e1071)
svm_model <- svm(class~., data=train_set, type='C-classification', kernel='radial')
pred_test <-predict(svm_model,test_set)
mean(pred_test==test_set$class)
table(pred_test, test_set$class)
plot(svm_model, test_set)

#MODEL 3: KNN
model_fit = train(class~., method = "knn", data = train_set, trControl = trainControl(method = 'cv', number = 10, classProbs = TRUE));
print(model_fit);
plot(model_fit);

# Classify from our reserved test set.
testing_set_predict = predict(model_fit, newdata = test_set); 
# Verifying our model from the classifications.
table(testing_set_predict, test_set$class);
mean(testing_set_predict==test_set$class)

'test_set$Correct = testing_set_predict == test_set$class;
accuracy = length(testing_set$Correct[testing_set$Correct == TRUE]) / length(testing_set$Correct);
paste("Training accuracy:", accuracy);'

important_features = varImp(model_fit, scale = FALSE);
print(important_features);
plot(important_features);
 
#MODEL 4: Random Forest
library(randomForest)
rf = randomForest(class ~ ., ntree = 100,data = train_set)
plot(rf)
varImpPlot(rf,  sort = T,n.var=10,main="Top 10 - Variable Importance")
rf_result <- predict(rf ,test_set)
table(rf_result, test_set$class);
mean(rf_result==test_set$class)

#MODEL 5: Boosted Logistic Regression
#trainControl for Boosted Logisitic Regression
fitControl <- trainControl(method = 'repeatedcv', repeats = 5,
                           number = 5, verboseIter = T)

# Run a Boosted logisitic regression over the training set
log.fit <- train(class~.,  data = train_set, 
                 method = "LogitBoost", trControl = fitControl,
                 tuneLength = 5)

# Predict the testing target
log.predict <- predict(log.fit, test_set)
table(log.predict, test_set$class)

mean(log.predict==test_set$class)

confusionMatrix(log.predict, test_set$class)

#MODEL 6: Treebag

fitControl = trainControl(method = "repeatedcv", repeats = 5,
                          number = 5, verboseIter = T)

# Run a Treebag classification over the training set
treebag.fit <- train(class~ .,  data = train_set,
                     method = "treebag", importance = T)

# Predict the testing target
treebag.predict <- predict(treebag.fit, test_set)
mean(treebag.fit==test_set$class)
table(treebag.fit, test_set$class)
confusionMatrix(log.predict, test_set$class)
confusionMatrix(treebag.predict, test_set$class)
