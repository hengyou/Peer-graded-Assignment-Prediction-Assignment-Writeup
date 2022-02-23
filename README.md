# Peer-graded-Assignment-Prediction-Assignment-Writeup


```{r results='hide', message=FALSE}
library(caret)
library(rattle)
library(corrplot)
```


Load the dataset.
```{r results='hide', message=FALSE}

TrainData <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"),header=TRUE)
dim(TrainData)
TestData <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"),header=TRUE)
dim(TestData)


```{r  message=FALSE}

 # The training dataset is  partitioned into 2 to create a Training set with 70% of the data for the modeling process.  A Test set is created with the remaining 30% for the validations.


set.seed(32343)
inTrain <- createDataPartition(TrainData$classe, p = 0.7, list = FALSE)
trainData <- TrainData[inTrain, ]
testData <- TrainData[-inTrain, ]
dim(trainData)
dim(testData)




NZV <- nearZeroVar(trainData)
trainData <- trainData[, -NZV]
testData  <- testData[, -NZV]
dim(trainData)
dim(testData)


mostlyNA <- sapply(trainData, function(x) mean(is.na(x))) > 0.95
mostlyNATest <- sapply(testData, function(x) mean(is.na(x))) > 0.95
trainData <- trainData[, mostlyNA==F]
testData <- testData[, mostlyNATest==F]

dim(trainData)
dim(testData)


trainData <- trainData[, -(1:5)]
testData <- testData[, -(1:5)]

dim(trainData)
dim(testData)



```{r results='hide', message=FALSE}

correlation <- cor(trainData[, -54])
corrplot(correlation, method="circle")

# The circles with dark colors show highly correlated variables in the graph above. Correlations do not seem to give any analysis points as they are very less.

```



```{r message=FALSE}

trControl <- trainControl(method="cv", number=5)
model_CT <- train(classe~., , method="rpart", data=trainData, trControl=trControl)

fancyRpartPlot(model_CT$finalModel)

predict_train <- predict(model_CT,newdata=testData)

confMatClassTree <- confusionMatrix(testData$classe,predict_train)

#Display confusion matrix and model accuracy

confMatClassTree$table

confMatClassTree$overall[1]

```


```{r message=FALSE}
random_forest <- trainControl(method="cv", number=3, verboseIter=FALSE)
model_RF1 <- train(classe ~ ., data=trainData, method="rf", trControl=random_forest)
model_RF1$finalModel
plot(model_RF1,main="Accuracy of Random forest model by number of predictors")

predict_train <- predict(model_RF1,newdata=testData)

confMatRF <- confusionMatrix(testData$classe,predict_train)




confMatRF


plot(model_RF1$finalModel)
```



```{r  message=FALSE}
set.seed(12345)
GBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
model_GBM  <- train(classe ~ ., data=trainData, method = "gbm", trControl = GBM, verbose = FALSE)
model_GBM$finalModel

predictGBM <- predict(model_GBM, newdata=testData)
confMatGBM <- confusionMatrix(predictGBM, testData$classe)
confMatGBM
```

```{r message=FALSE}

# The predictive accuracies of the above methods are:

#Classification Tree Model: 49.62 %
#Generalized Boosted Model: 98.96 %       
#Random Forest Model: 99.71 %
#
#The Random Forest model has the best accuracy and hence it is used for predictions on the 20 data points from the original testing dataset.


predict_test <- predict(model_RF1, newdata = TestData)
predict_test
```

