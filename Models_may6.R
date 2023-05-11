cat("\014") # clear console panel
rm(list = ls()) # clear global environment
graphics.off() # close all graphics


########################### DATA MODELLING ###########################
# Import libraries
library(infotheo)
library(dplyr)
library(caret)
library(chisquare)
library(smotefamily)
library(imbalance)
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)
library(e1071)
library("kernlab")
library(DMwR2)
library(ROSE)
library(randomForest)
library(ROCR)
library(deepnet)

# Load a "bank marketing" dataset
bank_marketing = read.csv("/Users/reneec/Adelphi University/Strategic Capstone Project/Bank Marketing-20230331/Bank Marketing-separeted columns.csv")
dim(bank_marketing)
# Remove 5 social and economic variables as per the instructions mentioned in the dataset description
bank_marketing = subset(bank_marketing, select = -c(emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed))
dim(bank_marketing)
# Check the structure of the new dataframe
str(bank_marketing) #the new dataframe comprises of 16 variables and 41188 observations

############################### DECISION TREE #################
############################### Data Splitting ##################################
# Proportion table to see how imbalanced an output variable is
barplot(prop.table(table(bank_marketing_final$subscription)),
        col = rainbow(2),
        ylim = c(0, 0.7),
        main = "Subscription")

# Set seed to random
set.seed(345)
# Importing SMOTE balanced data into training & testing
training <- read.csv("/Users/reneec/Adelphi University/Strategic Capstone Project/training_ros.csv")
testing <- read.csv("/Users/reneec/Adelphi University/Strategic Capstone Project/testing_ros_bal.csv")

# Check dimension of the dataframe
dim(training)
dim(testing)

# Create a classification decision tree model to predict the subscription variables 
# based on all other variables in the training set
model <- rpart(subscription~., data = training)
rpart.plot(model)

# Make predictions based on the testing set
pred_dt <- predict(model, testing, type = 'vector')

# Making a Confusion Matrix
predicted_labels <- ifelse(pred_dt > 0.5, 1, 0)
true_labels <- testing$subscription
cm <- table(predicted_labels, true_labels)
cm

# Extracting True (positives & negatives) & False (positives & negatives) from the matrix
TP <- cm[2, 2]
TN <- cm[1, 1]
FP <- cm[1, 2]
FN <- cm[2, 1]
# Printing True positives/negatives, and false positives/negative for each model to confirm they're all different
print(TP)
print(TN)
print(FP)
print(FN)

# Calculate the metrics of the model
# 1. Sensitivity
sensitivity <- TP / (TP + FN)
sensitivity

# 2. Specificity
specificity <- TN / (TN + FP)
specificity

# 3. Precision
precision <- cm[2, 2] / sum(cm[, 2])
precision

# 4. G-Mean (Geometric Mean)
gmean <- sqrt(sensitivity * specificity)
gmean

# 5. Accuracy
accuracy <- sum(diag(cm)) / sum(cm)
accuracy

# 6. AUC (Area Under the Curve)
auc <- auc(testing$subscription, pred_dt)
auc

# Compute the ROC curve and AUC value
roc_obj <- roc(testing$subscription, pred_dt)
auc_val <- auc(testing$subscription, pred_dt)

# Plot the ROC curve with AUC value
plot(roc_obj, print.auc = TRUE, main = paste("AUC =", auc_val))

######################## Scenario 3: DO NOTHING + Random Forest #################
# Check for missing values
sum(is.na(training))
# Check dimension of the training set
dim(training)

# Pass the training set to train Random Forest model
rf <- randomForest(subscription~., data = training)
rf

# Making predictions based on model
pred_rf <- predict(rf, newdata = testing)

# Making a Confusion Matrix
predicted_labels <- ifelse(pred_rf > 0.5, 1, 0)
true_labels <- testing$subscription
cm <- table(predicted_labels, true_labels)
cm

# Printing True positives/negatives, and false positives/negative for each model to confirm they're all different
TP <- cm[2, 2]
TN <- cm[1, 1]
FP <- cm[1, 2]
FN <- cm[2, 1]
print(TP)
print(TN)
print(FP)
print(FN)

# Calculate the metrics of the model
# 1. Accuracy
accuracy <- sum(diag(cm)) / sum(cm)
accuracy

# 2. Precision
precision <- cm[2, 2] / sum(cm[, 2])
precision

# 3. Recall
recall <- cm[2, 2] / sum(cm[2, ])
recall

# 4. F1 Score
f1score <- 2 * precision * recall / (precision + recall)
f1score

# 5. Specificity
specificity <- TN / (TN + FP)
specificity

# 8. Sensitivity
sensitivity <- TP / (TP + FN)
sensitivity

# 6. G-Mean (Geometric Mean)
gmean <- sqrt(sensitivity * specificity)
gmean

# 7. AUC (Area Under the Curve)
auc <- auc(testing$subscription, pred_rf)
auc

# Compute the ROC curve and AUC value
roc_obj <- roc(testing$subscription, pred_rf)
auc_val <- auc(testing$subscription, pred_rf)

# Plot the ROC curve with AUC value
plot(roc_obj, print.auc = TRUE, main = paste("AUC =", auc_val))

# Artificial Neural Network Algorithm - Perceptron
# Training the model
perceptron <- train(subscription ~ ., data = training, method = "svmLinear", trControl = trainControl(method = "cv"))

# Making predictions on testing data
pred_percep <- predict(perceptron, newdata = testing)

# Generating Confusion Matrix
predicted_labels <- ifelse(pred_percep > 0.5, 1, 0)
true_labels <- testing$subscription
cm <- table(predicted_labels, true_labels)
cm

# Printing True positives/negatives, and false positives/negative for each model to confirm they're all different
TP <- cm[2, 2]
TN <- cm[1, 1]
FP <- cm[1, 2]
FN <- cm[2, 1]
print(TP)
print(TN)
print(FP)
print(FN)

# Calculate the metrics of the model
# 1. Accuracy
accuracy <- sum(diag(cm)) / sum(cm)
accuracy

# 2. Precision
precision <- cm[2, 2] / sum(cm[, 2])
precision

# 5. Specificity
specificity <- TN / (TN + FP)
specificity

# 8. Sensitivity
sensitivity <- TP / (TP + FN)
sensitivity

# 6. G-Mean (Geometric Mean)
gmean <- sqrt(sensitivity * specificity)
gmean

# 7. AUC (Area Under the Curve)
auc <- auc(testing$subscription, pred_percep)
auc

# Compute the ROC curve and AUC value
roc_obj <- roc(testing$subscription, pred_percep)
auc_val <- auc(testing$subscription, pred_percep)

# Plot the ROC curve with AUC value
plot(roc_obj, print.auc = TRUE, main = paste("AUC =", auc_val))

######################## LINEAR REGRESSION MODEL ########################
#data <- read.csv("/Users/reneec/Adelphi University/Strategic Capstone Project/bank_mkt_important_features.csv")
#trainIndex <- sample(1:nrow(data), floor(0.70*nrow(data)))
training <- read.csv("/Users/reneec/Adelphi University/Strategic Capstone Project/training_ros.csv")
testing <- read.csv("/Users/reneec/Adelphi University/Strategic Capstone Project/testing_ros_bal.csv")

# Training the model on train data set
model_lr <- lm(subscription ~., data = training)
summary(model_lr)

# Making prediction on the LR model using test data set
pred_lr <- predict(model_lr, newdata = testing)
# Calculate the mean squared error of the predictions
mse <- mean((testing$subscription - pred_lr)^2)
print(mse)

# Generating Confusion Matrix
predicted_labels <- ifelse(pred_lr > 0.5, 1, 0)
true_labels <- testing$subscription
cm <- table(predicted_labels, true_labels)
cm

# Extracting True (positives & negatives) & False (positives & negatives) from the matrix
TP <- cm[2, 2]
TN <- cm[1, 1]
FP <- cm[1, 2]
FN <- cm[2, 1]
# Printing True positives/negatives, and false positives/negative for each model to confirm they're all different
print(TP)
print(TN)
print(FP)
print(FN)

# Calculate the metrics of the model
# 1. Sensitivity
sensitivity <- TP / (TP + FN)
sensitivity

# 2. Specificity
specificity <- TN / (TN + FP)
specificity

# 3. Precision
precision <- cm[2, 2] / sum(cm[, 2])
precision

# 4. G-Mean (Geometric Mean)
gmean <- sqrt(sensitivity * specificity)
gmean

# 5. Accuracy
accuracy <- sum(diag(cm)) / sum(cm)
accuracy

# 6. AUC (Area Under the Curve)
auc <- auc(testing$subscription, pred_lr)
auc

# Compute the ROC curve and AUC value
roc_obj <- roc(testing$subscription, pred_lr)
auc_val <- auc(testing$subscription, pred_lr)

# Plot the ROC curve with AUC value
plot(roc_obj, print.auc = TRUE, main = paste("AUC =", auc_val))


################################ DEEP BELIEF NETWORK (DBN) ################################
#data <- read.csv("/Users/reneec/Adelphi University/Strategic Capstone Project/bank_mkt_important_features.csv")
#trainIndex <- sample(1:nrow(data), floor(0.70*nrow(data)))
training <- read.csv("/Users/reneec/Adelphi University/Strategic Capstone Project/training_ros.csv")
testing <- read.csv("/Users/reneec/Adelphi University/Strategic Capstone Project/testing_ros_bal.csv")

# Create the DBN
model_dbn <- dbn.dnn.train(training, hidden = c(10, 5), output = 3, activationfun = "sigmoid")

# Print the summary of the DBN
summary(dbn)


# Generating Confusion Matrix
predicted_labels <- ifelse(pred_lr > 0.5, 1, 0)
true_labels <- testing$subscription
cm <- table(predicted_labels, true_labels)
cm












