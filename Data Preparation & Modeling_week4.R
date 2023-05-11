cat("\014") # clear console panel
rm(list = ls()) # clear global environment
graphics.off() # close all graphics


########################### DATA PREPARATION ###########################
# Import necessary libraries
library(infotheo)
library(dplyr)
library(caret)
library(chisquare)
library(smotefamily)
library(imbalance)
library(caret)
library(pROC)


# Load a "bank marketing" dataset
bank_marketing = read.csv("/Users/reneec/Adelphi University/Strategic Capstone Project/Bank Marketing-20230331/Bank Marketing-separeted columns.csv")
dim(bank_marketing)
# Remove 5 social and economic variables as per the instructions mentioned in the dataset description
bank_marketing = subset(bank_marketing, select = -c(emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed))
dim(bank_marketing)
# Check the structure of the new dataframe
str(bank_marketing) #the new dataframe comprises of 16 variables and 41188 observations

# Check the summary statistics for numeric variables
summary(bank_marketing)


#################### Missing Values & Outlier Detection #################
# Check for missing values
sum(is.na(bank_marketing)) # -> no missing values since the result is 0


# Check for outliers using boxplot approach -> outliers identified in the below variables
boxplot(bank_marketing$age, main = "Boxplot of Age", ylab = "Age", col = "blue")
boxplot(bank_marketing$duration, main = "Boxplot of Duration", ylab = "Duration",col = "blue")
boxplot(bank_marketing$campaign, main = "Boxplot of Campaign", ylab = "Campaign", col = "blue")


################################ Binning ################################ 
# Apply different binning methods to address issues with noisy data
# 1) Age: created 5 bins with 5 different age groups such as 16-25, 25-35, 35-45, 45-55, and 65+
# based on the article "How Age Impacts Consumer Behavior in Retail Banking"
# source: https://thefinancialbrand.com/news/financial-education/age-consumer-behavior-patterns-banking-61246/
bank_marketing$age<- cut(bank_marketing$age, breaks = c(16, 25, 35, 45, 55, 65, 100), labels=c(1,2,3,4,5,6))
# Convert from factor to numeric data type
bank_marketing$age<-as.numeric(bank_marketing$age)
# Check the effect of the above transformation -> no more outliers
boxplot(bank_marketing$age, main = "Boxplot of Age", ylab = "Age", col = "blue")
histogram(bank_marketing$age, main = "Histogram of Age", xlab = "Age", ylab = "Frequency", col = "red")

# 2) Duration: created 3 bins based on equal frequency
# Transform "duration" column from seconds to minutes for more convenient interpretation
bank_marketing$duration = round(bank_marketing$duration/60)
# Apply discretization to reduce noise in continuous variable
bank_marketing[,"duration"] = discretize(bank_marketing$duration,"equalfreq",3)
# Check the effect of the above transformation
hist(bank_marketing$duration, main = "Histogram of duration", xlab = "Duration", ylab = "Frequency", col = "blue")

# 3) Campaign: created 2 bins: the 1st for clients contacted once (value = 1),
# and the 2nd for those contacted more than once (value > 1) DURING THE CAMPAIGN
bank_marketing$campaign <- ifelse(bank_marketing$campaign == 1, 0, 1)
hist(bank_marketing$campaign, main = "Histogram of campaign", xlab = "Campaign", ylab = "Frequency", col = "blue")


######################### Transformation of variables ######################
# Contact: rename the variable to "method of contact", and transform it 
# from a categorical data type into a new binary data type to prepare the variable
# for the modeling part
colnames(bank_marketing)[8] = "method_of_contact"
bank_marketing$method_of_contact = ifelse(bank_marketing$method_of_contact == "telephone", 1, 0) # telephone = 1, cellular = 0
hist(bank_marketing$method_of_contact, main = "Histogram of Method of Contact", xlab = "Method of Contact", ylab = "Frequency", col = "blue")


########### Check for zero and near-zero variance predictors ###############
nzv <- nearZeroVar(bank_marketing, saveMetrics= TRUE)
# Filter rows that have low variance
nzv[nzv$nzv,] 
# pdays was identified as a near zero variance predictor, hence, it needs to
# be transformed to pass the zero and near-zero variance test


######### Transform of zero and near-zero variance predictor ###############
# Step 1: Check for unique values
unique(bank_marketing$pdays) # 999,6,4,3,5,1,0,10,7,8,9,11,2,12,13,14,15,16,21,17,18,22,25,26,19,27,20
# Step 2: Create two bins: 1st with 999 and 0 values, and 2nd with the rest of the values
bank_marketing$pdays <- ifelse(bank_marketing$pdays %in% c(0, 999), 0, 1) # 0 - 39688, 1 - 1500
# Step 3: Check the new frequency 
data.frame(table(bank_marketing$pdays)) # -> 0 - 39688, 1 - 1500
# Step 4: Rename "pdays" to "contacted_previous_campaign" since the variable was transformed in the previous steps
# The values of new variable indicate if clients were contacted or not contacted at all from a PREVIOUS CAMPAIGN
colnames(bank_marketing)[13] = "contacted_previous_campaign"
# plot histogram to see the distribution after grouping 
hist(bank_marketing$contacted_previous_campaign, main = "Histogram of Contacted from Previous Campaign", xlab = "Contacted", ylab = "Frequency", col = "blue")


######################### Encode categorical variables #####################
# One hot encoding (dummy variables) for nominal categorical variable
# Step 1. Remove "month" and "day_of_week" because they are ordinal categorical variables
bank_marketing_encoded <- subset(bank_marketing, select = -c(month, day_of_week))
# Step 2. Use dummyVars to perform one-hot encoding
dummies <- dummyVars(y ~ ., data = bank_marketing_encoded,fullRank=T)
bank_marketing_encoded <- data.frame(predict(dummies, newdata = bank_marketing))

# Convert response/outcome variable 'y' to binary numeric values
bank_marketing$y <- ifelse(bank_marketing$y == "yes", 1, 0)
# Combine the "y", "month", and "day_of_week" variables and the encoded data frame
bank_marketing_final <- cbind(bank_marketing[,c(16, 9, 10)], bank_marketing_encoded)
# Rename all the below added columns
colnames(bank_marketing_final)[1] = "subscription"
colnames(bank_marketing_final)[2] = "month"
colnames(bank_marketing_final)[3] = "day_of_week"

# Integer encoding for ordinal categorical variables
# 1) Month:
unique(bank_marketing$month) # "may" "jun" "jul" "aug" "oct" "nov" "dec" "mar" "apr" "sep"
bank_marketing_final$month <- factor(bank_marketing_final$month, 
          levels = c("mar","apr", "may","jun", "jul", "aug", "sep", "oct", "nov", "dec"), 
          labels = c(1,2,3,4,5,6,7,8,9,10), ordered= TRUE)
# Convert from factor to numeric data type
bank_marketing_final$month<-as.numeric(bank_marketing_final$month)
data.frame(table(bank_marketing$month))
# 2) Day_of_week: 
unique(bank_marketing$day_of_week)
data.frame(table(bank_marketing$day_of_week))
#Integer encoding for ordinal categorical variables
bank_marketing_final$day_of_week <- factor(bank_marketing_final$day_of_week, 
          levels = c("mon","tue", "wed","thu", "fri"), labels = c(1,2,3,4,5), ordered= TRUE)
# Convert from factor to numeric data type
bank_marketing_final$day_of_week<-as.numeric(bank_marketing_final$day_of_week)


#write.csv(bank_marketing_final, "/Users/reneec/Adelphi University/Strategic Capstone Project/Cleaned_BankMarketing.csv", row.names=FALSE)


############################# Data Redundancy #############################
# Load a "bank marketing" dataset
bank_data = read.csv("/Users/reneec/Adelphi University/Strategic Capstone Project/Bank Marketing-20230331/Bank Marketing-separeted columns.csv")
bank_data = subset(bank_data, select = -c(emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed))

# Use Chi-Square Test to determine if there is a significant association 
# between categorical variables ie. input variables vs output variable
chisq.test(bank_data$job, bank_data$y) # significant association between variables b/c p-value < 0.05 significance level
chisq.test(bank_data$marital, bank_data$y) # significant association between variables b/c p-value < 0.05 significance level
chisq.test(bank_data$education, bank_data$y) # significant association between variables b/c p-value < 0.05 significance level
chisq.test(bank_data$default, bank_data$y) # significant association between variables b/c p-value < 0.05 significance level
chisq.test(bank_data$housing, bank_data$y) # no association b/c p-value > 0.05 significance level
chisq.test(bank_data$loan, bank_data$y) # no association b/c p-value > 0.05 significance level
chisq.test(bank_data$contact, bank_data$y) # significant association between variables b/c p-value < 0.05 significance level
chisq.test(bank_data$month, bank_data$y) # significant association between variables b/c p-value < 0.05 significance level
chisq.test(bank_data$day_of_week, bank_data$y) # significant association between variables b/c p-value < 0.05 significance level
chisq.test(bank_data$poutcome, bank_data$y) # significant association between variables b/c p-value < 0.05 significance level

# Use Chi-Square Test to determine associations among input variables
# Create a list of all possible pairs of variables
var_pairs <- combn(names(bank_data), 2, simplify = FALSE)

# Loop through the pairs and perform chi-square tests
for (pair in var_pairs) {
  var1 <- pair[1]
  var2 <- pair[2]
  suppressWarnings(chisq_result <- chisq.test(bank_data[[var1]], bank_data[[var2]]))
  print(paste0("Chi-square test for ", var1, " vs. ", var2, ":"))
  print(chisq_result)
}


############################### DATA MODELING #################################

############################### Data Splitting ################################
############################# Scenario 1: SMOTE ###############################
# Create an index of observations to partition the data into training and testing (70-30 split)
trainIndex <- createDataPartition(bank_marketing_final$subscription, p=0.7, list=F)
# Extract observations in the training set from bank_marketing_final dataset
train <- bank_marketing_final[trainIndex, ]
# Extract observations in the testing set from bank_marketing_final dataset
test <- bank_marketing_final[-c(trainIndex),]

########################### Handling Imbalanced Data ###########################
# The analyzed dataset has an imbalanced output variable "subscription"
# The ratio of "yes" values to "no" values is too low
hist(bank_marketing_final$subscription, main = "Subscription", col = "yellow",
                    xlab = "Subscription")

# To combat the number of imbalanced observations in the response variable, the 
# Synthetic Minority Oversampling Technique has been applied
balanced_bank_mkt <- imbalance::oversample(train, ratio = 0.8, 
                    method = "SMOTE", classAttr = "subscription")
# Plot a histogram to check the distribution of the output variable after using SMOTE
hist(balanced_bank_mkt$subscription, main = "Subscription", xlab = "Subscription", col = "red")
####### CONTINUATION IN PYTHON FILE -> Data_Modeling_Phase.ipynb ################


######################## Scenario 2: DO NOTHING + Decision Tree #################
############################### Data Splitting ##################################
# Proportion table to see how imbalanced an output variable is
barplot(prop.table(table(bank_marketing_final$subscription)),
        col = rainbow(2),
        ylim = c(0, 0.7),
        main = "Subscription")

# Set seed to random
set.seed(123)
# Split data into 75:25 ratio
data <- read.csv("/Users/reneec/Adelphi University/Strategic Capstone Project/bank_mkt_important_features.csv")
trainIndex <- sample(1:nrow(data), floor(0.70*nrow(data)))
training <- data[trainIndex, ]
testing <- data[-trainIndex, ]

# Check dimension of the dataframe
dim(training)
dim(testing)

# Import necessary libraries
library(rpart)
library(rpart.plot)
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
auc <- auc(testing$subscription, pred_dt)
auc

# Compute the ROC curve and AUC value
roc_obj <- roc(testing$subscription, pred_dt)
auc_val <- auc(testing$subscription, pred_dt)

# Plot the ROC curve with AUC value
plot(roc_obj, print.auc = TRUE, main = paste("AUC =", auc_val))



######################## Scenario 3: DO NOTHING + Random Forest #################
# Import necessary libraries
library(randomForest)
library(ROSE)
library(e1071)
library(ROCR)

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



######################## Scenario 4: DO NOTHING + Perceptron #################
# Artificial Neural Network Algorithm - Perceptron
library(e1071)
library("kernlab")
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
auc <- auc(testing$subscription, pred_percep)
auc

# Compute the ROC curve and AUC value
roc_obj <- roc(testing$subscription, pred_percep)
auc_val <- auc(testing$subscription, pred_percep)

# Plot the ROC curve with AUC value
plot(roc_obj, print.auc = TRUE, main = paste("AUC =", auc_val))



# Gaussain Naive Bayes
