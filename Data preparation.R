cat("\014")
rm(list = ls()) # clear global environment
graphics.off() # close all graphics

data = read.csv("/Users/reneec/Adelphi University/Strategic Capstone Project/Bank Marketing-20230331/Bank Marketing-separeted columns.csv")

################general functions################
summary(data)
str(data)
table(is.na(data))


################Zero and Near Zero Variance predictors################
library(caret)
data(mdrr)
?mdrr
mdrr<- data.frame(mdrrDescr)
str(mdrr)
data.frame(table(mdrrDescr$nR11))
frequency.ratio<- data.frame(table(mdrrDescr$nR11))[1,2]/data.frame(table(mdrrDescr$nR11))[3,2]
percent.of.unique.values<- (3/nrow(mdrrDescr))*100
data.frame(table(mdrrDescr$nR11)/nrow(mdrrDescr))

nzv <- nearZeroVar(mdrrDescr, saveMetrics= TRUE)
nzv[nzv$nzv,][1:5,]
?nearZeroVar
#zeroVar:zero variance
#nzv: near zero var



################encoding################
#Integer encoding for ordinal categorical variables
data$X <- factor(data$X, levels = c("a","b", "c","d"), labels = c(1,2,3,4), ordered= TRUE)


#One hot encoding (dummy varibles) for nominal categorical variables
install.packages("earth")
library(earth)
data(etitanic)
?etitanic

str(etitanic)
library(caret)
dummies <- dummyVars(survived ~ ., data = etitanic,fullRank=T)
newdata<-(predict(dummies, newdata = etitanic))
newdata<- cbind(etitanic[,2], newdata)
colnames(newdata)[1]<- "Survival"



################Imputation################
#use of mean and median for imputation

data$x[is.na(data$x)]<-mean(data$x,na.rm=TRUE)
data$x[is.na(data$x)]<-median(data$x,na.rm=TRUE)



##kNN imputation
installed.packages("VIM")
library(VIM)
data(iris)
iris
# provide several empty values, 10 in each column, randomly
for (i in 1:ncol(iris)){
  iris[sample(1:nrow(iris), 10, replace=FALSE), i] <- NA
}
summary(iris)
iris2 <- kNN(iris)
summary(iris2)
iris2 <- subset(iris2, select=Sepal.Length:Species)



################Binning################
library(infotheo)
library(dplyr) 
#Equal frequency
data[,"X"] <- discretize(data$X,"equalfreq",4)


#Define your own bins
data$X<- cut(data$X, breaks = c(0, 3, 6, 10), labels=c(1,2,3))
data$X<-as.numeric(data$X)



################Multi dimentional outlier detection###################
#1. outlier detection with clustering
data(iris)
iris3 <- iris[,1:4]
kmeans.result <- kmeans(iris3, centers=3)
# cluster centers
kmeans.result$centers
kmeans.result$cluster
centers <- kmeans.result$centers[kmeans.result$cluster, ]
distances <- sqrt(rowSums((iris3 - centers)^2))
# pick top 10 largest distances
outliers <- order(distances, decreasing=T)[1:10]

# What are the outliers?
print(outliers)
print(iris3[outliers,])

plot(iris3[,c("Sepal.Length", "Sepal.Width")], pch="o", 
     col=kmeans.result$cluster, cex=0.75)
# plot cluster centers
points(kmeans.result$centers[,c("Sepal.Length", "Sepal.Width")], col=1:3, 
       pch=10, cex=3)
# plot outliers
points(iris3[outliers, c("Sepal.Length", "Sepal.Width")], pch="+", col=4, cex=2)

iris3<- iris3[-c(outliers),] #remove outliers from the entire data set


#2. outlier detection with cook's distance method
mod <- lm(Petal.Width ~ ., data=iris)
cooksd <- cooks.distance(mod)
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4/nrow(iris), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4/nrow(iris),names(cooksd),""), col="red")  # add labels
influential <- as.numeric(names(cooksd)[(cooksd > 5/nrow(iris))])
data<- iris[-c(influential),]   #remove outliers from the entire data set


  
