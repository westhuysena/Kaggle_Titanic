# Script: titantic.R
# Author: Andre van der Westhuysen, 05/08/19
# Purpose: Script to perform wine quality classification

rm(list=ls())

# 1. Load libraries
library(readr)  #For reading .csv file
library(ggplot2)
library(stringr)
library(MASS)  #For Probabilistic Ordinal Logistic Regression
library(nnet)  #For Multinomial Logistic Regression
library(rpart)  #For CART decision trees
library(C50)  #For C5.0 decision trees
library(e1071)  #For Naive Bayes model
library(caret)  #For setting dummy variables
library(neuralnet)  #For neural network model

# 2. Read and partition the data
# 2.1 Read the data files
cat("Loading data...\n")
train_val <- read_csv("data/train.csv")
test <- read_csv("data/test.csv")

# 2.2 Partition the data into training and test sets, using a 70/30 split
set.seed(268)
titanic_sampling_vector <- createDataPartition(train_val$Survived, p=0.70, list=FALSE)
train <- train_val[titanic_sampling_vector,1:12]
val <- train_val[-titanic_sampling_vector,1:12]

# 3. Feature Engineering
# 3.1 Response
train$Survived = factor(train$Survived)
val$Survived = factor(val$Survived)

# 3.2 Passenger class (Pclass)
train$Pclass = factor(train$Pclass)
val$Pclass = factor(val$Pclass)
test$Pclass = factor(test$Pclass)

# 3.3 Sex
train$Sex = factor(train$Sex)
val$Sex = factor(val$Sex)
test$Sex = factor(test$Sex)

# 3.4 Embarked
train$Embarked = factor(train$Embarked)
val$Embarked = factor(val$Embarked)
test$Embarked = factor(test$Embarked)

# 3.5 Family size (from SibSp and Parch)
train$FamSize <- train$SibSp + train$Parch
train$SibSp = factor(train$SibSp)
train$Parch = factor(train$Parch)
train$FamSize = factor(train$FamSize)

val$FamSize <- val$SibSp + val$Parch
val$SibSp = factor(val$SibSp)
val$Parch = factor(val$Parch)
val$FamSize = factor(val$FamSize)

test$FamSize <- test$SibSp + test$Parch
test$SibSp = factor(test$SibSp)
test$Parch = factor(test$Parch)
test$FamSize = factor(test$FamSize)

# 3.6 Title
train$Title <- word(regmatches( train$Name, gregexpr("(?<=, ).*(?= .)", train$Name, perl = TRUE ) ), 1)
train$Title = factor(train$Title)

val$Title <- word(regmatches( val$Name, gregexpr("(?<=, ).*(?= .)", val$Name, perl = TRUE ) ), 1)
val$Title <- gsub('Don.', 'Mr.', val$Title)  #Impute value because it does not occur in train set
val$Title <- gsub('Jonkheer.', 'Master.', val$Title) 
val$Title <- gsub('Lady.', 'Mrs.', val$Title)
val$Title <- gsub('Mme.', 'Mrs.', val$Title)
val$Title <- gsub('Ms.', 'Mrs.', val$Title)
val$Title <- gsub('Sir.', 'Mr.', val$Title)
val$Title <- gsub('the', 'Mr.', val$Title)
val$Title = factor(val$Title)

test$Title <- word(regmatches( test$Name, gregexpr("(?<=, ).*(?= .)", test$Name, perl = TRUE ) ), 1)
test$Title <- gsub('Dona.', 'Mrs.', test$Title)  #Impute value because it does not occur in train set
test$Title <- gsub('Ms.', 'Mrs.', test$Title)
test$Title = factor(test$Title)

summary(train)
View(train)
summary(val)
View(val)
summary(test)
View(test)

# 4. Visualize results
# 4.1 Pclass
ggplot(train, aes(Pclass, fill=Survived)) + 
  geom_bar() + 
  labs(title="Bar Chart of Pclass by Survived (0=No, 1=Yes)", x="Pclass (-)", y="Frequency")
ggsave('bar_pclass.png', width=7, height=5)

# 4.2 Sex
ggplot(train, aes(Sex, fill=Survived)) + 
  geom_bar() + 
  labs(title="Bar Chart of Sex by Survived (0=No, 1=Yes)", x="Sex (-)", y="Frequency")
ggsave('bar_sex.png', width=7, height=5)

# 4.3 Embarked
ggplot(train, aes(Embarked, fill=Survived)) + 
  geom_bar() + 
  labs(title="Bar Chart of Embarked by Survived (0=No, 1=Yes)", x="Embarked (-)", y="Frequency")
ggsave('bar_embarked.png', width=7, height=5)

# 4.4 FamSize
ggplot(train, aes(FamSize, fill=Survived)) + 
  geom_bar() + 
  labs(title="Bar Chart of FamSize by Survived (0=No, 1=Yes)", x="FamSize (-)", y="Frequency")
ggsave('bar_famsize.png', width=7, height=5)

# 4.5 Title
ggplot(train, aes(Title, fill=Survived)) + 
  geom_bar() + 
  labs(title="Bar Chart of Title by Survived (0=No, 1=Yes)", x="Title (-)", y="Frequency")
ggsave('bar_title.png', width=7, height=5)

# 4.6 Age
ggplot(train, aes(Survived, Age)) + 
  geom_boxplot(aes(fill=Survived), outlier.shape=4) + 
  labs(title="Boxplot of Age by Survived (0=No, 1=Yes)", x="Survived (-)", y="Age")
ggsave('box_age.png', width=7, height=5)

# 4.7 Fare
ggplot(train, aes(Survived, Fare)) + 
  geom_boxplot(aes(fill=Survived), outlier.shape=4) + 
  labs(title="Boxplot of Fare by Survived (0=No, 1=Yes)", x="Survived (-)", y="Fare")
ggsave('box_fare.png', width=7, height=5)

# 5. Build a CART decision tree model
# 5.1 Train the model
cat("\nFitting CART tree model...\n")
cart_model <- rpart(Survived ~ Pclass + Sex + Embarked + FamSize + Title + Age + Fare, method="class", data=train, cp=0.005)

# 5.2 Evaluate CART tree results
cat("Evaluate the CART tree model...\n")
# (a) Plot cross-validation results
png("tree_cp.png", width=4.25, height=3.25, units="in", res=1200, pointsize=5)
plotcp(cart_model)
dev.off()

# (b) Print model training results
printcp(cart_model)

# (c) Print results
print(cart_model)

# (d) Print detail results
summary(cart_model)

# (e) Plot decision tree, including labels
png("tree_structure.png", width=4.25, height=3.25, units="in", res=300, pointsize=5)
plot(cart_model, uniform=TRUE)
text(cart_model, use.n=TRUE, all=TRUE, cex=.6)
dev.off()

# 5.3 Check accuracy on TRAIN data
cat("Accuracy of CART model on TRAIN data:\n")
train_predictions <- predict(cart_model, train, type="class")
print(mean(train_predictions == train$Survived))

# 5.4 Check accuracy on VAL data
cat("Accuracy of CART model on VAL data:\n")
val_predictions <- predict(cart_model, val, type="class")
print(mean(val_predictions == val$Survived))

# 5.5 Compute predictions for TEST data
cat("Compute predictions for CART model on TEST data:\n")
test_predictions <- predict(cart_model, test, type="class")
print(test_predictions)

# 6. Write predictions to file for submission
output <- data.frame(test$PassengerId, test_predictions)
colnames(output) = c("PassengerId","Survived")
write.table(output, "cart_submission.csv", append = FALSE, sep = ",", dec = ".", row.names = FALSE, col.names = TRUE, quote = FALSE)
