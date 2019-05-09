# Script: titantic.R
# Author: Andre van der Westhuysen, 05/08/19
# Purpose: Script to perform wine quality classification

rm(list=ls())

# 1. Load libraries
library(readr)  #For reading .csv file
library(ggplot2)
library(MASS)  #For Probabilistic Ordinal Logistic Regression
library(nnet)  #For Multinomial Logistic Regression
library(rpart)  #For CART decision trees
library(C50)  #For C5.0 decision trees
library(e1071)  #For Naive Bayes model
library(caret)  #For setting dummy variables
library(neuralnet)  #For neural network model

# 1. Read the data file and convert to factors
cat("Loading data...\n")
train <- read_csv("data/train.csv")
train$Survived = factor(train$Survived)
train$Pclass = factor(train$Pclass)
train$Sex = factor(train$Sex)
train$SibSp = factor(train$SibSp)
train$Parch = factor(train$Parch)
train$Embarked = factor(train$Embarked)
summary(train)
View(train)

# 2. Visualize results
#Visualize Pclass
ggplot(train, aes(Pclass)) + 
  geom_bar() + facet_wrap(~ Survived) +
  labs(title="Bar Chart of Pclass by Survived (0=No, 1=Yes)", x="Pclass (-)", y="Frequency")
ggsave('bar_pclass.png', width=7, height=5)

#Visualize Sex
ggplot(train, aes(Sex)) + 
  geom_bar() + facet_wrap(~ Survived) +
  labs(title="Bar Chart of Sex by Survived (0=No, 1=Yes)", x="Sex (-)", y="Frequency")
ggsave('bar_sex.png', width=7, height=5)

#Visualize Embarked
ggplot(train, aes(Embarked)) + 
  geom_bar() + facet_wrap(~ Survived) +
  labs(title="Bar Chart of Embarked by Survived (0=No, 1=Yes)", x="Embarked (-)", y="Frequency")
ggsave('bar_embarked.png', width=7, height=5)

#Visualize Age
ggplot(train, aes(Survived, Age)) + 
  geom_boxplot(aes(fill=Survived), outlier.shape=4) + 
  labs(title="Boxplot of Age by Survived (0=No, 1=Yes)", x="Survived (-)", y="Age")
ggsave('box_age.png', width=7, height=5)

#Visualize Fare
ggplot(train, aes(Survived, Fare)) + 
  geom_boxplot(aes(fill=Survived), outlier.shape=4) + 
  labs(title="Boxplot of Fare by Survived (0=No, 1=Yes)", x="Survived (-)", y="Fare")
ggsave('box_fare.png', width=7, height=5)

# 3. Build a CART decision tree model
# 3.1 Train the model
cat("\nFitting CART tree model...\n")
cart_model <- rpart(Survived ~ Pclass + Sex + Embarked + Age + Fare, method="class", data=train, cp=0.015)

# 3.2 Evaluate CART tree results
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

# 3.3 Check accuracy on TRAIN data
cat("Accuracy of CART model on TRAIN data:\n")
titanic_predictions <- predict(cart_model, train, type="class")
cat(mean(titanic_predictions == train$Survived))
