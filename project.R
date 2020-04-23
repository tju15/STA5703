library(corrplot)
library(reshape)
library(tidyverse)
library(caret)
library(glmnet)
library(mice)
library(VIM)
library(InformationValue)
library(MASS)
library(RColorBrewer)
library(leaps)

setwd('C:/Users/klwal/OneDrive/Desktop/data_mining_i/project')
data = read.csv(file = 'PHY_TRAIN.csv')
print('Data Read Into Variable')
head(data)

colSums(is.na(data))

# replace null values with column mean
for(i in 1:ncol(data)){
  data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)}

#print( colnames(data[ (colSums(is.na(data)) > 0) ]))

data2 = subset(data, select = -c(exampleid, feat29, feat47, feat48, feat49, feat50, feat51, feat55))
data2$target <- as.factor(data2$target)

# find correlation between features
#correlations <- cor(data)
#corrplot(correlations, method="circle")
## keep only the lower triangle by 
## filling upper with NA
#correlations[upper.tri(correlations, diag=TRUE)] <- NA
#m <- melt(correlations)
## sort by descending absolute correlation
#m <- m[order(- abs(m$value)), ]
## omit the NA values
#dfOut <- na.omit(m)
## if you really want a list and not a data.frame
#listOut <- split(dfOut, 1:nrow(dfOut))
#dfOut

# Split the data into training and test set
set.seed(123)
training.samples <- data2$target %>% 
  createDataPartition(p = 0.7, list = FALSE)
train.data  <- data2[training.samples, ]
test.data <- data2[-training.samples, ]

# step 5 - build logist model without interaction terms
log_mod = glm(target ~ ., data = train.data, family = binomial(link = "logit"))
#summary(log_mod)

# first model stats
glm.probs <- plogis(predict(log_mod, test.data, type = "response"))


optCutOff <- optimalCutoff(test.data$target, glm.probs)

misClassError(test.data$target, glm.probs, threshold = optCutOff) # Misclassification Rate
plotROC(test.data$target, glm.probs) # Finding ROC Curve and AUC
Concordance(test.data$target, glm.probs) # Finding concordance
sensitivity(test.data$target, glm.probs, threshold = optCutOff) # Sensitivity; % of 1s accurately predicted
specificity(test.data$target, glm.probs, threshold = optCutOff) # Specificity; % of 0s accurately predicted
confusionMatrix(test.data$target, glm.probs, threshold = optCutOff) # Confusion Matrix; Col = actuals, Row = predicted

# Performance Measures - input values from confusion matrix to calculate
tp1 = 5319
fn1 = 2198
tn1 = 5343
fp1 = 2139
accuracy1 = (tp1 + tn1)/(tp1 + tn1 + fp1 +fn1)
TPR1 = tp1 / (tp1 + fn1)
FPR1 = fp1 / (fp1 + tn1)
TNR1 = tn1 / (fp1 + tn1) 
Precision1 = tp1 / (tp1 + fp1)
F1 = 2*((Precision*TPR)/(Precision+TPR))
TPR1
FPR1
TNR1
Precision1
accuracy1
F1

# second model
glm.fit = glm(target ~ feat4 + feat8 + feat12 + feat13 + feat14 + 
                feat15 + feat20 + feat31 + feat39 + feat40 + 
                feat42 + feat56 + feat63 + feat66 + feat69 + 
                feat71 + feat75, data = data2, family = binomial(link = "logit"))

summary(glm.fit)

glm.probs <- plogis(predict(glm.fit, test.data, type = "response"))

# find optimal threshold for ROC curve
optCutOff <- optimalCutoff(test.data$target, glm.probs)

misClassError(test.data$target, glm.probs, threshold = optCutOff) # Misclassification Rate
plotROC(test.data$target, glm.probs) # Finding ROC Curve and AUC
Concordance(test.data$target, glm.probs) # Finding concordance
sensitivity(test.data$target, glm.probs, threshold = optCutOff) # Sensitivity; % of 1s accurately predicted
specificity(test.data$target, glm.probs, threshold = optCutOff) # Specificity; % of 0s accurately predicted
confusionMatrix(test.data$target, glm.probs, threshold = optCutOff) # Confusion Matrix; Col = actuals, Row = predicted

# Performance Measures - input values from confusion matrix to calculate
tp2 = 5345
fn2 = 2196
tn2 = 5345
fp2 = 2113
accuracy2 = (tp2 + tn2)/(tp2 + tn2 + fp2 +fn2)
TPR2 = tp2 / (tp2 + fn2)
FPR2 = fp2 / (fp2 + tn2)
TNR2 = tn2 / (fp2 + tn2)
Precision2 = tp2 / (tp2 + fp2)
F12 = 2*((Precision2*TPR2)/(Precision2+TPR2))
TPR2
FPR2
TNR2
Precision2
accuracy2
F12


