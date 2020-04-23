library(xgboost)
library(data.table)

setwd('C:/Users/klwal/OneDrive/Desktop/data_mining_i/project')
data = read.csv(file = 'PHY_TRAIN.csv')
print('Data Read Into Variable')
head(data)

colSums(is.na(data))

# replace null values with column mean
for(i in 1:ncol(data)){
  data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)}

data2 = subset(data, select = -c(exampleid, feat29, feat47, feat48, 
                                 feat49, feat50, feat51, feat55))

# Split the data into training and test set
set.seed(123)
training.samples <- data2$target %>% 
  createDataPartition(p = 0.7, list = FALSE)
train.data  <- data2[training.samples, ]
test.data <- data2[-training.samples, ]

set.seed(123)
training.samples <- data2$target %>% 
  createDataPartition(p = 0.7, list = FALSE)
train.data2  <- data2[training.samples, ]
test.data2 <- data2[-training.samples, ]

train = setDT(train.data)
test = setDT(test.data)

labels <- train$target 
ts_label <- test$target
new_tr <- model.matrix(~.+0,data = train.data[,-c("target"),with=F]) 
new_ts <- model.matrix(~.+0,data = test.data[,-c("target"),with=F])

# for use in xgboost model
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

# default params
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, 
               gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

# cv to find best nrounds
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T, 
                 stratified = T, print_every_n = 2, early_stop_round = 10, maximize = F)

min(xgbcv$test.error.mean)

# default - model training
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 34, 
                   watchlist = list(val=dtest,train=dtrain), print_every_n = 10, 
                   early_stop_round = 10, maximize = F , eval_metric = "error")
# model prediction
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

# confusion matrix
library(caret)
confusionMatrix (xgbpred, ts_label)
# Finding ROC Curve and AUC
plotROC(xgbpred, ts_label) 

# view variable importance plot
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:20]) 

# Performance Measures - input values from confusion matrix to calculate
tp1 = 5266
fn1 = 1911
tn1 = 5572
fp1 = 2251
accuracy1 = (tp1 + tn1)/(tp1 + tn1 + fp1 +fn1)
TPR1 = tp1 / (tp1 + fn1)
FPR1 = fp1 / (fp1 + tn1)
TNR1 = tn1 / (fp1 + tn1) 
Precision1 = tp1 / (tp1 + fp1)
F1 = 2*((Precision1*TPR1)/(Precision1+TPR1))
TPR1
FPR1
TNR1
Precision1
accuracy1
F1

library(mlr)

# create tasks 
traintask <- makeClassifTask (data = train.data2, target = "target")
testtask <- makeClassifTask (data = test.data2,target = "target")

# convert characters to factors
# create learner
lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.1)

# set parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")), 
                        makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
                        makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                        makeNumericParam("subsample",lower = 0.5,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1), 
                        makeIntegerParam("nrounds", lower = 15L, upper = 150L))

# set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

# search strategy
ctrl <- makeTuneControlRandom(maxit = 50)

# set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

# parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)

# set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

# train model
xgmodel <- train(learner = lrn_tune,task = traintask)

# predict model
xgpred <- predict(xgmodel,testtask)
calculateConfusionMatrix(xgpred)

# transform ytrue and ypred to numeric for plotting ROC curve
a = as.numeric(as.character(xgpred$data$response))
b = as.numeric(as.character(xgpred$data$truth))

# Finding ROC Curve and AUC
plotROC(a,b) 

getHyperPars(lrn)
featimp = getFeatureImportance(xgmodel)
featimp$res$importance

# Performance Measures - input values from confusion matrix to calculate
tp2 = 5358
fn2 = 2159
tn2 = 5599
fp2 = 1884
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


