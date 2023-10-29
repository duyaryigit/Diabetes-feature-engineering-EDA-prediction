## Diabetes Feature Engineering [EDA] & Prediction 🫀

<p align="center">
  <img src="https://editor.analyticsvidhya.com/uploads/30738medtec-futuristic-650.jpg" alt="Diabetes"/> 
</p>

---

### Business Problem

#### It is desired to develop a machine learning model that can predict whether people have diabetes or not when their characteristics are specified. Before developing the model, the necessary data analysis and feature engineering steps must be performed.

### Dataset Story

The dataset is part of a large dataset held at the National Institutes of Diabetes-Digestive-Kidney Diseases in the USA. It is the data used for diabetes research on women, consisting of Pima Indian Women aged 21 and over living in Phoenix, the 5th largest city of the State of Arizona in the USA. The target variable is specified as "Outcome"; 1 indicates positive diabetes test result, 0 indicates negative.

### Dataset

 Sr. | Feature  | Description |
--- | --- | --- | 
1 |Pregnancies| Number of Pregnancy                                |
2 |Glucose| 	2-hour plasma glucose concentration in the oral glucose tolerance test    |    
3 |Blood Pressure	| 	Blood Pressure (Small Blood Pressure) (mmHg) |
4 |SkinThickness	| 	Skin Thickness    |    
5 |Insulin| 	2-hour serum insulin (mu U/ml)    |    
6 |DiabetsPedigreeFunction| A function that calculates the probability of having diabetes according to one's descendants   |    
7 |BMI|	Body mass index    |    
8 |Age| Age    |    
9 |Outcome| 	1 positive indicates does have diabetes, 0 indicates negative does not have diabetes.    |    

### Models

LR

roc_auc score : 0.9461310541310542
f1 score : 0.818130760218734
precision score : 0.8629921744921745
recall score : 0.7796296296296296
accuracy score : 0.878896103896104

KNN

roc_auc score : 0.7741965811965812
f1 score : 0.5866698689811842
precision score : 0.6407626994583516
recall score : 0.5447293447293446
accuracy score : 0.7343643198906358

RF

roc_auc score : 0.8272678062678063
f1 score : 0.6328878646802402
precision score : 0.7046265828124898
recall score : 0.5967236467236466
accuracy score : 0.7577580314422421

GBM

roc_auc score : 0.8377122507122507
f1 score : 0.6411116662010293
precision score : 0.7032753922461569
recall score : 0.6118233618233618
accuracy score : 0.7643028024606972

XGBoost

roc_auc score : 0.8257834757834758
f1 score : 0.615988883679632
precision score : 0.6462319023569023
recall score : 0.5968660968660969
accuracy score : 0.7421736158578265

CatBoost

roc_auc score : 0.8401538461538461
f1 score : 0.6364472431568802
precision score : 0.7007725285986155
recall score : 0.5896011396011397
accuracy score : 0.7656185919343814

LightGBM

roc_auc score : 0.8341282051282051
f1 score : 0.6293806543626935
precision score : 0.6619558755631318
recall score : 0.6044159544159544
accuracy score : 0.7525632262474369
