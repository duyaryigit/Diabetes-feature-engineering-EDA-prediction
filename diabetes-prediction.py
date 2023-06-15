import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import warnings
import missingno as msno

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

############## DATASET HISTORY ##############################################

# Pregnancies: Number of times pregnant
# Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test
# BloodPressure: Diastolic blood pressure (mm Hg)
# SkinThickness: Triceps skin fold thickness (mm)
# Insulin: 2-Hour serum insulin (mu U/ml) hormon
# BMI: Body mass index (weight in kg/(height in m)2)
# DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)
# Age: Age (years)
# Outcome: Class variable (0 if non-diabetic, 1 if diabetic) "Dependent Variable"

##########################################################################

# TASK 1: EXPLORATORY DATA ANALYSIS
           # Step 1: Examine the overall picture.
           # Step 2: Capture the numeric and categorical variables.
           # Step 3: Analyze the numerical and categorical variables.
           # Step 4: Perform target variable analysis. (The mean of the target variable according to the categorical variables, the mean of the numeric variables according to the target variable)
           # Step 5: Analyze outliers.
           # Step 6: Perform a missing observation analysis.
           # Step 7: Perform correlation analysis.

# TASK 2: FEATURE ENGINEERING
           # Step 1: Take necessary actions for missing and outlier values. There are no missing observations in the data set, but Glucose, Insulin etc.
           # observation units containing 0 in variables may represent missing values. For example; a person's glucose or insulin value
           # 0 will not be possible. Considering this situation, we assign the zero values to the relevant values as NaN and then set the missing values.
           # you can apply operations.

           # Step 2: Create new variables.
           # Step 3: Perform the encoding operations.
           # Step 4: Standardize for numeric variables.
           # Step 5: Create the model.

##################################
# TASK 1: EXPLORATORY DATA ANALYSIS
##################################

def load():
    data = pd.read_csv("datasets/diabetes.csv")
    return data

df = load()
df_ = df.copy()

df["Outcome"].value_counts()

##################################
# CAPTURE OF NUMERICAL AND CATEGORICAL VARIABLES
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optional
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical view cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.

    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car


##################################
# ANALYSIS OF CATEGORY VARIABLES
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, "Outcome",plot=True)


##################################
# ANALYSIS OF NUMERICAL VARIABLES
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

##################################
# ANALYSIS OF NUMERICAL VARIABLES ACCORDING TO TARGET
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


##################################
# CORRELATION ANALYSIS
##################################

df.corr()

# Correlation Matrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

df.corrwith(df["Outcome"]).sort_values(ascending=False)


diabetic = df[df.Outcome == 1]
healthy = df[df.Outcome == 0]

plt.scatter(healthy.Age, healthy.Insulin, color="green", label="Healthy", alpha = 0.4)
plt.scatter(diabetic.Age, diabetic.Insulin, color="red", label="Diabetic", alpha = 0.4)
plt.xlabel("Age")
plt.ylabel("Insulin")
plt.legend()
plt.show()

##################################
# BASE MODEL
##################################

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   #("SVC", SVC()),
                   #("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   #('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('CatBoost', CatBoostClassifier(verbose=False)),
              ("LightGBM", LGBMClassifier())]

from sklearn.model_selection import cross_val_score

for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))

'''
LR

roc_auc score:0.829039886039886
f1 score:0.6239997074390266
precision score:0.7284056005652115
recall score:0.5519943019943019
accuracy score:0.7695317840054683

KNN

roc_auc score:0.748448717948718
f1 score:0.5655214062832494
precision score:0.618033776471655
recall score:0.5266381766381767
accuracy score:0.7213773069036227

RF

roc_auc score:0.8248361823361823
f1 score:0.6216715594496588
precision score:0.7316847657808755
recall score:0.5782051282051281
accuracy score:0.7577922077922079

GBM

roc_auc score:0.8305356125356125
f1 score:0.6327000636380691
precision score:0.6892628658198373
recall score:0.6118233618233618
accuracy score:0.7604066985645933

XGBoost

roc_auc score:0.7955299145299145
f1 score:0.6060284638945773
precision score:0.6358062368723032
recall score:0.5896011396011397
accuracy score:0.7357142857142857

CatBoost

roc_auc score:0.8334957264957265
f1 score:0.6353695650384438
precision score:0.690292680121902
recall score:0.6005698005698006
accuracy score:0.762987012987013

LightGBM

roc_auc score:0.8054216524216524
f1 score:0.6075187280502355
precision score:0.6502262536314259
recall score:0.5817663817663817
accuracy score:0.7409432672590568
'''

##################################
# TASK 2: FEATURE ENGINEERING
##################################

##################################
# MISSING VALUE ANALYSIS
##################################

# It is known that variable values other than Pregnancies and Outcome cannot be 0 in a human.
# Therefore, an action decision should be taken regarding these values. Values that are 0 can be assigned NaN.

zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

zero_columns

# 2nd Solution:

for i in df.columns:
    print('{} zero values: {}'.format(i, (df[i] == 0).sum()))

# We went to each of the variables with 0 in the observation units and changed the observation values containing 0 with NaN.

for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

# Missing Observation Analysis

df.isnull().sum()

# Examining the missing data structure

msno.matrix(df)
plt.show()

msno.bar(df)
plt.show()

msno.heatmap(df)
plt.show()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 3)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns, missing_df

na_columns, missing_df = missing_values_table(df, na_name=True)


# Examining the Relationship of Missing Values with the Dependent Variable

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_columns)

# Filling in Missing Values

for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

df.isnull().sum()

# 2nd alternative: filling with Median

def replace_na_to_median(dataframe, na_col):
    for j in na_col:
        if (dataframe[j] == 0).any() == True:
            dataframe[j] = dataframe[j].replace(to_replace=0, value=dataframe[j].median())
    print(dataframe.head())


replace_na_to_median(df, ["Insulin", "SkinThickness", "Glucose", "BloodPressure", "BMI"])

# 3rd alternative: with missing value (value 0) considering other variables as independent
# For example, you can create a reg model for Insulin and make predictions based on it.

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

y = df["Insulin"]
X = df.drop("Insulin", axis=1)
reg_model = LinearRegression().fit(X, y)
y_pred = reg_model.predict(X)

mean_absolute_error(y, y_pred)

y_pred = pd.DataFrame(y_pred).astype(int)
df["y_pred"] = y_pred
df.loc[(df["Insulin"] == 0), "Insulin"]= df["y_pred"]
df["Insulin"]

def replace_na_to_reg(dataframe, na_col):
    for j in na_col:
        y_pred = 0
        if (dataframe[j] == 0).any() == True:
            y = dataframe[j]
            X = dataframe.drop(j, axis=1)
            reg_model = LinearRegression().fit(X, y)
            y_pred = reg_model.predict(X)
            dataframe["y_pred"] = pd.DataFrame(y_pred)
            dataframe.loc[(dataframe[j] == 0), j] = abs(dataframe["y_pred"])
            print(dataframe.head())


replace_na_to_reg(df, ["Insulin", "SkinThickness", "Glucose", "BloodPressure", "BMI"])


##################################
# OUTLIER ANALYSIS
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# LOCAL OUTLIER FACTOR

clf=LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df_)
df_scores=clf.negative_outlier_factor_
df_scores[0:5]
np.sort(df_scores)[0:8]

scores=pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True,xlim=[0,20],style=".-")
plt.show()

esik_deger=np.sort(df_scores)[5]

df[df_scores<esik_deger].shape

df[df_scores<esik_deger].index

df[df_scores<esik_deger].drop(axis=0,labels=df[df_scores<esik_deger].index)

num_cols=[col for col in df.columns if df[col].dtypes!="O"]
outlier_thresholds(df,num_cols)
check_outlier(df,num_cols)

for i in df.index:
    for esik in df[df_scores < esik_deger].index:
        if i==esik:
            for col in num_cols:
                print(i,col,replace_with_thresholds(df,col))


# df["Insulin"] = df["Insulin"].fillna(df.groupby("NEW_GLUCOSE_CAT")["Insulin"].transform("median"))

# FEATURE ENGINEERING

df["NEW_STHICKNESS_BMI"] = df["SkinThickness"] / df["BMI"]
df["NEW_AGE_DPEDIGREE"] = df["Age"] / df["DiabetesPedigreeFunction"]
df["NEW_GLUCOSE_BPRESSURE"] = (df["BloodPressure"] * df["Glucose"])/100

df.loc[(df['BMI'] < 18.5), 'NEW_BMI_CAT'] = "underweight"
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] <= 24.9), 'NEW_BMI_CAT'] = 'normal'
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30), 'NEW_BMI_CAT'] = 'overweight'
df.loc[(df['BMI'] >= 30), 'NEW_BMI_CAT'] = 'obese'

df.loc[(df['Age'] < 21), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 21) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['Pregnancies'] == 0), 'NEW_PREGNANCY_CAT'] = 'no_pregnancy'
df.loc[(df['Pregnancies'] == 1), 'NEW_PREGNANCY_CAT'] = 'one_pregnancy'
df.loc[(df['Pregnancies'] > 1), 'NEW_PREGNANCY_CAT'] = 'multi_pregnancy'

df.loc[(df['Glucose'] >= 170), 'NEW_GLUCOSE_CAT'] = 'dangerous'
df.loc[(df['Glucose'] >= 105) & (df['Glucose'] < 170), 'NEW_GLUCOSE_CAT'] = 'risky'
df.loc[(df['Glucose'] < 105) & (df['Glucose'] > 70), 'NEW_GLUCOSE_CAT'] = 'normal'
df.loc[(df['Glucose'] <= 70), 'NEW_GLUCOSE_CAT'] = 'low'

df.loc[(df['BloodPressure'] >= 110), 'NEW_BLOODPRESSURE_CAT'] = 'hypersensitive crisis'
df.loc[(df['BloodPressure'] >= 90) & (
        df['BloodPressure'] < 110), 'NEW_BLOODPRESSURE_CAT'] = 'hypertension'
df.loc[(df['BloodPressure'] < 90) & (df['BloodPressure'] > 70), 'NEW_BLOODPRESSURE_CAT'] = 'normal'
df.loc[(df['BloodPressure'] <= 70), 'NEW_BLOODPRESSURE_CAT'] = 'low'

df.loc[(df['Insulin'] >= 160), 'NEW_INSULIN_CAT'] = 'high'
df.loc[(df['Insulin'] < 160) & (df['Insulin'] >= 16), 'NEW_INSULIN_CAT'] = 'normal'
df.loc[(df['Insulin'] < 16), 'NEW_INSULIN_CAT'] = 'low'

# Capitalize columns:

df.columns = [col.upper() for col in df.columns]

##################################
# ENCODING
##################################

# The process of separating variables according to their datatypes

cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.dtypes

# LABEL ENCODING

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


#label_encoder.inverse_transform([0,1])

for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding
# Update process of the cat_cols list

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
##################################
# STANDARDIZATION
##################################

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

### MODELLING ###

y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   #("SVC", SVC()),
                   #("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   #('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('CatBoost', CatBoostClassifier(verbose=False)),
              ("LightGBM", LGBMClassifier())]

from sklearn.model_selection import cross_val_score

for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))

'''
LR

roc_auc score:0.9461310541310542
f1 score:0.818130760218734
precision score:0.8629921744921745
recall score:0.7796296296296296
accuracy score:0.878896103896104

KNN

roc_auc score:0.7741965811965812
f1 score:0.5866698689811842
precision score:0.6407626994583516
recall score:0.5447293447293446
accuracy score:0.7343643198906358

RF

roc_auc score:0.8272678062678063
f1 score:0.6328878646802402
precision score:0.7046265828124898
recall score:0.5967236467236466
accuracy score:0.7577580314422421

GBM

roc_auc score:0.8377122507122507
f1 score:0.6411116662010293
precision score:0.7032753922461569
recall score:0.6118233618233618
accuracy score:0.7643028024606972

XGBoost

roc_auc score:0.8257834757834758
f1 score:0.615988883679632
precision score:0.6462319023569023
recall score:0.5968660968660969
accuracy score:0.7421736158578265

CatBoost

roc_auc score:0.8401538461538461
f1 score:0.6364472431568802
precision score:0.7007725285986155
recall score:0.5896011396011397
accuracy score:0.7656185919343814

LightGBM

roc_auc score:0.8341282051282051
f1 score:0.6293806543626935
precision score:0.6619558755631318
recall score:0.6044159544159544
accuracy score:0.7525632262474369
'''

def plot_importance(model, X, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    plt.figure(figsize=(10, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[1:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    # plt.savefig('importances-01.png')
    plt.show()


plot_importance(rf_model, X)
