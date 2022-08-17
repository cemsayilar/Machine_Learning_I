import pandas as pd
import numpy as np
import math
import scipy.stats as st
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats._stats_py import ttest_ind
import matplotlib as mt
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from datetime import date
from statsmodels.stats.proportion import proportions_ztest
### Recomendation Systems
from mlxtend.frequent_patterns import apriori, association_rules
### Measurement
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
### Feature Engineering
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
### Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
### Logistic Regression
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
pd.set_option('display.max_columns',None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv('/Users/buraksayilar/Desktop/machine_learning/Maas_Tahmin_Modeli/hitters.csv')

def check_df(dataframe, head=5):
    print(f'{" Shape ":-^100}')
    print(dataframe.shape)
    print(f'{" Info":-^100}')
    print(dataframe.info(head))
    print(f'{" Head ":-^100}')
    print(dataframe.head(head))
    print(f'{" Tail ":-^100}')
    print(dataframe.tail(head))
    print(f'{" NA ":-^100}')
    print(dataframe.isnull().sum())
    print(f'{" Quantiles ":-^100}')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, num_but_car = grab_col_names(df)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
for col in num_cols:
    print(num_summary(df, col))
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
for col in cat_cols:
    print(cat_summary(df, col))
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

target_summary_with_cat(df, 'Salary', cat_cols)
## Outlier Threshold
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.90):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + interquantile_range * 1.5
    low_limit = quantile1 - interquantile_range * 1.5
    return low_limit, up_limit
def check_outlier(dataframe,col_name):
    low_limit, up_limit = outlier_thresholds(dataframe,col_name,q3=0.99)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
       return True
    else:
        return False
def check_outlier(dataframe,col_name):
    low_limit, up_limit = outlier_thresholds(dataframe,col_name,q3=0.90)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
       return True
    else:
        return False
for col in num_cols:
    print(col, check_outlier(df, col))


def grab_outliers(dataframe, col_name, index=False):
        low, up = outlier_thresholds(dataframe, col_name)
        if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
            print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].head())
        else:
            print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)])
            if index:
                outlier_index = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].index
                return outlier_index
grab_outliers(df, 'CHits')
grab_outliers(df, 'CHmRun')
df.drop([249,236], axis=0, inplace=True)
# Now we dropped our outlier value. Because we are looking for sallary, it can be missleading seriously so I select
# third outlier (3) lower (Rather than 0.99).

# Missing Value Analysis
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns
na_columns = missing_values_table(df, True)
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
missing_vs_target(df, 'Salary', na_columns)
# Missing values only exists in  target values :)
df.dropna(inplace=True)
sns.heatmap(df.corr(), cmap='RdBu')

## FEATURE ENGINEERING
# The 30 teams in MLB divided in to 2 leagues: American and National. Each league divided 3 divisions: East Central and West.
## Looking League change
df['change_league'] = np.where(df['League'] != df['NewLeague'], 1, 0)
target_summary_with_num(df,'change_league','Salary')
# Checking our potential new feature.
df[(df.change_league == 1) & (df.League == 'N')]['Salary'].mean()
df[(df.change_league == 1) & (df.League == 'A')]['Salary'].mean()
df[(df.change_league == 1) & (df.League == 'N')]['Salary']
df[(df.change_league == 1) & (df.League == 'A')]['Salary']
# New Feature *1
df['unique_players_86-87'] =np.where((df['AtBat'] > df['AtBat'].quantile(0.99)) |
                                     (df['Hits'] > df['Hits'].quantile(0.99)) |
                                     (df['HmRun'] > df['HmRun'].quantile(0.99)) |
                                     (df['Runs'] > df['Runs'].quantile(0.99)), 1, 0)
# New Feature *2
df['average_players_career'] =np.where((df['CAtBat'].between((df['CAtBat'].quantile(0.40)),(df['CAtBat'].quantile(0.60)))) |
                                       (df['CHits'].between((df['CHits'].quantile(0.40)),(df['CHits'].quantile(0.60)))) |
                                       (df['CHmRun'].between((df['CHmRun'].quantile(0.40)),(df['CHmRun'].quantile(0.60)))) |
                                       (df['CRuns'].between((df['CRuns'].quantile(0.40)),(df['CRuns'].quantile(0.60)))), 1, 0)
# New Feature *3
df['challenger_players_career'] = np.where((df['CRBI'] > df['CRBI'].quantile(0.85)) | (df['CWalks'] > df['CWalks'].quantile(0.85)), 1, 0)

### Encoding Categorical Variables
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Now, maybe I want to ignore my target value, or I already
# label encoded my 'Sex' feature. So I can select categorical variables
# for One Hot Encoding.
#ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)


# Standardization
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

# Model with Multiple Linear Regression
X = df.drop('Salary', axis=1)
y = df[['Salary']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
y_test.shape
y_train.shape

reg_model = LinearRegression().fit(X_train,y_train)
reg_model.intercept_
reg_model.coef_


## Prediction Accuracy
# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# 10 Folded CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))
cv_results = cross_validate(reg_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

# New Feature Performance

plt.figure(figsize=(10, 10))
sns.set(font_scale=1)
sns.barplot(x="Value", y="Feature", data=reg_model.coef_)
plt.title('Features')
plt.tight_layout()
plt.show()


