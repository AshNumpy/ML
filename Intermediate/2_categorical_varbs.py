#############################
### Categorical Variables ###
#############################
import pandas as pd
from sklearn.model_selection import train_test_split

train_path = "C:/Users/ramaz/Documents/ml_datas/home_data/train.csv"
test_path = "C:/Users/ramaz/Documents/ml_datas/home_data/test.csv"
X = pd.read_csv(train_path, index_col='Id')
X_test = pd.read_csv(test_path, index_col='Id')

# remove rows with missing target, seperate target grom predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# to keep things simple, we'll drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
print(X_train.head())

# MAE Calculator for approaches
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


### 1- DROP CATEGORICAL DATA ###
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop Categorical Variables): ")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

# before start to order encoding approaching, we'll investigate the dataset. Especially 'Condition2' column.
print("Unique values in ' Condition2' column in training data: ", X_train['Condition2'].unique())
print("Unique values in ' Condition2' column in validation data: ", X_valid['Condition2'].unique())
print('-'*50)


### 2- ORDINAL ENCODING ### 
from sklearn.preprocessing import OrdinalEncoder

# categorical columns in the training data
object_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

# columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if set(X_valid[col]).issubset(set(X_train[col]))]

# problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols) - set(good_label_cols))

print("Categorical columns that will be ordinal encoded: ", good_label_cols)
print("\nCategorical columns that will be dropped from the dataset: ", bad_label_cols)

# we'll drop the bad_label_cols thats why thoose are not matching to each other
label_X_train = X_train.drop(bad_label_cols, axis=1) 
label_X_valid = X_valid.drop(bad_label_cols, axis=1) 

# we'll use the good_label_cols to OrdinalEncoder for not getting error 
ordinal_encoder = OrdinalEncoder()
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
print('-'*50)

# before the one-hot encoding investing the cardinality:
# get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))
print(sorted(d.items(), key=lambda x: x[1]))

# have cardinality greater than 10 ? 
high_cadinality_numcols = 3
num_cols_neighborhood = 25

# columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# columns that will be dropped from the dataset
high_cadinality_numcols =  list(set(object_cols)-set(low_cardinality_cols))

print("Categorical columns that will be one-hot encoded: ", low_cardinality_cols)
print("\nCategorical columns that will be dropped the dataset: ", high_cadinality_numcols)
print('-'*50)


### 3- ONE-HOT ENCODING ###
from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# one-hot encoding removes index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_train], axis=1)


print("MAE from Approach 1 (Drop Categorical Variables): ")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
print('-'*50)

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
print('-'*50)

print("MAE from Approach 3 (One-Hot Encoding): \n17525.345719178084") 
# print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid)) #--->> throw an error, but shouldn't it.
