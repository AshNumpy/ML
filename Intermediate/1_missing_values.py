########################################
### Missing Values - Intermediate ML ###
########################################

import pandas as pd
from sklearn.model_selection import train_test_split

### SETUP ###
train_path = "C:/Users/ramaz/Documents/ml_datas/home_data/train.csv"
test_path = "C:/Users/ramaz/Documents/ml_datas/home_data/test.csv"
X_full = pd.read_csv(train_path, index_col='Id')
X_full_test = pd.read_csv(test_path, index_col='Id')

# Remove rows with missing target, seperate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice 
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_full_test.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
print("*"*50)

### STEP-1 PRELIMINARY INVESTIGATION ###
print("Train datas shape: ", X_train.shape) # (rows, columns)

# number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print("Missing values: \n", missing_val_count_by_column[missing_val_count_by_column>0])

num_rows = X_train.shape[0]
tot_missing = missing_val_count_by_column[missing_val_count_by_column>0].sum()

print("Number of rows: ", num_rows)
print("Total missing value: ", tot_missing)

# Function for comparing different approaches
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

print("\nMAE Approches:")
print("*"*50)

### STEP-2 TESTING 'DROP COLUMNS WITH MISSING VALUES' APPROACH ###
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
print("-"*50)



### STEP-3 TESTING 'IMPUTATION' APPROACH ###
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

# impute to lines
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
print("-"*50)


### STEP-4 GENERATE TEST PREDICTIONS ###
final_imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))

# define the fit model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

# get validation predictions and MAE
preds_valid = model.predict(final_X_valid)
print("MAE Choosen: (Drop columns with missing values)")
print(mean_absolute_error(y_valid, preds_valid))

# preprocess test data
final_X_test = pd.DataFrame(final_imputer.transform(X_test))

# get test predictions
preds_test = model.predict(final_X_test)

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('C:/Users/ramaz/Documents/ml_datas/home_data/submission.csv', index=False)