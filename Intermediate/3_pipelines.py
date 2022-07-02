#################
### Pipelines ###
#################
import pandas as pd 
from sklearn.model_selection import train_test_split

#datas
X_full = pd.read_csv("C:/Users/ramaz/Documents/ml_datas/home_data/train.csv")
X_test_full = pd.read_csv("C:/Users/ramaz/Documents/ml_datas/home_data/test.csv")
print(X_full)
print(X_test_full.head())
print('='*50)

#remove rows with missing values
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True) # seperate SalePrice column to train dataset

#set train and test data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, random_state=0)

#select low cardinality (.nunique() < 10) categorical (.dtype() == 'object') columns
categorical_cols = [cname for cname in X_train_full if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object']

#select numerical columns
numerical_cols = [cname for cname in X_train_full if X_train_full[cname].dtype in ['int64','float64']]

#keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
print(X_train.head())
###############################################

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

#preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='most_frequent')),
  ('onehot', OneHotEncoder(handle_unknown='ignore'))  
])

#bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

#define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

#bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

#preprocessing of training data, fit model
clf.fit(X_train, y_train)

#preprocessing of validation data
preds = clf.predict(X_valid)

print(f"MAE: {mean_absolute_error(y_valid, preds)}")

