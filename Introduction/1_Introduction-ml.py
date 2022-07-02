#################################
### Intro to Machine Learning ###
#################################

import pandas as pd

### 1st BASIC DATA EXP. ###
home_data_path = "C:/Users/ramaz/Documents/ml_datas/home_data/home_data.csv"
home_data = pd.read_csv(home_data_path)

print(home_data.describe())
print("-"*50)

### 2nd FIRST ML ###
print(home_data.columns)
y = home_data.SalePrice #Reel prices
feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
x = home_data[feature_names] 

print(x.head())
print(x.describe())

from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(x,y)

print('\nPredictions: ')
predicted_home_prices = iowa_model.predict(x)
print(predicted_home_prices)
print("-"*50)

### 3rd MODEL VALIDATION ###
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(predicted_home_prices, y) # diff. of real and predicted prices
print('\nMAE: ',mae)

from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1) #split data as train and validation
iowa_model.fit(train_x,train_y)
val_predictions = iowa_model.predict(val_x)

print('\nValidation Predicts: ',val_predictions[:4])
print('Real Prices: \n', train_y.head(4))

val_MAE = mean_absolute_error(val_predictions, val_y)
print("Val MAE: ",val_MAE)
print("-"*50)


### 4th UNDERFITTING & OVERFITTING ###
def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes ,random_state=0)
    model.fit(train_x, train_y)
    preds_vals = model.predict(val_x)
    mae = mean_absolute_error(val_y, preds_vals)
    return mae

border_mae = val_MAE
for i in range(10,1000,1):
    my_mae = get_mae(i, train_x, val_x, train_y, val_y)
    my_mae = round(my_mae,1)
    if(my_mae < border_mae):
        border_mae = my_mae
        best_tree_size = i
        print(f"Max Leaf Nodes: {i} \t\t Mean Absolute Error: {my_mae}")

print("-"*50)

### 5th RANDOM FOREST ###
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_x, train_y)
rf_val_mae = mean_absolute_error(rf_model.predict(val_x), val_y)
print("\nValidation MAE for Random Forest Model: ",rf_val_mae)

# https://www.kaggle.com/learn/intro-to-machine-learning