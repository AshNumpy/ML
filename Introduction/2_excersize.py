#####################################
### Intermadiate Machine Learning ###
#####################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

### SETTINGS ###
X_full = pd.read_csv('C:/Users/ramaz/Documents/ml_datas/home_data.csv', index_col='Id')
feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
X = X_full[feature_names]
y = X_full.SalePrice

### SPLIT DATA ###
train_x, val_x, train_y, val_y = train_test_split(X, y, random_state=1)

### RANDOM FOREST MODELS ###
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

### BEST MODEL ###
def score_model(model, X_t=train_x, X_v = val_x, y_t=train_y, y_v=val_y):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

print('Models MAE: ')
for i in range(0, len(models)):
    print(f"Model {i+1} \t\t Models MAE: {score_model(models[i])}")

print(f'Best RF Model: {model_2} \t\t Models MAE: 22118.6')