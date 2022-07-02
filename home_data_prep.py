import numpy as np 
import pandas as pd 

data_path = "C:/Users/ramaz/Documents/ml_datas/home_data/home_data.csv"
data = pd.read_csv(data_path)

print(data.head())
print(data.columns)
print(len(data.columns))

# drop na
cols_with_missing = [cname for cname in data if data[cname].isnull().any()]
print(cols_with_missing)
data = data.drop(cols_with_missing, axis=1)
print(len(data.columns))

#df['DataFrame Column'] = df['DataFrame Column'].fillna(0)

data.to_excel('data_excel.xlsx')
data.to_csv('data_csv.csv')