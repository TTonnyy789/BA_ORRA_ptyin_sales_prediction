#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score


# Load the dataset
product = pd.read_csv('Products_Information.csv')


# Convert the 'date' column to datetime format and set it as the index
product['date'] = pd.to_datetime(product['date'])
product.set_index('date', inplace=True)
# Ensure 'product_type' is of categorical data type
product['product_type'] = product['product_type'].astype('category')
    # Build a dictionary for the store-product grouping 

# Separate the data from the answer
start_date = '2013-01-01'
end_date = '2017-07-30'
filtered = product[(product.index >= start_date) & (product.index <= end_date)]

segmented_data = {}
    # Grouping the data by store and product, and storing each group in the dictionary
for (store, product_type), group in product.groupby(['store_nbr', 'product_type'], observed=True):
            segmented_data[(store, product_type)] = group[['sales', 'special_offer', 'id']]

store_lingerie = segmented_data[1,'LINGERIE']

# Assuming X_train, y_train, X_test, y_test are your prepared datasets
train_size = int(len(store_lingerie) * 0.8)
val_size = len(store_lingerie) - train_size

X = store_lingerie.drop(columns=['sales'])  
y = store_lingerie['sales']

X_train, X_val = X.iloc[:train_size], X.iloc[val_size:]
y_train, y_val = y.iloc[:train_size], y.iloc[val_size:]



# Initialize the MLPRegressor
mlp = MLPRegressor(
                hidden_layer_sizes=(100, ), 
                activation='relu', 
                solver='adam', 
                max_iter=1000
                )


# Train the model
mlp.fit(X_train, y_train)


# Predict on validation set
predictions = mlp.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, predictions)
print(f"Mean Squared Error on Validation Set: {mse}")


# Defining forecasting date
future_dates = pd.date_range(start='2017-07-31', end='2017-08-15', freq='D')  # Define future dates

# Create feature data for these future dates (similar to X_train)
# This could involve extracting or generating features for these dates
# Example: Create a DataFrame with columns similar to X_train for the future dates

# Concatenate future dates and feature data into X_future DataFrame
start_date1 = '2017-07-31'
end_date1 = '2017-08-15'
ans = product[(product.index >= start_date1) & (product.index <= end_date1)]
segmented_data_ans = {}
# Grouping the data by store and product, and storing each group in the dictionary
for (store, product_type), group in ans.groupby(['store_nbr', 'product_type'], observed=True):
    segmented_data_ans[(store, product_type)] = group[['sales', 'special_offer', 'id']]

store_lingerie_ans = segmented_data_ans[(1, 'LINGERIE')]

X_future = store_lingerie_ans.drop(columns=['sales']) 
X_future.index = future_dates 
# Predict using the trained model
future_predictions = mlp.predict(X_future)

# Train the model with the initial subset of features
selected_features = ['sales', 'id']  # Update with the remaining columns


X_subset = store_lingerie[selected_features]
X_train_subset, X_val_subset = X_subset.iloc[:train_size], X_subset.iloc[train_size:]


train_size = int(len(store_lingerie) * 0.8)
val_size = len(store_lingerie) - train_size

X = store_lingerie.drop(columns=['sales'])  
y = store_lingerie['sales']

X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]


mlp.fit(X_train_subset, y_train)
predictions_subset = mlp.predict(X_val_subset)
mse_subset = mean_squared_error(y_val, predictions_subset)

print(f"MSE with initial subset of features: {mse_subset}")

r2_subset = r2_score(y_val, predictions_subset)
print(f"R-squared with initial subset of features: {r2_subset}")

#for features 'sales', 'special_offer', 'id'
#MSE with initial subset of features: 45.13634551346867
#R-squared with initial subset of features: -0.5578480385070217

#for features 'sales', 'special offer'
#MSE with initial subset of features: 0.0002741594568207731
#R-squared with initial subset of features: 0.9999905375863467

#for features 'sales', 'id'
#MSE with initial subset of features: 52.04890810390571
#R-squared with initial subset of features: -0.7964300936128341


# In[ ]:






# In[ ]:




