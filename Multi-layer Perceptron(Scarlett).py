#%%#
### Multi-layer Perceptron ###
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


product = pd.read_csv('/Users/ttonny0326/GitHub_Project/BA_ORRA_python_Grocery_prediction/Products_Information.csv')
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
for (store, product_type), group in filtered.groupby(['store_nbr', 'product_type'], observed=True):
    segmented_data[(store, product_type)] = group[['sales', 'special_offer', 'id']]


store1beauty = segmented_data[(1, 'BEAUTY')]

# Assuming X_train, y_train, X_test, y_test are your prepared datasets
train_size = int(len(store1beauty) * 0.8)
val_size = len(store1beauty) - train_size

X = store1beauty.drop(columns=['sales'])  
y = store1beauty['sales']

X_train, X_val = X.iloc[:train_size], X.iloc[val_size:]
y_train, y_val = y.iloc[:train_size], y.iloc[val_size:]



# Initialize the MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', max_iter=1000)

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

store1beauty_ans = segmented_data_ans[(1, 'BEAUTY')]

X_future = store1beauty_ans.drop(columns=['sales']) 
X_future.index = future_dates 
# Predict using the trained model
future_predictions = mlp.predict(X_future)

future_predictions[0:15]


# %%
