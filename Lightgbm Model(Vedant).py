
#### alternative to Randomforest can be lightgbm or xgboost  models which may compute the data more efficiently w.r.t time.
###  below is the code snippet for Lightgbm. Please check the code and run on your compiler.

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class NonStationarySalesForecaster:
    # Loading the data form the file
    product = pd.read_csv("/Users/ttonny0326/BA_ORRA/Python_Programming/Products_Information.csv")
    # Convert the 'date' column to datetime format and set it as the index
    product['date'] = pd.to_datetime(product['date'])
    product.set_index('date', inplace=True)
    # Ensure 'product_type' is of categorical data type
    product['product_type'] = product['product_type'].astype('category')
    # Build a dictionary for the store-product grouping 
    segmented_data = {}
    # Grouping the data by store and product, and storing each group in the dictionary
    for (store, product_type), group in product.groupby(['store_nbr', 'product_type'], observed=True):
            segmented_data[(store, product_type)] = group[['sales', 'special_offer', 'id', 'store_nbr']]

    def __init__(self, store_number, product_type, train_end_date='2016-12-31', validation_end_date='2017-07-31', n_estimators=250, max_depth=15, random_state=42, max_features= 'sqrt', min_samples_leaf=2, min_samples_split=5):
        self.store_number = store_number
        self.product_type = product_type
        self.train_end_date = train_end_date
        self.validation_end_date = validation_end_date
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_features = max_features
        self.model = None
    # ... (your existing class code)

    def lightgbm_fit(self):
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        X_train = specific_segment[:self.validation_end_date][['special_offer', 'id', 'store_nbr']].values
        Y_train = specific_segment[:self.validation_end_date]['sales']

        # Initialize the LightGBM model with specified parameters
        self.model = lgb.LGBMRegressor(n_estimators=self.n_estimators,
                                       max_depth=self.max_depth,
                                       random_state=self.random_state,
                                       learning_rate=0.1
                                       )

        # Train the model on the entire training data
        self.model.fit(X_train, Y_train)

    def lightgbm_predict(self):
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        X_test = specific_segment[self.validation_end_date:][['special_offer', 'id', 'store_nbr']].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        y_predict = self.model.predict(X_test)

        mse = mean_squared_error(Y_test, y_predict)

        # Plot the forecast alongside the actual test data
        plt.figure(figsize=(12, 6))
        plt.plot(Y_test.index, y_predict, color='blue', label='Predicted Sales')
        plt.plot(Y_test.index, Y_test, color='red', label='Actual Sales')
        plt.title('Sales Forecast vs Actuals')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        # Print the MSE
        print("Mean Squared Error:", mse)
