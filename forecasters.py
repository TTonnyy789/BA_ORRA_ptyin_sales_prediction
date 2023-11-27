#%%#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, r2_score


class ZeroSalesForecaster:
    def predict(self, periods):
        return np.zeros(periods)



class StationarySalesForecaster:
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

    def __init__(self, store_number, product_type, order=(8, 2, 6), train_end_date='2016-12-31', validation_end_date='2017-07-31'):  
        # self.segmented_data = segmented_data
        self.store_number = store_number
        self.product_type = product_type
        self.order = order
        self.train_end_date = train_end_date
        self.validation_end_date = validation_end_date
        # self.n_estimators = n_estimators
        # self.max_depth = max_depth
        self.model = None

    def arima_fit(self):
        # Retrieve the specific segment of the data
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Ensuring the date index is in datetime format
        specific_segment.index = pd.to_datetime(specific_segment.index)
        # Splitting the data for training
        train_data = specific_segment[:self.train_end_date]['sales']
        validation_data = specific_segment[self.train_end_date:self.validation_end_date]['sales']
        # Splitting the data for testing 
        test_data = specific_segment[self.validation_end_date:]['sales']

        # Fit the ARIMA model
        self.model = ARIMA(train_data, order=self.order)
        self.model = self.model.fit()

    def arima_predict(self):
        # Retrieve the specific segment of the data
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]
        
        # Setting up for the testing data
        test_data = specific_segment[self.validation_end_date:]['sales']
        
        # Forecasting sales for the specified number of steps
        forecast = self.model.forecast(steps=len(test_data))

        # Plot the forecast alongside the actual test data
        plt.figure(figsize=(12,6))
        plt.plot(test_data.index, forecast, color='blue', label='Predicted Sales')
        plt.plot(test_data.index, test_data, color='red', label='Actual Sales')
        plt.title('Sales Forecast vs Actuals')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        # Evaluate the model's performance
        mse = mean_squared_error(test_data, forecast)
        print("ARIMA Mean Squared Error:", mse)

    def create_lag_features(self, lags=12):
        # using the self.segmented_data to create the lag features
        # using the previous lags to predict the future sales

        specific_segment = self.segmented_data[(self.store_number, self.product_type)].copy()
        for lag in range(1, lags + 1):
            specific_segment[f'lag_{lag}'] = specific_segment['sales'].shift(lag)
        self.segmented_data[(self.store_number, self.product_type)] = specific_segment
    
    def linear_regression_predict(self, lags=21):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import matplotlib.pyplot as plt
        import numpy as np

        # Create lag features
        self.create_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Drop rows with NaN values to align X and Y
        specific_segment.dropna(subset=[f'lag_{i}' for i in range(1, lags + 1)], inplace=True)

        # Prepare the features and target variable
        lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        X = specific_segment[lag_columns]
        Y = specific_segment['sales']

        # Split the data into training and testing sets
        X_train = X[:self.validation_end_date].values
        Y_train = Y[:self.validation_end_date]
        X_test = X[self.validation_end_date:].values
        Y_test = Y[self.validation_end_date:]

        # Initialize and train the Linear Regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train, Y_train)

        # Make predictions
        y_predict = lr_model.predict(X_test)

        # Plotting the forecast alongside the actual test data
        plt.figure(figsize=(22, 6))
        plt.plot(Y_test.index, y_predict, color='blue', label='Predicted Sales')
        plt.plot(Y_test.index, Y_test, color='red', label='Actual Sales')
        plt.title(f'Linear Regression Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        # Evaluating the model's performance
        print("Linear Regression MSE:", mean_squared_error(Y_test, y_predict))
        print("Linear Regression R2 Score:", r2_score(Y_test, y_predict))

        # Day-by-day evaluation using MAE
        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)
    
    def optimize_linear_regression(self):
        from sklearn.linear_model import LinearRegression
        import pandas as pd
        from sklearn.model_selection import GridSearchCV, PredefinedSplit

        # Best parameters initialization
        best_mse = float('inf')
        best_lag = 0
        best_params = None

        # Define specific lag values to test
        lag_values = [14] 
        # You can use multiple lag values as well, like [14, 21, 28, 35, 42, 49]

        for lag in lag_values:
            self.create_lag_features(lags=lag)
            specific_segment = self.segmented_data[(self.store_number, self.product_type)].dropna()
            
            # Splitting the data
            train_data = specific_segment[:self.train_end_date]
            validation_data = specific_segment[self.train_end_date:]
            feature_cols = [col for col in train_data.columns if col != 'sales']
            X_train = train_data[feature_cols]
            y_train = train_data['sales']
            X_validation = validation_data[feature_cols]
            y_validation = validation_data['sales']

            # Combine train and validation sets
            X_combined = pd.concat([X_train, X_validation])
            y_combined = pd.concat([y_train, y_validation])

            # Define the split index
            split_index = [-1]*len(X_train) + [0]*len(X_validation)
            pds = PredefinedSplit(test_fold=split_index)

            # Parameter grid for Linear Regression
            param_grid = {
                'fit_intercept': [True, False]
                }

            # Grid Search with predefined split
            lr = LinearRegression()
            grid_search = GridSearchCV(
                estimator=lr, 
                param_grid=param_grid, 
                cv=pds, 
                scoring='neg_mean_squared_error', 
                n_jobs=-1
            )
            
            grid_search.fit(X_combined, y_combined)

            # Evaluate the best model
            best_model_mse = -grid_search.best_score_
            if best_model_mse < best_mse:
                best_mse = best_model_mse
                best_lag = lag
                best_params = grid_search.best_params_

        # Output the best settings
        print(f"Best MSE: {best_mse}")
        print(f"Best Lag: {best_lag}")
        print(f"Best Parameters: {best_params}")

    def randomforest_date_predict(self, lags=14):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_absolute_error

        self.create_lag_features(lags=lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]
        specific_segment.dropna(inplace=True)  # Drop rows with NaNs

        # Prepare the features and target variable
        lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        X_train = specific_segment[:self.validation_end_date][lag_columns].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        
        X_test = specific_segment[self.validation_end_date:][lag_columns].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Train Random Forest model
        rf_model = RandomForestRegressor(
                                        n_estimators=300, 
                                        max_depth=20, 
                                        random_state=34,
                                        min_samples_leaf=4,
                                        min_samples_split=10,
                                        n_jobs=-1
                                        )
        
        rf_model.fit(X_train, Y_train)
        rf_predictions = rf_model.predict(X_test)

        # Plotting
        plt.figure(figsize=(22,6))
        plt.plot(Y_test.index, rf_predictions, color='green', label='RF Predicted Sales')
        plt.plot(Y_test.index, Y_test, color='red', label='Actual Sales')
        plt.title(f'Random Forest Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        # Model evaluation
        r2_rf = r2_score(Y_test, rf_predictions)
        print("Random Forest r2_score:", r2_rf)

        # Day-by-day evaluation using MAE
        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, rf_predictions)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)
    
    def optimize_date_randomforest(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV, PredefinedSplit

        # Best parameters initialization
        best_mse = float('inf')
        best_lag = 0
        best_params = None

        # Define specific lag values to test
        # lag_values = [14]    
        lag_values = [14, 21, 28, 35, 42, 49]

        for lag in lag_values:
            self.create_lag_features(lags=lag)
            specific_segment = self.segmented_data[(self.store_number, self.product_type)].dropna()
            
            # Splitting the data
            train_data = specific_segment[:self.train_end_date]
            validation_data = specific_segment[self.train_end_date:]
            feature_cols = [col for col in train_data.columns if col != 'sales']
            X_train = train_data[feature_cols]
            y_train = train_data['sales']
            X_validation = validation_data[feature_cols]
            y_validation = validation_data['sales']

            # Combine train and validation sets
            X_combined = pd.concat([X_train, X_validation])
            y_combined = pd.concat([y_train, y_validation])

            # Define the split index
            split_index = [-1]*len(X_train) + [0]*len(X_validation)
            pds = PredefinedSplit(test_fold=split_index)

            # Parameter grid for LightGBM
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'n_jobs': [-1]
                }
            # Grid Search with cross-validation
            rf_model = RandomForestRegressor(
                                    random_state=34
                                    )
            
            grid_search = GridSearchCV(
                                        estimator=rf_model, param_grid=param_grid, 
                                        cv=pds, 
                                        scoring='neg_mean_squared_error', n_jobs=-1
                                        )
            
            grid_search.fit(X_combined, y_combined)

            # Evaluate the best model
            best_model_mse = -grid_search.best_score_
            if best_model_mse < best_mse:
                best_mse = best_model_mse
                best_lag = lag
                best_params = grid_search.best_params_

        # Output the best settings
        print(f"Best MSE: {best_mse}")
        print(f"Best Lag: {best_lag}")
        print(f"Best Parameters: {best_params}")
    
    def lightgbm_date_predict(self, lags=14):
        import lightgbm as lgb
        from sklearn.metrics import r2_score, mean_absolute_error
        # Create lag features
        self.create_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        X_train = specific_segment[:self.validation_end_date][lag_columns].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        
        X_test = specific_segment[self.validation_end_date:][lag_columns].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Initialize and train the LightGBM model
        lgb_model = lgb.LGBMRegressor(
                                    n_estimators=100, 
                                    max_depth=20,
                                    random_state=42,
                                    learning_rate=0.05,
                                    min_samples_leaf=4,
                                    min_samples_split=2,
                                    n_jobs=-1
                                    )
        lgb_model.fit(X_train, Y_train)

        # Make predictions
        y_predict = lgb_model.predict(X_test)

        # Plotting the forecast alongside the actual test data
        plt.figure(figsize=(22, 6))
        plt.plot(Y_test.index, y_predict, color='blue', label='Predicted Sales')
        plt.plot(Y_test.index, Y_test, color='red', label='Actual Sales')
        plt.title(f'LightGBM-PureDate Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        # Evaluating the model's performance   
        r2_date_lgb = r2_score(Y_test, y_predict)
        print("LightGBM r2_score:", r2_date_lgb)

        # Day-by-day evaluation using MAE
        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)

    def lightgbm_EXfeature_predict(self):
        import lightgbm as lgb
        from sklearn.metrics import r2_score, mean_absolute_error
        # Create lag features
        # self.create_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        # lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        X_train = specific_segment[:self.validation_end_date][['special_offer', 'id', 'store_nbr']].values
        Y_train = specific_segment[:self.validation_end_date]['sales']

        X_test = specific_segment[self.validation_end_date:][['special_offer', 'id', 'store_nbr']].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Initialize and train the LightGBM model
        lgb_model = lgb.LGBMRegressor(n_estimators=100, 
                                       max_depth=10,
                                       random_state=42,
                                       learning_rate=0.1)
        lgb_model.fit(X_train, Y_train)

        # Make predictions
        y_predict = lgb_model.predict(X_test)

        # Plotting the forecast alongside the actual test data
        plt.figure(figsize=(12, 6))
        plt.plot(Y_test.index, y_predict, color='blue', label='Predicted Sales')
        plt.plot(Y_test.index, Y_test, color='red', label='Actual Sales')
        plt.title(f'LightGBM-EXfeatures Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        # Evaluating the model's performance
        r2_EXfeature_lgb = r2_score(Y_test, y_predict)
        print("LightGBM r2_score:", r2_EXfeature_lgb)

        # Day-by-day evaluation using MAE
        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)
    
    def optimize_date_lightgbm(self):
        import lightgbm as lgb
        from sklearn.model_selection import GridSearchCV, PredefinedSplit

        # Best parameters initialization
        best_mse = float('inf')
        best_lag = 0
        best_params = None

        # Define specific lag values to test
        # lag_values = [14]    
        lag_values = [14, 21, 28, 35, 42, 49]

        for lag in lag_values:
            self.create_lag_features(lags=lag)
            specific_segment = self.segmented_data[(self.store_number, self.product_type)].dropna()
            
            # Splitting the data
            train_data = specific_segment[:self.train_end_date]
            validation_data = specific_segment[self.train_end_date:]
            feature_cols = [col for col in train_data.columns if col != 'sales']
            X_train = train_data[feature_cols]
            y_train = train_data['sales']
            X_validation = validation_data[feature_cols]
            y_validation = validation_data['sales']

            # Combine train and validation sets
            X_combined = pd.concat([X_train, X_validation])
            y_combined = pd.concat([y_train, y_validation])

            # Define the split index
            split_index = [-1]*len(X_train) + [0]*len(X_validation)
            pds = PredefinedSplit(test_fold=split_index)

            # Parameter grid for LightGBM
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_jobs': [-1]
            }

            # Grid Search with predefined split
            lgb_model = lgb.LGBMRegressor(
                                        random_state=42, 
                                        force_col_wise=True
                                        )
            
            grid_search = GridSearchCV(
                                        estimator=lgb_model, param_grid=param_grid, 
                                        cv=pds, 
                                        scoring='neg_mean_squared_error', n_jobs=-1
                                        )
            grid_search.fit(X_combined, y_combined)

            # Evaluate the best model
            best_model_mse = -grid_search.best_score_
            if best_model_mse < best_mse:
                best_mse = best_model_mse
                best_lag = lag
                best_params = grid_search.best_params_

        # Output the best settings
        print(f"Best MSE: {best_mse}")
        print(f"Best Lag: {best_lag}")
        print(f"Best Parameters: {best_params}")

        # Best Lag: 14
        # Best Parameters: {'learning_rate': 0.1, 'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200, 'n_jobs': -1}

    def xgboost_date_predict(self, lags=28):
        import xgboost as xgb
        from sklearn.metrics import r2_score, mean_absolute_error
        # Create lag features
        self.create_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        X_train = specific_segment[:self.validation_end_date][lag_columns].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        
        X_test = specific_segment[self.validation_end_date:][lag_columns].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Initialize and train the XGBoost model
        xgb_model = xgb.XGBRegressor(
                                    n_estimators=200, 
                                    max_depth=50,
                                    random_state=42,
                                    learning_rate=0.01,
                                    n_jobs=-1
                                    )
        xgb_model.fit(X_train, Y_train)

        # Make predictions
        y_predict = xgb_model.predict(X_test)

        # Plotting the forecast alongside the actual test data
        plt.figure(figsize=(22, 6))
        plt.plot(Y_test.index, y_predict, color='blue', label='Predicted Sales')
        plt.plot(Y_test.index, Y_test, color='red', label='Actual Sales')
        plt.title(f'XGBoost-PureDate Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        # Evaluating the model's performance   
        r2_date_xgb = r2_score(Y_test, y_predict)
        print("XGBoost r2_score:", r2_date_xgb)

        # Day-by-day evaluation using MAE
        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)

    def optimize_date_xgboost(self, cv_splits=3):
        import xgboost as xgb
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        # Best parameters initialization
        best_mse = float('inf')
        best_lag = 0
        best_params = None

        # Define specific lag values to test
        lag_values = [14, 21, 28, 35, 42, 49]

        for lag in lag_values:
            self.create_lag_features(lags=lag)
            specific_segment = self.segmented_data[(self.store_number, self.product_type)].dropna()
            
            # Splitting the data
            train_data = specific_segment[:self.train_end_date]
            feature_cols = [col for col in train_data.columns if col != 'sales']
            X_train = train_data[feature_cols]
            y_train = train_data['sales']

            # Parameter grid for XGBoost
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_jobs': [-1]
            }

            # Time series cross-validator
            tscv = TimeSeriesSplit(n_splits=cv_splits)

            # Grid Search with cross-validation
            xgb_model = xgb.XGBRegressor(
                                        random_state=42,
                                        force_col_wise=True
                                        )
            
            grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Evaluate the best model
            best_model_mse = -grid_search.best_score_
            if best_model_mse < best_mse:
                best_mse = best_model_mse
                best_lag = lag
                best_params = grid_search.best_params_

        # Output the best settings
        print(f"Best MSE: {best_mse}")
        print(f"Best Lag: {best_lag}")
        print(f"Best Parameters: {best_params}")







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

    def randomforest_fit(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split, cross_val_score
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        X_train = specific_segment[:self.validation_end_date][['special_offer', 'id', 'store_nbr']].values
        Y_train = specific_segment[:self.validation_end_date]['sales']

        # Initialize the RandomForestRegressor with specified parameters
        self.model = RandomForestRegressor(n_estimators = self.n_estimators, 
                                           max_depth = self.max_depth, 
                                           random_state = self.random_state,
                                           max_features = self.max_features
                                           )
        # Perform cross-validation
        # cv_scores = cross_val_score(self.model, X_train, Y_train, cv=10)  # cv=5 for 5-folds cross-validation

        # Train the model on the entire training data
        self.model.fit(X_train, Y_train)

        # Print the mean cross-validation score
        # print("Average Cross-Validation Score:", np.mean(cv_scores))

    def randomforest_predict(self):
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        X_test = specific_segment[self.validation_end_date:][['special_offer', 'id', 'store_nbr']].values

        Y_test = specific_segment[self.validation_end_date:]['sales']

        y_predict = self.model.predict(X_test)

        mse = mean_squared_error(Y_test, y_predict)

        # Plot the forecast alongside the actual test data
        plt.figure(figsize=(12,6))
        plt.plot(Y_test.index, y_predict, color='blue', label='Predicted Sales')
        plt.plot(Y_test.index, Y_test, color='red', label='Actual Sales')
        plt.title('Sales Forecast vs Actuals')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        # Print the MSE
        print("Mean Squared Error:", mse)

    def randomforest_fine_tune(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        X_train = specific_segment[:self.train_end_date][['special_offer', 'id', 'store_nbr']].values
        Y_train = specific_segment[:self.train_end_date]['sales']

        # Define the parameter grid to search
        param_grid = {
            'n_estimators': [20, 50, 100, 150, 200, 250, 300, 350, 400, 500],
            'max_depth': [10, 12, 15, 20, 25, 30, 35, 40, 45, 50],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6, 10],
            'max_features': ['auto', 'sqrt']
        }

        # Initialize the RandomForestRegressor
        rf = RandomForestRegressor(random_state=self.random_state)

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, Y_train)

        # Update the model with the best parameters
        self.model = grid_search.best_estimator_

        # Print the best parameters
        print("Best Parameters:", grid_search.best_params_)
    



    def prepare_data(self, data):
        # Implement your feature engineering here
        pass

    def prepare_prediction_data(self, start_date, end_date):
        # Prepare the data for prediction for the given date range
        pass



# %%
