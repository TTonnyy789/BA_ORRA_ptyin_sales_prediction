#%%#
### Step 1  ######################################################################
### Import essential library and load the data

# Import pandas, numpy, and etc.
import openpyxl 
import numpy as np
import pandas as pd
import seaborn as sns
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Read the data and understanding our data set, assign your personal file path for the purpose of loading data
product = pd.read_csv("/Users/ttonny0326/BA_ORRA/Python_Programming/Products_Information.csv")


# Using info() and head() functions to briefly read the overall structure
product_info = product.info()
product_describe = product.describe()
product_rows = product.head()


# Check our data set
print(product_info)
print("------------------------------------------------")
print(" ")
print(product_describe)
print("------------------------------------------------")
print(" ")
print(product_rows)
###-------------------------------------------------------------------------------



#%%#
### Step 2  ######################################################################
### Data pre-processing, such as handle Missing Values and etc.

# Deal with missing value
missing_values = product.isnull().sum()


# Convert the 'date' column to datetime format and set it as the index
product['date'] = pd.to_datetime(product['date'])
product.set_index('date', inplace=True)


# Check the latest processing result
print(missing_values)
print("------------------------------------------------")
print(" ")
print(product.head())

# Save processed data
# product.to_csv("processed_product_data.csv")
###-------------------------------------------------------------------------------



#%%#
### Step 3  ######################################################################
### Overall EDA processing and Graph plotting

# Setting aesthetics for plots
sns.set(style="darkgrid")
sns.set_theme()
sns.set_context("talk", font_scale=0.8, rc={"lines.linewidth": 1})

# Plotting the overall special_offer over whole time period
# product['special_offer'] = product['special_offer'].astype('int')

# Plot sales over all time (supplementary)
# Aggregating sales data
overall_sales = product.groupby('date')['sales'].sum()
# Creating the plot
plt.figure(figsize=(14, 7))
plt.plot(overall_sales, label='Total Sales')
plt.title('Overall sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.show()

# Summarize the stats of sales
print(product['sales'].describe())

# Distribution of sales
plt.figure(figsize=(14, 7))
sns.histplot(product['sales'], bins=50, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.xlim(0, 15000) 
plt.show()

# Sales by Store (important)
plt.figure(figsize=(14, 7))
sns.boxplot(x='store_nbr', y='sales', data=product)
plt.title('Sales by Store')
plt.xlabel('Store Number')
plt.ylabel('Sales')
plt.show()

# Sales by Product
plt.figure(figsize=(14, 7))
sns.boxplot(x='product_type', y='sales', data=product)
plt.title('Sales by Product')
plt.xlabel('Product Type')
plt.ylabel('Sales') 
plt.xticks(rotation=90)
plt.show()
###-------------------------------------------------------------------------------



#%%#
### Step 4  ######################################################################
### Data advanced processing, group by both store and product (store-product combination)


# Ensure 'product_type' is of categorical data type
product['product_type'] = product['product_type'].astype('category')

##################################################################################
##################################################################################
########### Creating a dictionary to hold the time series data for each product-store combination (Important Step)
##################################################################################
##################################################################################

segmented_data = {}

##################################################################################
##################################################################################

# Grouping the data by store and product, and storing each group in the dictionary
for (store, product_type), group in product.groupby(['store_nbr', 'product_type'], observed=True):
    segmented_data[(store, product_type)] = group[['sales', 'special_offer', 'id']]


# Displaying the number of segments created and a sample segment
num_segments = len(segmented_data)
sample_segment_key = list(segmented_data.keys())[1]
# sample_segment_key = list(segmented_data.keys())[0] -> will be store 1's AUTOMOTIVE
sample_segment_data = segmented_data[sample_segment_key].head()


print("Number of segments:", num_segments)
print("------------------------------------------------")
print(" ")
print("Sample segment key:", sample_segment_key)
print("------------------------------------------------")
print(" ")
print(sample_segment_data)
###-------------------------------------------------------------------------------




#%%#
### Step 5  ######################################################################
### Segment Data Checking, for individual store and product


# After group our data by both store and product, we can select the specific store's product by execute following command
# Define the store number and product type you are interested in(store number 1 and its 'AUTOMOTIVE' product)
store_number = 30
product_type = 'BEAUTY'


# Access the sales data for the specific store and product
specific_segment = segmented_data[(store_number, product_type)]


# Display the data using first 15 days data
print("Number of Store:", store_number)
print("Specific Product Type:", product_type)
print("------------------------------------------------")
print(" ")
print(specific_segment.head(15))
###-------------------------------------------------------------------------------




#%%#
### Step 6-1  ####################################################################
### Time series data checking (Important Step)
### Following are the standards of time series data

### 1. Datetime Index: Ensure that your data is indexed by a datetime type, which enables resampling and other time-based operations.

### 2. Frequency: Time series should have a consistent frequency (e.g., hourly, daily, monthly). You should check if the data is regular and if not, resample it to a regular frequency if appropriate.

### 3. Stationarity: Many time series models assume that the data is stationary or can be made stationary through transformations (e.g., differencing, log transformation). Stationarity means the statistical properties of the series do not change over time.


# Set up graph configuration
sns.set(style="darkgrid")
sns.set_theme()
sns.set_context("talk", font_scale=0.8, rc={"lines.linewidth": 1})


# Define the overall sales number
overall_sales = product.groupby('date')['sales'].sum()


# Plot the overall sales number over whole time period (same as Step 3, overall EDA processing but only for the sales and whole time period)
plt.figure(figsize=(14, 7))
overall_sales.plot(title='Overall Sales Over Time', label='Total Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


# Plot Sample Individual Time Series checking (specific store and product)
# Looping through the selected segments and plotting each one separately
for key in list(segmented_data.keys())[:3]:
    segment = segmented_data[key]
    
    # Creating a new plot for each segment
    plt.figure(figsize=(14, 4))
    plt.plot(segment.index, segment['sales'])
    
    # Adding titles and labels
    plt.title(f'Time Series of Sales - Store: {key[0]}, Product: {key[1]}')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    
    # Displaying the plot
    plt.show()
# Results indicate that there was a tiny trend between store 1's AUTOMOTIVE and store 1's BEAUTY. However, for the store 1's BABY CARE, there was no sales number throughout whole time period. 


# As a result of that, check specific store-product data (store 1's BABYCARE)
# Filter the Data for store 1 and BABYCARE
store_1_babycare_data = product[(product['store_nbr'] == 1) & (product['product_type'] == 'BABY CARE')]


# Displaying the first few rows of the data
print(store_1_babycare_data.head(15))


# Check Sales Data
# Summarizing the sales data to check if all sales values are zero
sales_summary = store_1_babycare_data['sales'].describe()
print("\nSales Summary:")
print(sales_summary)


# Checking if all sales values are zero
all_zero_sales = store_1_babycare_data['sales'].sum() == 0
print("\nAll Sales Are Zero:", all_zero_sales)
###-------------------------------------------------------------------------------



#%%#
### Step 6-2  ####################################################################
### Statistical Tests for Time Series Data Confirmation (ADF test) for specific store-product


# First and foremost, deal with the specific store-product Augmented Dickey-Fuller (ADF) test on the sales data of the "BEAUTY" product in store 1
# Store 1's BEAUTY data
store_1_beauty_data = product[(product['store_nbr'] == 1) & (product['product_type'] == 'BEAUTY')]['sales']

# Performing ADF test
adf_result_store_1_beauty = adfuller(store_1_beauty_data.dropna())


# Creating a DataFrame to hold the results
adf_results = pd.DataFrame(columns=['Store-Product', 'ADF Statistic', 'p-value', 'Critical Values'])


# Creating a new DataFrame for the result of Store 1's BEAUTY
new_row = pd.DataFrame({
    'Store-Product': ['Store 1 - BEAUTY'], 
    'ADF Statistic': [adf_result_store_1_beauty[0]], 
    'p-value': [adf_result_store_1_beauty[1]], 
    'Critical Values': [adf_result_store_1_beauty[4]]
})

# Adding the result to the adf_results DataFrame
adf_results = pd.concat([adf_results, new_row], ignore_index=True)


# Displaying the ADF result for Store 1's BEAUTY
print(adf_results)
# Given that the ADF statistic (-4.213618) is less than the critical value at the 1% level (-3.4343), and the p-value is very low (0.000625), we can reject the null hypothesis and conclude that the time series data for "BEAUTY" in store 1 is stationary.
###-------------------------------------------------------------------------------



#%%#
### Step 6-3  ####################################################################
### Statistical Tests for Stationary Time Series Data Confirmation (ADF test) for all store-product combinations


# For checking the stationarity of time series data
# DataFrames to hold different types of results
adf_results_zero_sales = pd.DataFrame(columns=['Store-Product', 'Sales Sum', 'Special Offer Sum'])
adf_results_non_stationary_non_zero = pd.DataFrame(columns=['Store-Product', 'ADF Statistic', 'p-value', 'Critical Values', 'Sales Sum', 'Special Offer Sum'])
adf_results_stationary_non_zero = pd.DataFrame(columns=['Store-Product', 'ADF Statistic', 'p-value', 'Critical Values', 'Sales Sum', 'Special Offer Sum'])

# Dictionaries to hold segments
zero_sales_segments = {}
non_stationary_non_zero_segments = {}
stationary_non_zero_segments = {}

# Iterate through the segments
for (store, product), data in segmented_data.items():
    sales_data = data['sales'].dropna()
    special_offer_sum = data['special_offer'].sum()
    sales_sum = sales_data.sum()
    
    # Check for zero sales
    if sales_sum == 0:
        zero_sales_segments[(store, product)] = data
        new_row = pd.DataFrame([{
            'Store-Product': f'Store {store} - {product}', 
            'Sales Sum': sales_sum,
            'Special Offer Sum': special_offer_sum
        }])
        adf_results_zero_sales = pd.concat([adf_results_zero_sales, new_row], ignore_index=True)
    else:
        # Non-zero sales, perform ADF test
        try:
            adf_result = adfuller(sales_data)
            store_product = f'Store {store} - {product}'
            adf_row = pd.DataFrame([{
                'Store-Product': store_product, 
                'ADF Statistic': adf_result[0], 
                'p-value': adf_result[1], 
                'Critical Values': adf_result[4],
                'Sales Sum': sales_sum,
                'Special Offer Sum': special_offer_sum
            }])

            # Check if the series is stationary
            if adf_result[1] < 0.05:
                stationary_non_zero_segments[(store, product)] = data
                adf_results_stationary_non_zero = pd.concat([adf_results_stationary_non_zero, adf_row], ignore_index=True)
            else:
                non_stationary_non_zero_segments[(store, product)] = data
                adf_results_non_stationary_non_zero = pd.concat([adf_results_non_stationary_non_zero, adf_row], ignore_index=True)

        except Exception as e:
            print(f"Error performing ADF test on Store {store} - {product}: {str(e)}")


# Displaying the ADF results
print("Zero Sales Segments:")
print(adf_results_zero_sales)
print("Non-Stationary Non-Zero Sales Segments:")
print(adf_results_non_stationary_non_zero)
print("Stationary Non-Zero Sales Segments:")
print(adf_results_stationary_non_zero)


# Save to CSV if needed
# adf_results_zero_sales.to_csv("adf_results_zero_sales.csv")
# adf_results_non_stationary_non_zero.to_csv("adf_results_non_stationary_non_zero.csv")
# adf_results_stationary_non_zero.to_csv("adf_results_stationary_non_zero.csv")



##### Important Notification !!! #####

### Even if a time series is non-stationary, it is still a time series and can be analyzed and forecasted; it just means that we may need to transform the data to make it stationary or use models that can handle non-stationarity.

### If there are some segment has been considered non-stationary time series data, we should build other models to forecasting those types of condition


##### Conclusion #####

### A time series does not have to pass the ADF test to be considered "a time series." The ADF test simply determines whether or not the data is stationary. Many time series are inherently non-stationary, but they can still be modeled and forecasted after applying the appropriate transformations or by using models designed to handle trends and seasonality, such as ARIMA with differencing or SARIMA (Seasonal ARIMA).

###-------------------------------------------------------------------------------



#%%#
### Step 7-1  ####################################################################
### Sales Forecasting - Time series Decomposition Plotting (Take Non-Stationary Time Series Sub-dataset as an example, Store-1 'BOOKS')

### This is just a simple checking of one of non-zero number of sales and non-stationary subset (Store 1, BOOKS)



### Use store_number = 1, product_type = 'BABY CARE' as our example for the purpose of ensuring the outcome prediction will work successfully for our complete data set

store_number = 1
product_type = 'BABY CARE'
specific_segment = segmented_data[(store_number, product_type)]

# Select results that we tend to observe by using "seasonal_decompose" function
result_baby = seasonal_decompose(specific_segment['sales'], model='additive', period=7)

# Adjusting the line thickness for each component of the decomposition plot individually
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 7))

# Original Time Series (for store_number = 1, product_type = 'BABY CARE')
result_baby.observed.plot(ax=ax1, linewidth=1)
ax1.set_ylabel('Observed')

# Trend Component (for store_number = 1, product_type = 'BABY CARE')
result_baby.trend.plot(ax=ax2, linewidth=1)
ax2.set_ylabel('Trend')

# Seasonal Component (for store_number = 1, product_type = 'BABY CARE')
result_baby.seasonal.plot(ax=ax3, linewidth=1)
ax3.set_ylabel('Seasonality')

# Residual Component (for store_number = 1, product_type = 'BABY CARE')
result_baby.resid.plot(ax=ax4, linewidth=1)
ax4.set_ylabel('Residual')

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Adjust the layout to prevent overlapping
plt.suptitle('Time Series Decomposition of Store 1 - BABY CARE', fontsize=15, y=1.02)
plt.show()



### Use store_number = 1, product_type = 'BEAUTY' as our example for the purpose of ensuring the outcome prediction will work successfully for our complete data set

store_number = 1
product_type = 'BEAUTY'
specific_segment = segmented_data[(store_number, product_type)]

# Select results that we tend to observe by using "seasonal_decompose" function
result_beauty = seasonal_decompose(specific_segment['sales'], model='additive', period=7)

# Adjusting the line thickness for each component of the decomposition plot individually
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 7))

# Original Time Series (for store_number = 1, product_type = 'BEAUTY')
result_beauty.observed.plot(ax=ax1, linewidth=1)
ax1.set_ylabel('Observed')

# Trend Component (for store_number = 1, product_type = 'BEAUTY')
result_beauty.trend.plot(ax=ax2, linewidth=1)
ax2.set_ylabel('Trend')

# Seasonal Component (for store_number = 1, product_type = 'BEAUTY')
result_beauty.seasonal.plot(ax=ax3, linewidth=1)
ax3.set_ylabel('Seasonality')

# Residual Component (for store_number = 1, product_type = 'BEAUTY')
result_beauty.resid.plot(ax=ax4, linewidth=1)
ax4.set_ylabel('Residual')

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Adjust the layout to prevent overlapping
plt.suptitle('Time Series Decomposition of Store 1 - BEAUTY', fontsize=15, y=1.02)
plt.show()



### Use store_number = 1, product_type = 'BOOKS' as our example for the purpose of ensuring the outcome prediction will work successfully for our complete data set
store_number = 1
product_type = 'BOOKS'
specific_segment = segmented_data[(store_number, product_type)]

# Select results that we tend to observe by using "seasonal_decompose" function
result_books = seasonal_decompose(specific_segment['sales'], model='additive', period=7)

# Adjusting the line thickness for each component of the decomposition plot individually
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 7))

# Original Time Series (for store_number = 1, product_type = 'BOOKS')
result_books.observed.plot(ax=ax1, linewidth=1)
ax1.set_ylabel('Observed')

# Trend Component (for store_number = 1, product_type = 'BOOKS')
result_books.trend.plot(ax=ax2, linewidth=1)
ax2.set_ylabel('Trend')

# Seasonal Component (for store_number = 1, product_type = 'BOOKS')
result_books.seasonal.plot(ax=ax3, linewidth=1)
ax3.set_ylabel('Seasonality')

# Residual Component (for store_number = 1, product_type = 'BOOKS')
result_books.resid.plot(ax=ax4, linewidth=1)
ax4.set_ylabel('Residual')

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Adjust the layout to prevent overlapping
plt.suptitle('Time Series Decomposition of Store 1 - BOOKS', fontsize=15, y=1.02)
plt.show()



##### Conclusion #####

### 1. Zero Sales Throughout the Period (Store 1 - BABY CARE)
# For series that are entirely zeros, there's no need for forecasting since there's no variation in the data. However, if you expect that these series might change in the future due to new products, promotions, or market changes, you may want to consider using qualitative forecasting methods (like market research or Delphi method) instead of quantitative methods.


### 2. Non-zero Sales and Stationary (Store 1 - BEAUTY)
# For stationary time series data with non-zero sales, ARIMA or SARIMA models are suitable since they assume the underlying process is stable over time:

# ARIMA/SARIMA Models: Good for capturing the autocorrelation in stationary data.
# Exponential Smoothing Models: Such as Holt-Winters, which are simpler than ARIMA models and can perform well if the data has a clear trend or seasonality.


### 3. Non-zero Sales but Non-stationary (Store 1 - BOOKS)
# For non-stationary time series data, you typically need to transform the data to make it stationary before applying ARIMA-type models. 

# However, if the series cannot be transformed to become stationary or if the non-stationarity is not related to trend or seasonality, you might use:

# Random Walk Model: If you believe the future sales will continue to increase or decrease as they have in the past.
# Machine Learning Models: Non-time series-specific models such as Random Forest or Gradient Boosting can be used. These models can capture complex relationships in the data without assuming stationarity. Feature engineering will be crucial here, including lagged variables, rolling statistics, etc.

###-------------------------------------------------------------------------------




#%%#
### Step 7-2  ####################################################################
### Sales Forecasting - Fit ARIMA Model, including train-test splitting, model training(model fitting), and performance evaluating (For Time Series Sub-dataset)


# Store number and product type for which to forecast sales
store_number = 1
product_type = 'BEAUTY'
# Retrieve the specific segment of the data
specific_segment = segmented_data[(store_number, product_type)]


# Ensuring the date index is in datetime format
specific_segment.index = pd.to_datetime(specific_segment.index)


# Defining the split dates
train_end_date = '2016-12-31'
validation_end_date = '2017-07-31'


# Splitting the data again
train_data = specific_segment[:train_end_date]
validation_data = specific_segment[train_end_date:validation_end_date]
test_data = specific_segment[validation_end_date:]


# Displaying the sizes of each set
print("Training Data Size:", len(train_data))
print("Validation Data Size:", len(validation_data))
print("Testing Data Size:", len(test_data))
print(" ")
print("------------------------------------------------")
print(" ")

# Displaying the first few rows of the training data
print(train_data.head(10))
print(validation_data.head(10))
print(test_data)
###-------------------------------------------------------------------------------



#%%#
### Step 7-4  ####################################################################
### Import the class from forecasters to conduct fitting and forecasting

from forcasters import StationarySalesForecaster, NonStationarySalesForecaster, ZeroSalesForecaster

store1_beauty = StationarySalesForecaster(store_number=1, product_type='BEAUTY')
store1_beauty.arima_fit()
store1_beauty.arima_predict()

store1_babycare = NonStationarySalesForecaster(store_number=1, product_type='AUTOMOTIVE')
store1_babycare.randomforest_fit()
store1_babycare.randomforest_predict()





#%%#
### Step 7-5  ####################################################################
### Forecasting Trail on NON-Zero and stationary data

# Retrieve the specific segment based on store number and product type
specific_segment = segmented_data[(1, 'BEAUTY')]

# Ensure the date index is in datetime format and set it as index
specific_segment.index = pd.to_datetime(specific_segment.index)

# Split the data into training, validation, and test sets
train_end_date = '2016-12-31'
validation_end_date = '2017-07-31'
train_data = specific_segment[:train_end_date]['sales']
validation_data = specific_segment[train_end_date:validation_end_date]['sales']
test_data = specific_segment[validation_end_date:]['sales']

# Fit an ARIMA model to the training data
model = ARIMA(train_data, order=(8, 2, 6))  # Adjust the order parameters as necessary
fitted_model = model.fit()

# Forecast using the fitted model
forecast = fitted_model.forecast(steps=len(test_data))

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
print('Mean Squared Error:', mse)





#%%#
### Step 7-3  ####################################################################
### Grid search for the best arguments for the ARIMA model (p, d, q) from 0 to 9
import itertools

# Store number and product type for which to forecast sales
store_number = 1
product_type = 'BEAUTY'

# Retrieve the specific segment of the data
specific_segment = segmented_data[(store_number, product_type)]

# Ensuring the date index is in datetime format
specific_segment.index = pd.to_datetime(specific_segment.index)

# Defining the split dates
train_end_date = '2016-12-31'
validation_end_date = '2017-07-31'

# Splitting the data
train_data = specific_segment[:train_end_date]['sales']
validation_data = specific_segment[train_end_date:validation_end_date]['sales']
test_data = specific_segment[validation_end_date:]['sales']

# Hyperparameter tuning for ARIMA
p = d = q = range(0, 10)
pdq = list(itertools.product(p, d, q))

best_mse = float('inf')
best_order = None
best_model = None

for order in pdq:
    try:
        model = ARIMA(train_data, order=order)
        model_fit = model.fit()
        predictions = model_fit.forecast(len(validation_data))
        mse = mean_squared_error(validation_data, predictions)
        if mse < best_mse:
            best_mse = mse
            best_order = order
            best_model = model_fit
    except:
        continue

print(f"Best ARIMA Order: {best_order}")
print(f"Best MSE: {best_mse}")

# Retraining the best model on the full dataset (train + validation)
best_model_full = ARIMA(pd.concat([train_data, validation_data]), order=best_order)
best_model_full = best_model_full.fit()

# Forecasting on the test set
test_predictions = best_model_full.forecast(len(test_data))

# Evaluating performance on the test set
test_mse = mean_squared_error(test_data, test_predictions)
print(f"Test MSE: {test_mse}")


# %%#


