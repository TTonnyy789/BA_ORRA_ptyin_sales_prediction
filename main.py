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
product = pd.read_csv("/Users/ttonny0326/GitHub_Project/BA_ORRA_python_Grocery_prediction/Products_Information.csv")


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

# Deal with missing value, firstly check the missing value
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
### Overall EDA and Graph plotting

# Setting plotting configuration for plots
sns.set(style="darkgrid")
sns.set_theme()
sns.set_context("talk", font_scale=0.8, rc={"lines.linewidth": 1})

# Plotting the overall special_offer over whole time period
# product['special_offer'] = product['special_offer'].astype('int')

# Plot sales over all time (supplementary plot)
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

# Sales by Product (important)
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
### Data advanced processing, group by both store and product (store-product combinations)


# Ensure 'product_type' is of categorical data type
product['product_type'] = product['product_type'].astype('category')

##################################################################################
##################################################################################
#####   Creating a dictionary to hold the data for each product-store combination (Important Step)
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
### Individual store and product demonstration (store-product combinations)


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
### Individual combinations of store and product grouping by checking the scales of the sales data for the purpose of segment the data into zero sales, non-zero sales.


# Set up graph configuration
sns.set(style="darkgrid")
sns.set_theme()
sns.set_context("talk", font_scale=0.8, rc={"lines.linewidth": 1})


# Demonstrate the sales data for store 1 and its 'BABY CARE' product, which consists of zero sales during the whole period of time
store_number = 1
product_type = 'BABY CARE'
specific_segment = segmented_data[(store_number, product_type)]

# Ensure 'date' is the index and in the correct format
specific_segment.index = pd.to_datetime(specific_segment.index)

# Filter the data to include only dates up to 2017-07-30
specific_segment = specific_segment[:"2017-07-30"]

# Setting up the plot
sns.set(style="darkgrid")
sns.set_theme()
sns.set_context("talk", font_scale=0.8, rc={"lines.linewidth": 1})

# Creating the plot
plt.figure(figsize=(14, 7))
plt.plot(specific_segment.index, specific_segment['sales'], linewidth=1)

# Adding titles and labels
plt.title(f'Sales Over Time for Store {store_number} - {product_type}')
plt.xlabel('Date')
plt.ylabel('Sales')

plt.show()


# Demonstrate the sales data for store 1 and its 'BEVERAGES' product, which consists of non-zero sales during the whole period of time
store_number = 1
product_type = 'BEVERAGES'
specific_segment = segmented_data[(store_number, product_type)]

# Ensure 'date' is the index and in the correct format
specific_segment.index = pd.to_datetime(specific_segment.index)

# Filter the data to include only dates up to 2017-07-30
specific_segment = specific_segment[:"2017-07-30"]

# Setting up the plot
sns.set(style="darkgrid")
sns.set_theme()
sns.set_context("talk", font_scale=0.8, rc={"lines.linewidth": 1})

# Creating the plot
plt.figure(figsize=(22, 6))
plt.plot(specific_segment.index, specific_segment['sales'], linewidth=1)

# Adding titles and labels
plt.title(f'Sales Over Time for Store {store_number} - {product_type}')
plt.xlabel('Date')
plt.ylabel('Sales')

plt.show()


# Demonstrate the sales data for store 1 and its 'BEVERAGES' product, which consists of non-zero sales during the whole period of time
store_number = 1
product_type = 'BOOKS'
specific_segment = segmented_data[(store_number, product_type)]

# Ensure 'date' is the index and in the correct format
specific_segment.index = pd.to_datetime(specific_segment.index)

# Filter the data to include only dates up to 2017-07-30
specific_segment = specific_segment[:"2017-07-30"]

# Setting up the plot
sns.set(style="darkgrid")
sns.set_theme()
sns.set_context("talk", font_scale=0.8, rc={"lines.linewidth": 1})

# Creating the plot
plt.figure(figsize=(22, 6))
plt.plot(specific_segment.index, specific_segment['sales'], linewidth=1)

# Adding titles and labels
plt.title(f'Sales Over Time for Store {store_number} - {product_type}')
plt.xlabel('Date')
plt.ylabel('Sales')

plt.show()

### Conclusion ###

### We can see that the sales data for store 1 and its 'BABY CARE' product consists of zero sales during the whole period of time,
### and the sales data for store 1 and its 'BEVERAGES' product consists of non-zero sales during the whole period of time.

### Therefore, we can segment the data into zero sales, non-zero sales. However, we need to consider the sales data for store 1 and its 'BOOKS' product, which consists of both zero sales and non-zero sales during the whole period of time.

### Therefore, we need to segment the data into zero sales, non-zero sales, and mixed sales.

###-------------------------------------------------------------------------------



#%%#
### Step 6-2  ####################################################################
### Create a functions to segment the data into zero sales, non-zero sales 

# Function to load and process data
def load_and_process_data(filepath):
    product = pd.read_csv(filepath)
    product['date'] = pd.to_datetime(product['date'])
    product.set_index('date', inplace=True)
    product['product_type'] = product['product_type'].astype('category')
    return product

# Function to segment data
def segment_data(product):
    segmented_data = {}
    for (store, product_type), group in product.groupby(['store_nbr', 'product_type'], observed=True):
        segmented_data[(store, product_type)] = group
    return segmented_data

# Function to divide segments into zero and non-zero sales
def divide_segments(segmented_data):
    zero_sales_segments = []
    non_zero_sales_segments = []

    for key, segment in segmented_data.items():
        store, product_type = key
        segment = segment[:"2017-07-30"]  # Filter data up to 2017-07-30

        if segment['sales'].sum() == 0:
            zero_sales_segments.append(segment.assign(store=store, product=product_type))
        else:
            non_zero_sales_segments.append(segment.assign(store=store, product=product_type))

    return pd.concat(zero_sales_segments).reset_index(), pd.concat(non_zero_sales_segments).reset_index()

# Main script
if __name__ == "__main__":
    filepath = "/Users/ttonny0326/BA_ORRA/Python_Programming/Products_Information.csv"
    product = load_and_process_data(filepath)
    segmented_data = segment_data(product)
    zero_sales_segments, non_zero_sales_segments = divide_segments(segmented_data)

    # Example of how to use the segmented data
    print("Zero Sales Segments:")
    print("------------------------------------------------")
    print(zero_sales_segments.head())
    print("\n")
    print("Non-Zero Sales Segments:")
    print("------------------------------------------------")
    print(non_zero_sales_segments.tail())


# Save processed data
# zero_sales_segments.to_csv("zero_sales_segments.csv")
# non_zero_sales_segments.to_csv("non_zero_sales_segments.csv")

###-------------------------------------------------------------------------------



#%%#
### Step 7-1  ####################################################################
### Import forecasting models and use the models to predict the sales for each store-product combination

from forecasters import SalesForecaster, ZeroSalesForecaster





##### Conclusion #####


###-------------------------------------------------------------------------------



#%%#
### Step 7-2  ####################################################################
### 


##### Conclusion #####

###-------------------------------------------------------------------------------



#%%#
### Step 7-3  ####################################################################
### 






