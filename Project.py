import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'Amazon_Sale_Report.xlsx'
df = pd.read_excel(file_path)
print("This work correctly")

# Ensure 'Date' is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Histogram: Distribution of 'Amount'
plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'], bins=10, kde=True)
plt.title('Distribution of Amount')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

#insure the graph has been showed correctly by printing a string
print("Distribution of 'Amount' graph has been loaded")


# Bar Plot: Count of different 'Status'
plt.figure(figsize=(10, 6))
sns.countplot(x='Courier Status', data=df)
plt.title('Count of Different Order Status')
plt.xlabel('Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#insure the graph has been showed correctly by printing a string
print("Count of different 'Status' graph has been loaded")


# Line Plot: Sales Amount over Time
plt.figure(figsize=(14, 7))
sns.lineplot(x='Date', y='Amount', data=df)
plt.title('Sales Amount Over Time')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.show()

#insure the graph has been showed correctly by printing a string
print("Sales Amount over Time graph has been loaded")


# Bar Plot: Count of different 'Quantity'
plt.figure(figsize=(10, 6))
sns.countplot(x='Qty', data=df)
plt.title('Count of Different Order Quantity')
plt.xlabel('Quantity')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#insure the graph has been showed correctly by printing a string
print("Count of different 'Quantity' graph has been loaded")


############################
#Second Part Of Data Visualization(Visual Analysis):

#Visualize Sales Trends Over Time

# Extract month and year from 'Date'
df['Month'] = df['Date'].dt.month_name()

# Group by month name and calculate total sales
monthly_sales = df.groupby('Month')['Amount'].sum().reindex([
    'January', 'February', 'March', 'April', 'May', 'June', 
    'July', 'August', 'September', 'October', 'November', 'December'
]).reset_index()


# Line Plot: Monthly Sales Trends
plt.figure(figsize=(14, 7))
sns.lineplot(x='Month', y='Amount', data=monthly_sales)
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Sales Amount')
plt.xticks(rotation=45)
plt.show()

#insure the graph has been showed correctly by printing a string
print("Sales Trends Over Time graph has been loaded")


#Bar Plot for Sales by Category
sales_by_category = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=sales_by_category.index, y=sales_by_category.values, palette='viridis')
plt.title('Sales by Category')
plt.xlabel('Category')
plt.ylabel('Total Sales Amount')
plt.xticks(rotation=45)
plt.show()

#insure the graph has been showed correctly by printing a string
print("Top selling category graph has been loaded")


#Bar Plot for Sales by City

# Group by ship-city and calculate total sales
city_sales = df.groupby('ship-city')['Amount'].sum().sort_values(ascending=False).reset_index()

# Plot the top 10 cities by sales
top_cities = city_sales.head(10)

plt.figure(figsize=(14, 7))
sns.barplot(x='Amount', y='ship-city', data=top_cities, palette='viridis')
plt.title('Top 10 Cities by Sales')
plt.xlabel('Total Sales Amount')
plt.ylabel('City')
plt.show()
