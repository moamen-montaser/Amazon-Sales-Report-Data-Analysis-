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
print("First graph has been loaded")


# Bar Plot: Count of different 'Status'
plt.figure(figsize=(10, 6))
sns.countplot(x='Courier Status', data=df)
plt.title('Count of Different Order Status')
plt.xlabel('Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#insure the graph has been showed correctly by printing a string
print("second graph has been loaded")


# Line Plot: Sales Amount over Time
plt.figure(figsize=(14, 7))
sns.lineplot(x='Date', y='Amount', data=df)
plt.title('Sales Amount Over Time')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.show()

#insure the graph has been showed correctly by printing a string
print("3rd graph has been loaded")


# Bar Plot: Count of different 'Quantity'
plt.figure(figsize=(10, 6))
sns.countplot(x='Qty', data=df)
plt.title('Count of Different Order Quantity')
plt.xlabel('Quantity')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#insure the graph has been showed correctly by printing a string
print("4th graph has been loaded")
