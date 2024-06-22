import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output


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
print("(Distribution of 'Amount') graph has been loaded")


# Bar Plot: Count of different 'Status'
plt.figure(figsize=(10, 6))
sns.countplot(x='Courier Status', data=df)
plt.title('Count of Different Order Status')
plt.xlabel('Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#insure the graph has been showed correctly by printing a string
print("(Count of different 'Status') graph has been loaded")


# Line Plot: Sales Amount over Time
plt.figure(figsize=(14, 7))
sns.lineplot(x='Date', y='Amount', data=df)
plt.title('Sales Amount Over Time')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.show()

#insure the graph has been showed correctly by printing a string
print("(Sales Amount over Time) graph has been loaded")


# Bar Plot: Count of different 'Quantity'
plt.figure(figsize=(10, 6))
sns.countplot(x='Qty', data=df)
plt.title('Count of Different Order Quantity')
plt.xlabel('Quantity')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#insure the graph has been showed correctly by printing a string
print("(Count of different 'Quantity') graph has been loaded")


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
print("(Sales Trends Over Time) graph has been loaded")


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
print("(Top selling category) graph has been loaded")


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


###########################################
######################################
######################################
######################################
# Initialize the Dash app

monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
monthly_sales['Date'] = monthly_sales['Date'].dt.to_timestamp()


app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Sales Dashboard"),
    
    # Histogram of Amount
    html.Div([
        html.H2("Distribution of Amount"),
        dcc.Graph(
            figure=px.histogram(df, x='Amount', title='Distribution of Amount', nbins=50)
        )
    ]),
    
    # Line plot of Monthly Sales Trends
    html.Div([
        html.H2("Monthly Sales Trends"),
        dcc.Graph(
            figure=px.line(monthly_sales, x='Date', y='Amount', title='Monthly Sales Trends')
        )
    ]),
    
    # Dropdown to select top-selling products
    html.Div([
        html.H2("Amount Of Products By Size"),
        dcc.Dropdown(
            id='product-dropdown',
            options=[{'label': i, 'value': i} for i in df['Size'].unique()],
            value=df['Size'].unique()[0]
        ),
        dcc.Graph(id='top-products-bar')
    ]),
    
    # Geographical Sales Distribution
    html.Div([
        html.H2("Sales Distribution by City"),
        dcc.Graph(
            figure=px.scatter_geo(df, locations="ship-city", locationmode="country names",
                                  color="Amount", size="Amount",
                                  hover_name="ship-city", title='Sales Distribution by City')
        )
    ])
])

# Callback to update top-selling products bar plot based on dropdown selection
@app.callback(
    Output('top-products-bar', 'figure'),
    Input('product-dropdown', 'value')
)
def update_bar_chart(selected_product):
    filtered_df = df[df['Size'] == selected_product]
    top_products = filtered_df.groupby('Size')['Amount'].sum().reset_index()
    fig = px.bar(top_products, x='Size', y='Amount', title=f'Top-Selling Product: {selected_product}')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)