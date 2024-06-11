# project2

1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
•	pandas: For data manipulation and analysis.
•	numpy: For numerical operations.
•	sklearn.model_selection: For splitting the dataset into training and testing sets.
•	sklearn.ensemble: For using the RandomForestClassifier.
•	sklearn.metrics: For evaluating the model performance.
•	matplotlib.pyplot and seaborn: For data visualization.
2. Load the Product Data
# Load the actual product table
product_df = pd.read_csv('product_data.csv')

# Ensure 'Discount_Applied' and 'Gender' columns are in the correct format
product_df['Discount_Applied'] = product_df['Discount_Applied'].map({'Yes': 'Yes', 'No': 'No'})
product_df['Gender'] = product_df['Gender'].astype('category')
•	product_df: Reads product data from a CSV file.
•	Discount_Applied: Maps 'Yes'/'No' to 'Yes'/'No' for consistency.
•	Gender: Converts the Gender column to a categorical type.
3. Visualize Product Data
# Visualize product data
def visualize_product_data():
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Discount_Applied', data=product_df)
    plt.title('Distribution of Discount Applied in Products')
    plt.xlabel('Discount Applied')
    plt.ylabel('Number of Products')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Gender', data=product_df)
    plt.title('Distribution of Product Gender')
    plt.xlabel('Gender')
    plt.ylabel('Number of Products')
    plt.show()

visualize_product_data()
•	visualize_product_data(): Function to plot the distribution of discounts and gender in the product data using count plots.
4. Load Customer Purchase History Data
# Load customer purchase history data
df = pd.read_csv('customer_purchase_data.csv')

# Convert 'Yes'/'No' in 'Discount_Applied' to 1/0
df['Discount_Applied'] = df['Discount_Applied'].map({'Yes': 1, 'No': 0})

# Ensure numeric columns are indeed numeric and handle non-numeric values
numeric_columns = ['Review_Rating', 'Discount_Applied', 'Previous_Purchases', 'Frequency of Purchases']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
•	df: Reads customer purchase history data from a CSV file.
•	Discount_Applied: Converts 'Yes'/'No' to 1/0 for numerical analysis.
•	numeric_columns: Ensures certain columns are numeric and fills missing values with 0.
5. Visualize Customer Purchase Data
# Visualize customer purchase data
def visualize_customer_data():
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Review_Rating'], kde=True)
    plt.title('Distribution of Review Ratings')
    plt.xlabel('Review Rating')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Previous_Purchases'], kde=True)
    plt.title('Distribution of Previous Purchases')
    plt.xlabel('Previous Purchases')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Frequency of Purchases'], kde=True)
    plt.title('Distribution of Frequency of Purchases')
    plt.xlabel('Frequency of Purchases')
    plt.ylabel('Frequency')
    plt.show()

visualize_customer_data()
•	visualize_customer_data(): Function to plot the distributions of review ratings, previous purchases, and purchase frequency using histograms.
6. Feature Engineering
# Feature Engineering
high_rating_threshold = 4

df['Discount_Purchase_Ratio'] = df.groupby('Customer_ID')['Discount_Applied'].transform('sum') / df.groupby('Customer_ID')['Item_Purchased_ID'].transform('count')
df['High_Rating_Purchase_Ratio'] = df.apply(lambda x: 1 if x['Review_Rating'] >= high_rating_threshold else 0, axis=1)
df['High_Rating_Purchase_Ratio'] = df.groupby('Customer_ID')['High_Rating_Purchase_Ratio'].transform('sum') / df.groupby('Customer_ID')['Item_Purchased_ID'].transform('count')
df['Previous_Purchases'] = df.groupby('Customer_ID')['Previous_Purchases'].transform('max')
df['Frequency_of_Purchases'] = df.groupby('Customer_ID')['Frequency of Purchases'].transform('max')
•	high_rating_threshold: Sets the threshold for what constitutes a high rating (4).
•	Discount_Purchase_Ratio: Calculates the ratio of discounted purchases to total purchases for each customer.
•	High_Rating_Purchase_Ratio: Calculates the ratio of high ratings to total purchases for each customer.
•	Previous_Purchases: Captures the maximum number of previous purchases per customer.
•	Frequency_of_Purchases: Captures the maximum frequency of purchases per customer.
7. Visualize Engineered Features
# Visualize feature engineering with different charts
def visualize_features():
    # Histogram for Discount Purchase Ratio
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Discount_Purchase_Ratio'], bins=10, kde=True)
    plt.title('Distribution of Discount Purchase Ratio')
    plt.xlabel('Discount Purchase Ratio')
    plt.ylabel('Number of Customers')
    plt.show()
    
    # Violin Plot for High Rating Purchase Ratio
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=df['High_Rating_Purchase_Ratio'])
    plt.title('Distribution of High Rating Purchase Ratio')
    plt.xlabel('High Rating Purchase Ratio')
    plt.ylabel('Density')
    plt.show()

visualize_features()
•	visualize_features(): Function to plot the distribution of discount purchase ratios using a histogram and high rating purchase ratios using a violin plot.
8. Define Customer Segments
# Define segments with clear and mutually exclusive conditions
conditions = [
    (df['Discount_Applied'] == 1),  # Customer who shop more on discount products
    (df['Discount_Applied'] == 0) & (df['Previous_Purchases'] < 30) & (df['Review_Rating'] < high_rating_threshold),  # Customer who buy non discount products
    (df['Previous_Purchases'] >= 30),  # Customer who’s Previous Purchases are 30 and more
    (df['Review_Rating'] >= high_rating_threshold)  # Customer who give ratings 4 and more
]

# Assign segment IDs
choices = [1, 2, 3, 4]

# Apply the conditions
df['Segment'] = np.select(conditions, choices, default=0)

# Correct segment names for plotting
segment_names = {
    1: "Discount Shoppers",
    2: "Non-Discount Shoppers",
    3: "High Purchase Shoppers",
    4: "High Rating Givers",
    0: "Unsegmented"
}

# Map segment names
df['Segment_Name'] = df['Segment'].map(segment_names)
•	conditions: Defines conditions for segmenting customers into different groups based on their purchase behavior.
•	choices: Segment IDs for each condition.
•	Segment_Name: Maps segment IDs to meaningful names for better understanding.
9. Visualize Customer Segments
# Visualize segments
def visualize_segments():
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Segment_Name', data=df, order=segment_names.values())
    plt.title('Distribution of Customer Segments')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=45)
    plt.show()

visualize_segments()
•	visualize_segments(): Function to plot the distribution of customer segments using a count plot.
10. Visualize Segments as Clusters
# Visualize segments as clusters
def visualize_clusters():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Discount_Purchase_Ratio', y='High_Rating_Purchase_Ratio', hue='Segment', palette='viridis')
    plt.title('Customer Segments as Clusters')
    plt.xlabel('Discount Purchase Ratio')
    plt.ylabel('High Rating Purchase Ratio')
    plt.show()

visualize_clusters()
•	visualize_clusters(): Function to plot customer segments as clusters using a scatter plot. Each point represents a customer, colored by their segment.
11. Train a Model to Predict the Segment
# Train a model to predict the segment
features = ['Discount_Purchase_Ratio', 'High_Rating_Purchase_Ratio', 'Frequency_of_Purchases', 'Previous_Purchases']
X = df[features]
y = df['Segment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
•	features: Defines the feature set used to train the model.
•	X, y: Input features and target variable.
•	train_test_split: Splits the data into training and testing sets.
•	RandomForestClassifier: Trains a random forest model to predict customer segments.
•	classification_report: Prints the performance of the model on the test set.
12. Visualize Model Performance
# Visualize model performance
def visualize_model_performance():
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title('Confusion Matrix of the Random Forest Model')
    plt.show()

visualize_model_performance()
•	visualize_model_performance(): Function to display the confusion matrix of the trained model, showing the model's accuracy and areas of confusion.
13. Segment a Single Customer
# Function to segment a single customer using the trained model
def segment_customer(customer_id):
    customer_data = df.loc[df['Customer_ID'] == customer_id, features]
    
    if customer_data.empty:
        print("Customer ID not found.")
        return None
    
    features_data = customer_data.iloc[0].values.reshape(1, -1)
    segment = model.predict(features_data)[0]
    return segment
•	segment_customer(): Segments a single customer based on their ID using the trained model.
14. Recommend Products
# Function to recommend products based on the segment, customer gender, and size
def recommend_products(segment, gender, size):
    if segment == 1:
        recommendations = product_df.loc[(product_df['Discount_Applied'] == 'Yes') & (product_df['Size'] == size) & ((product_df['Gender'] == gender) | (product_df['Gender'] == 'Unisex'))]
    elif segment == 2:
        recommendations = product_df.loc[(product_df['Discount_Applied'] == 'No') & (product_df['Size'] == size) & ((product_df['Gender'] == gender) | (product_df['Gender'] == 'Unisex'))]
    elif segment == 3:
        recommendations = product_df.loc[(product_df['Size'] == size) & ((product_df['Gender'] == gender) | (product_df['Gender'] == 'Unisex'))].copy()
        recommendations['Discount_Applied'] = 'Yes'  # Apply additional discount for high purchase customers
        recommendations['Price'] = recommendations['Price'] * 0.8  # Apply 20% discount
    elif segment == 4:
        recommendations = product_df.loc[(product_df['Review_Rating'] >= high_rating_threshold) & (product_df['Size'] == size) & ((product_df['Gender'] == gender) | (product_df['Gender'] == 'Unisex'))]
    else:
        recommendations = product_df.loc[(product_df['Size'] == size) & ((product_df['Gender'] == gender) | (product_df['Gender'] == 'Unisex'))]
    
    # Limit the recommendations to 20 products or less
    recommendations = recommendations.head(20)
    
    return recommendations
•	recommend_products(): Recommends products based on the customer segment, gender, and size preferences. Applies an additional 20% discount for high purchase customers.
15. Calculate Totals
# Function to calculate total amount, discount applied, and net total
def calculate_totals(recommendations):
    total_amount = recommendations['Price'].sum()
    discount_applied = total_amount * 0.2 if recommendations['Discount_Applied'].any() else 0
    net_total = total_amount - discount_applied
    return total_amount, discount_applied, net_total
•	calculate_totals(): Calculates the total amount, discount applied, and net total for the recommended products.
16. Main Function
# Prompt user for Customer_ID and produce recommendations
def main():
    visualize_segments()
    visualize_clusters()
    
    customer_id = int(input("Enter Customer ID: "))
    customer_data = df.loc[df['Customer_ID'] == customer_id]
    if customer_data.empty:
        print("Customer ID not found.")
        return
    
    customer_gender = customer_data['Gender'].iloc[0]
    customer_segment = segment_customer(customer_id)
    if customer_segment is None:
        return
    
    # Get the most common size purchased by the customer
    customer_size = customer_data['Size'].mode()[0]
    
    recommended_products = recommend_products(customer_segment, customer_gender, customer_size)
    total_amount, discount_applied, net_total = calculate_totals(recommended_products)
    
    segment_names = {
        1: "Customer who do shop more on discount products",
        2: "Customer who buy non discount products",
        3: "Customer who’s Previous Purchases are 30 and more",
        4: "Customer who give ratings 4 and more"
    }
    
    print(f"Customer ID: {customer_id}")
    print(f"Customer Segment: {segment_names[customer_segment]}")
    print("Recommended Products:")
    print(recommended_products)
    print(f"Total Amount: ${total_amount:.2f}")
    print(f"Discount Applied: ${discount_applied:.2f} (20% Discount Applied)")
    print(f"Net Total: ${net_total:.2f}")

if __name__ == "__main__":
    main()
•	main(): The main function ties everything together. It visualizes segments and clusters, prompts the user for a Customer ID, segments the customer, recommends products, and calculates totals, displaying all relevant information.

