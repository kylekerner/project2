# Personalization:  Customer Segmentation and Product Recommendation System
## Introduction
Problem Statement

Group 2 Retail Company - aims to enhance customer experience by providing personalized product recommendations based on their purchasing behavior. The goal is to:

1. Identify customers who shop more on discounted products.

2. Identify customers who buy non-discounted products.

3. Identify customers who have made 30 or more previous purchases.

4. Identify customers who give high ratings (4 and above).

5. Recommend products accordingly, with an additional 20% discount for customers who have made 30 or more purchases.

6. Ensure that recommendations are limited to 20 products or less and that they match the customer's size preferences.

7. Calculate and present the total amount, discount applied, and net total for the recommended products.

Business Logic
To achieve these goals, we implemented the following business logic:

Segmentation: Customers are segmented based on their purchasing behavior into four categories:

Discount Shoppers: Customers who primarily shop on discounted products.

Non-Discount Shoppers: Customers who primarily buy non-discounted products and have fewer than 30 purchases and lower review ratings.

High Purchase Shoppers: Customers who have made 30 or more purchases.

High Rating Givers: Customers who frequently give high ratings (4 and above).

Product Recommendations: Products are recommended based on the customer's segment, gender, and size preferences. For high purchase shoppers, an additional 20% discount is applied.

Calculation of Totals: For the recommended products, the total amount, discount applied, and net total are calculated and presented to the customer.

Implementation

The implementation involved several steps, from data loading and preprocessing to feature engineering, customer segmentation, and product recommendation. Here's a detailed breakdown:

1. Data Loading and Preprocessing

We loaded product data and customer purchase history data from CSV files and ensured that certain columns were in the correct format. This step included converting 'Yes'/'No' values to 1/0 and handling non-numeric values by converting them to numeric types and filling missing values.

2. Visualization

We created functions to visualize both product and customer data. These visualizations included distributions of discounts and gender for products and distributions of review ratings, previous purchases, and purchase frequency for customers.

3. Feature Engineering

We engineered new features to capture key aspects of customer behavior:

Discount_Purchase_Ratio: The ratio of discounted purchases to total purchases.

High_Rating_Purchase_Ratio: The ratio of high-rated purchases to total purchases.

Previous_Purchases: The maximum number of previous purchases by a customer.

Frequency_of_Purchases: The maximum frequency of purchases by a customer.

These features helped us in segmenting the customers effectively.

4. Customer Segmentation

Customers were segmented based on the following conditions:

Discount Shoppers: Customers with a high ratio of discounted purchases.

Non-Discount Shoppers: Customers with a low ratio of discounted purchases, fewer than 30 previous purchases, and low review ratings.

High Purchase Shoppers: Customers with 30 or more previous purchases.

High Rating Givers: Customers with high review ratings (4 and above).

The segments were defined using clear and mutually exclusive conditions to ensure customers fell into only one segment.

5. Model Training

We trained a Random Forest classifier to predict customer segments based on the engineered features. This model was evaluated using a classification report and a confusion matrix to ensure its accuracy.

6. Product Recommendation

A function was created to recommend products based on the customer's segment, gender, and size preferences. If the customer fell into the "High Purchase Shoppers" segment, an additional 20% discount was applied to the recommended products. The recommendations were limited to 20 products or less to keep them manageable.

7. Calculation of Totals

Another function calculated the total amount, discount applied, and net total for the recommended products. This information was presented to the customer along with the product recommendations.

8. User Interaction

The main function brought everything together, prompting the user for a Customer ID, segmenting the customer, and providing personalized product recommendations along with detailed pricing information.

Summary

This implementation allows the business to enhance customer experience by providing personalized product recommendations based on detailed analysis of their purchasing behavior. By segmenting customers and tailoring recommendations to their preferences, we can improve customer satisfaction and potentially increase sales. The additional 20% discount for loyal customers (those with 30 or more purchases) helps in retaining these valuable customers, fostering loyalty, and encouraging further purchases.

This project provides recommendations for customers based on previous purchases, discount or no discount, price, color, and size.

## Data Source
https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset

The data used for this analysis is a collection of consumer information crucial for market analysis and tailored marketing strategies. The dataset is stored in a CSV file shopping_behavior_updated.csv, located in the project2 folder.  We modified this dataset to better achieve the goals as consumer_purchase_data.csv and product_data.csv, also located in the project2 folder.

## Libraries and Dependencies
The project utilizes several Python libraries for data analysis and visualization:

**Pandas**: For data manipulation and analysis.
**Scikit-learn (sklearn)**: For dataset loading, metrics, train_test_split, accuracy_score, classification report, Logistic Regression, RandomForestClassifier,  KMeans, AgglomerativeClustering, Birch, StandardScaler, balanced_accuracy_score, OneHotEncoder, OrdinalEncoder, PCA, LabelEncoder, TruncatedSVD, mean_squared_error, r2_score
**Numpy**: For working with numerical data
**Scipy (scipy)**: For statistical analysis.
**Matplotlib**: For creating visualizations.

## Installation
To run the code locally, follow these steps:
1. Clone the repository to your machine:
    1.  git clone https://github.com/kylekerner/project2.git
2. Navigate to the project directory:
    2.  cd project2

## Usage
Ensure you have Python installed on your machine.
Install the project dependencies as described in the installation section.
Navigate to the notebook file/branch. For example: sonu from the project explorer.
Open the notebook using the Jupyter extension in VSCode.
Run the cells within the notebook to execute the analysis interactively.
Project Structure
The project repository is organized into branches corresponding to each contributor:
sonu: Contains code and analysis related to personalized customer recommendations.
Kyle: Contains code and analysis related to classification.
Scott: Contains code and analysis related to customer product recommendation based on label encoding and Truncated Singular Variable Decomposition (SVD).
Each branch includes scripts, notebooks, and visualizations based on each person's approach to solving the problem.

## Data Preprocessing
The CSV data is read into a Pandas DataFrame.
Rows with missing or NaN (Not a Number) values are removed from the DataFrame to ensure data integrity.

## Analysis
Used Label Encoding to prepare the data for Truncated SVD to make purchase recommendations to the customer based on past purchases.  Adjusted R-squared score achieved was 0.8136 which is considered a good score.

### Results
Adjusted R-squared score achieved for Truncated SVD was 0.8136 which is considered a good score which would indicate that the recommendations for the customers are good.

### Conclusion
Through this analysis, we aim to provide recommended personalized items for a customer to purchase in the future. The results obtained can serve to assist customer future product needs and wants..


## Acknowledgements
### Dataset Credits
The data used in this project was sourced from Kaggle. There was no listed author for this dataset.  We would like to thank Zee Solver, the owner of this dataset, for the information provided.

## Code credits
We would like to thank Open AI's ChatGPT for the code suggestion to use the Truncated SVD to find customer recommendations based customer's previously purchased items.


#### Contributors
* Kyle Kerner
* Sonu Sharma
* Randolph "Scott" Bradley

