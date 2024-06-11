# Personalization:  Customer Segmentation and Product Recommendation System
## Introduction
This project provides recommendations for customers based on previous purchases, discount or no discount , price, color, and size.

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

