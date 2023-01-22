# Let's talk with Black-friday-sales Data

I downloaded black friday sales data from kaggle(https://www.kaggle.com/datasets/sdolezel/black-friday). I used various libraries and modules to make this project eg: Numpy, Pandas, Matplotlib, Seaborn, opendatasets and scipy. 

## Problem Statement
 A retail company ABC Private Limited wants to understand the customer purchese behaviour (specifically purshase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. The dataset also contains customer demographics (Gender, Age, Occupation, City Category, marital status, stay_in_current _city), ,product detalis (product_id and product category) and Total purchase_amount from last month.
Data Overview

-	User_ID: Unique ID of the user.
-	Product_ID: Unique ID of the product.
-	Gender: indicates the gender of the person making the transaction.
-	Age: indicates the age group of the person making the transaction.
-	Occupation: shows the occupation of the user, already labeled with numbers 0 to 20.
-	City_Category: User's living city category. Cities are categorized into 3 different categories 'A', 'B' and 'C'.
-	Stay_In_Current_City_Years: Indicates how long the users has lived in this city.
-	Marital_Status: is 0 if the user is not married and 1 otherwise.
-	Product_Category_1 to _3: Category of the product. All 3 are already labaled with numbers.
-	Purchase: Purchase amount.

## Downloading the Dataset

- we import 'opendatasets' module to download the dataset from kaggle.
- we import 'os' module to check data in directory.

## Data Preparation and Cleaning

we check the empty values in each column and fill it to the backward and forward method with fillna function. Prepare the data for Data Analysis and visualization.  

### Handling the missing Data.


## Exploratory Analysis and Visualization

Before we ask the questions about the dataset we will classify the numeric and non-numeric columns and analyze each and every column of the dataset. Explore the distribution of the each numeric columns and basic charts for non-numeric columns and find useful insight about the columns.








 
