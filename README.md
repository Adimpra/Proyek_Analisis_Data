# E-Commerce Dataset Analysis Project

## Overview
This project performs comprehensive analysis on e-commerce datasets to extract valuable insights about customers, products, sales patterns, and seller performance. The analysis includes data cleaning, exploratory data analysis, and visualization of key metrics.

## Datasets
The project uses multiple interconnected datasets:

1. **Customers Dataset**: Contains customer information including IDs, locations, and demographics.
2. **Geolocation Dataset**: Maps ZIP codes to geographic coordinates and regions.
3. **Orders Dataset**: Contains order information, status, and timestamps.
4. **Order Items Dataset**: Details about products within each order.
5. **Payments Dataset**: Payment information including methods and installments.
6. **Products Dataset**: Product details including categories, dimensions, and weights.
7. **Product Category Translation**: Translations of Portuguese category names to English.
8. **Sellers Dataset**: Information about sellers and their locations.
9. **Reviews Dataset**: Customer reviews and satisfaction scores.

## Project Structure
- `Data/`: Contains all raw dataset files
- `Proyek_Analisis_Data.ipynb`: Main Jupyter notebook containing all analysis
- `README.md`: Project documentation

## Setup Instructions
1. Ensure you have Python 3.x installed
2. Install required libraries:
   ```
   pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter
   ```
3. Clone this repository or download the files
4. Open the Jupyter notebook: `jupyter notebook Proyek_Analisis_Data.ipynb`

## Analysis Workflow
1. **Data Loading**: Import all datasets from CSV files
2. **Data Cleaning**:
   - Handle missing values (e.g., product category names, dimensions)
   - Check for duplicates
   - Convert data types as needed
3. **Exploratory Data Analysis**:
   - Analyze customer demographics and locations
   - Study product categories and their popularity
   - Investigate order patterns and delivery times
   - Analyze payment methods and pricing
   - Evaluate seller performance and customer satisfaction
4. **Visualization**: Create charts and graphs to illustrate findings
5. **Insights Generation**: Draw conclusions based on the analysis

## Key Insights
- Most popular product categories and their sales performance
- Customer geographic distribution across Brazil
- Payment preferences and trends over time
- Factors affecting delivery times and customer satisfaction
- Correlations between product attributes and sales performance

## Business Questions Answered
- Which product categories are the most and least popular?
- How do sales trends change over time?
- What are the customer segments based on RFM analysis?
- How does customer location influence orders and delivery?

## Author
Ahmad Imam Prasojo (ahmadimam.2020@student.uny.ac.id)
