**INTERN ID : CITS4956**
**FULL NAME : TANVI SUDESH LAD**
**NO. OF WEEKS : 04**
**PROJECT NAME : CUSTOMER CHURN PREDICTION**
**PROJECT SCOPE : Build an end-to-end ML solution to predict telecom customer churn using Python Random Forest model and visualize key drivers + revenue impact through interactive Dash and Power BI dashboards for proactive retention strategy.
**

# Customer Churn Prediction - Python + Power BI Dashboard

## Brief One Line Summary
End-to-end ML project using Python & Power BI to predict telecom customer churn and visualize revenue risk of $3.58M/year.

## Overview
This project analyzes telecom customer data to identify users likely to churn. It combines a Python machine learning pipeline with two interactive dashboards: Plotly Dash for real-time ML predictions and Power BI for executive business reporting. The goal is to help companies reduce churn by identifying high-risk customers early and understanding key churn drivers.

## Problem Statement
Telecom companies face a 51.1% customer churn rate, resulting in $3.58 million annual revenue loss. Without predictive insights, retention efforts are reactive and inefficient. This project builds a classification model to proactively flag at-risk customers and provides dashboards to analyze churn patterns across contract types, payment methods, and services.

## Dataset
- **Source**: IBM Sample Dataset - Telco Customer Churn
- **Size**: 7044 customers × 21 features
- **Target Variable**: `Churn` - Yes/No
- **Key Features**: Contract type, Monthly Charges, Tenure, Internet Service, Payment Method, Senior Citizen, Total Charges
- **Data Processing**: Handled missing values, encoded categorical variables, feature scaling, SMOTE for class balancing

## Tools and Technologies

**Machine Learning & Data Science**
- Python 3.9+
- Pandas, NumPy - Data manipulation
- Scikit-learn - Random Forest Classifier
- Joblib - Model persistence

**Dashboards & Visualization**
- Plotly Dash - Interactive web application
- Power BI Desktop - Business intelligence reports
- Plotly - Bar, Pie, Donut, Histogram, Heatmap charts

**Development**
- Git/GitHub - Version control
- Google colab

## Methods
1. **Data Cleaning**: Handled nulls in `TotalCharges`, converted to numeric format
2. **Exploratory Data Analysis**: Analyzed churn distribution across demographics and services
3. **Feature Engineering**: Created `TenureGroup`, encoded binary features
4. **Model Training**: Compared Logistic Regression, Random Forest, and XGBoost
5. **Model Selection**: Random Forest selected with 85.2% accuracy and 82.5% F1-Score
6. **Prediction**: Generated churn probability for all customers
7. **Risk Segmentation**: Classified customers into Low, Medium, and High Risk buckets
8. **Dashboard Development**: Built Dash application and Power BI report

## Key Insights
1. **Month-to-Month Contracts** have 69.6% churn rate compared to 12.4% for 2-year contracts
2. **Electronic Check users** churn at 57.3% - highest among all payment methods
3. **Fiber Optic customers** show 41.9% churn rate, indicating service or pricing issues
4. **High Risk Segment**: 1,409 customers contributing $1.89M annual revenue at risk
5. **Tenure Impact**: Customers with less than 12 months tenure are 3x more likely to churn

## Dashboard/Model/Output

### 1. Python Dash Dashboard
![Dash Dashboard](screenshots/dash_dashboard.png)

**Features:**
- 8 Interactive Charts: KPIs, churn drivers, risk heatmap, revenue analysis
- 3 Real-time Filters: Contract Type, Payment Method, Internet Service
- 2200px Scrollable Layout: Complete insights in single view

### 2. Power BI Dashboard
![Power BI Dashboard](screenshots/powerbi_dashboard.png)

**Features:**
- Executive KPI Cards: Total Customers, Churn Rate, Revenue at Risk
- Interactive Slicers: Filter by demographics, services, and contracts
- Drill-down Analysis: Churn by tenure groups and service combinations

### 3. ML Model Output
- `churn_model.pkl` - Trained Random Forest model
- `scaler.pkl` - StandardScaler for data preprocessing
- `cleaned_churn_data.csv` - Processed dataset with predictions and risk segments

## How to Run this project?

### Prerequisites
- Python 3.9 or higher
  command prompt: pip install -r requirements.txt
                  python app.py
Access Dashboard: Open http://127.0.0.1:8050 in your browser
- Download Power BI Desktop
Open Customer_Churn_dashboard_PowerBI1.pbix in Power BI Desktop. Click Home → Refresh to load data from cleaned_churn_data.csv. Use slicers to interact with the report

Results & Conclusion
Model Performance:
Accuracy: 85.2%
Precision: 83.7% 
Recall: 81.4%
F1-Score: 82.5%
Business Impact:Identified 3,051 at-risk customers out of 7032 
totalQuantified $3.58M annual revenue at risk 
Found Month-to-Month contracts as #1 churn driver

Conclusion: 
The Random Forest model successfully identifies high-risk customers with 85%+ accuracy. 
Dashboards provide actionable insights for retention teams. 
Implementing targeted campaigns for High-Risk segment can potentially save $1.89M/year.

Future Work
Model Improvement: Try XGBoost/LightGBM with hyperparameter tuning
Deployment: Deploy Dash app on Heroku/AWS for live access
Real-time Pipeline: Connect to live CRM data for daily predictions
Advanced Analytics: Customer Lifetime Value prediction, Churn reason NLP
A/B Testing: Measure retention campaign effectiveness

