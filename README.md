## Customer Churn Risk Prediction & Analysis

This project builds an end-to-end customer churn risk analytics pipeline, combining exploratory data analysis, machine learning modelling, and interactive visualisation to identify high-risk customers and key drivers of churn.

The goal is not just to predict churn, but to translate model outputs into clear, actionable business insights that could support retention strategy, credit decisioning, or customer lifecycle management.

---

## Project Overview

**Dataset**
- Telco customer subscription data

**Objectives**
- Predict customer churn risk  
- Understand why customers churn  
- Surface high-risk customers for targeted intervention  

**Approach**
- Exploratory Data Analysis (EDA)
- Logistic regression baseline model
- Probability-based customer risk scoring
- Interactive dashboard for decision-makers

---

## Project Stages

### Exploratory Data Analysis (EDA)

**Purpose**  
Understand churn patterns, data quality issues, and high-level drivers before modelling.

**What was done**
- Examined churn distribution and class imbalance  
- Analysed churn by contract type, internet service, tenure, and charges  
- Identified strong early signals (e.g. month-to-month contracts, fibre optic users)

---

### Data Preparation & Feature Engineering

**Purpose**  
Create a clean, model-ready dataset while preserving interpretability.

**What was done**
- Standardised missing values and numeric types  
- Encoded categorical variables  
- Built a preprocessing pipeline including:
  - Imputation  
  - Scaling  
  - One-hot encoding  

---

### Churn Risk Modelling

**Purpose**  
Estimate churn probability for each customer in a transparent, explainable way.

**What was done**
- Trained a logistic regression baseline model  
- Evaluated performance using:
  - ROC-AUC  
  - Precision and recall  
  - Confusion matrix  
- Extracted and ranked feature coefficients to identify key drivers

---

## Key Findings from the Model

**Primary churn drivers**
- Short tenure significantly increases churn risk  
- Month-to-month contracts are far riskier than fixed-term contracts  
- Fibre optic customers show higher churn than DSL  
- Higher monthly charges correlate with increased churn probability  

These drivers align with real-world customer behaviour and validate the model’s credibility.

---

## Dashboard & Business Insights

The dashboard combines **portfolio-level churn monitoring** with **customer-level drill-down**, enabling a smooth transition from strategy to action.

**At a high level, it shows**
- Observed churn rate compared with the model’s average predicted churn risk  
- Total number of customers classified as high risk  
- How churn risk varies by key drivers such as contract type and internet service  

**At a granular level, it surfaces**
- Predicted churn probability per customer  
- Monthly charges (as a proxy for customer value)  
- Tenure and contract type  

This supports prioritisation of retention efforts and operational decision-making.

---

## Key Conclusions from the Analysis

- Contract structure is the strongest driver of churn, with month-to-month customers exhibiting substantially higher risk than fixed-term contracts.  
- Short customer tenure strongly correlates with churn, indicating early-lifecycle vulnerability.  
- Fibre optic customers show higher churn risk than DSL users, suggesting service-specific friction rather than pricing alone.  
- Higher monthly charges are associated with increased churn risk, particularly when combined with flexible contracts.  
- A simple, interpretable model can produce actionable churn risk signals when paired with clear segmentation and visualisation.