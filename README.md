# Employee Attrition Analysis & Prediction

Why do employees leave, and can we spot who is at risk before they resign? Exploratory analysis, imbalance-aware machine learning and an interactive Power BI dashboard on IBM's HR dataset.

![Python Scikit-learn](https://img.shields.io/badge/Python-scikit--learn-blue)
![Imbalance Data:Smote](https://img.shields.io/badge/Imbalanced%20Data-SMOTE-orange)
![Power BI Dashboard](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811)

# The Problem

Replacing an employee is expensive, yet most organisations only learn why people leave during exit interviews, when it's too late. This project analyses employee records to:

• Identify the factors most strongly linked to attrition

• Build a model that flags employees at risk of leaving

• Present the findings in a dashboard that HR teams can explore themselves

# Key Findings

Driver	| Attrition rate
⏱️ Overtime: employees working overtime	| 30.5% vs 10.4% without (~3x higher)

🪜 Job level: entry-level (Level 1) employees |	26%, the highest of any level

⚖️ Work-life balance: employees rating it "Bad"	 | 31%

💼 Role: Sales Representatives	| 40%, the highest of any role

💍 Marital status: single employees	| 25.5% vs ~10–12% for married/divorced

Younger employees, those with fewer years at the company, frequent business travellers and those without stock options also show higher attrition.

# Dashboard

Interactive Power BI dashboard with filters for department, gender, age, job role and education field. It breaks down attrition by overtime, income, job level, role, travel frequency and salary hike.


# Modelling Approach

The challenge: class imbalance. Only 16% of employees left. A model that predicts "stays" for everyone would score 84% accuracy and still be useless, because it would never catch the people HR needs to retain.

What was done:

• Removed constant columns (EmployeeCount, Over18, StandardHours) and one-hot encoded categorical features

• Random Forest tuned with GridSearchCV (optimised for F1, class_weight='balanced')

• Gradient Boosting trained on SMOTE-balanced data to give the model enough examples of employees who left

• Evaluated with precision, recall and F1 on the attrition class, not just accuracy

# Result:
Both models reached ~86% accuracy. The difference shows up where it matters: compared with Random Forest, Gradient Boosting with SMOTE raised F1 on employees who left from 0.16 to 0.43 and recall from 10% to 38%, catching nearly 4x more at-risk employees.

Next Steps

This was an early project, and a few improvements are planned:

• Use a stratified train/test split to keep the attrition ratio consistent

• Tune Gradient Boosting with the same grid search used for Random Forest

• Add ROC-AUC / PR-AUC and decision-threshold tuning to balance precision and recall

• Generate predictions on the test set only, for an unbiased view of model performance

• Explain predictions with SHAP to show why each employee is flagged


# Repository Structure

├── Employee_attrition.ipynb          # EDA, preprocessing, modelling

├── WA_Fn-UseC_-HR-Employee-Attrition.csv

├── Employee_attrition_PBI.pbix       # Power BI dashboard

├── ss_PBI_Attrition.png              # dashboard screenshot

└── gradient_boosting_model.pkl       # saved model

# Data

IBM HR Analytics Employee Attrition dataset:
1,470 employees, 35 attributes (demographics, job role, satisfaction, compensation, tenure. 
[Link](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)


# Tech Stack

Analysis & ML: Python, Pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE) 
Visualization: Seaborn, Matplotlib, hvPlot, Power BI
