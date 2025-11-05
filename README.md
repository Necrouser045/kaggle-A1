# Kaggle Assignment 1 – House Price Prediction (MLP Term 3 2025)

This repository contains my notebook and code submission for **Kaggle Assignment 1 (MLP Term 3 2025)**.  
The goal of this project is to **predict house prices** using various machine learning models and data preprocessing pipelines.  
It demonstrates complete data cleaning, feature engineering, visualization, and model evaluation processes.

---

## Files in this Repository
| File | Description |
|------|--------------|
| `notebook.ipynb` | Main Jupyter notebook with all data analysis, preprocessing, and model training steps |
| `train.csv` / `test.csv` | Dataset files provided for training and evaluation (from Kaggle input path) |
| `README.md` | Project overview and documentation |

---

## Objective
To build a **regression model** that predicts the price of properties based on several numerical and categorical attributes such as:
- Area type  
- Availability  
- Location  
- Size  
- Total square feet  
- Number of bathrooms and balconies  

---

## Dataset Overview
**Training data:** `train.csv`  
**Test data:** `test.csv`  

| Feature | Description |
|----------|--------------|
| `id` | Unique property identifier |
| `area_type` | Category of the property (type_I, type_II, etc.) |
| `availability` | Availability status or date |
| `location` | Area/locality of the property |
| `size` | Number of bedrooms (e.g., “2 BHK”, “3 Bedroom”) |
| `total_sqft` | Total built-up area in square feet |
| `bath` | Number of bathrooms |
| `balcony` | Number of balconies |
| `price` | Target variable – price in lakhs (training data only) |

---

## 🧹 Data Cleaning & Preprocessing

### 1 Missing Values
- Numerical features: filled with **median** values  
- Categorical features: filled with **mode** values  
- Result: no missing data remaining

### 2 Outlier Handling
- Detected using **IQR** and percentile filtering  
- Trimmed top 1 % extreme values for features like `total_sqft`, `bath`, and `price`  

### 3 Duplicate Check
- No duplicate rows detected  

---
## Exploratory Data Analysis (EDA)
- **Price distribution:** Right-skewed — log transformation applied  
- **Top 10 locations:** Price variations visualized using boxplots  
- **Total sqft vs Price:** Positive correlation observed  
- **Correlation heatmap:** Highest correlation between price, area, and number of bathrooms  

---

## Feature Engineering
- Extracted **BHK** count from the `size` column  
- Added binary flag **is_ready** from `availability` (“Ready To Move”)  
- Frequency encoded high-cardinality columns like `location` and `availability`  
- One-hot encoded `area_type` (4 unique classes)  
- Removed rare categories and grouped them under “Other”  

---

## Model Building & Evaluation
Seven machine learning algorithms were trained and compared:

| Model | R² Score | MSE |
|:------|----------:|----------:|
| **XGBoost (tuned)** | **0.6123** | **6953.04** |
| Random Forest (tuned) | 0.5071 | 8838.73 |
| Gradient Boosting (tuned) | 0.4788 | 9347.33 |
| SVM (RBF) | 0.4671 | 9556.33 |
| Linear Regression | 0.4546 | 9780.49 |
| K-Nearest Neighbors | 0.4158 | 10477.18 |
| Decision Tree | −0.3030 | 23369.52 |

✅ **Best Model:** XGBoost (after hyperparameter tuning)  
💡 R² = 0.61 → explains ~61 % of price variance on validation data.

---

## Hyperparameter Tuning
- Performed using **GridSearchCV (3-fold CV)** for:
  - Random Forest  
  - XGBoost  
  - Gradient Boosting  

**Best parameters discovered:**
| Model | Parameters |
|--------|-------------|
| Random Forest | max_depth = 10, min_samples_split = 5, n_estimators = 200 |
| XGBoost | learning_rate = 0.1, max_depth = 3, n_estimators = 200 |
| Gradient Boosting | learning_rate = 0.1, max_depth = 4, n_estimators = 100 |

---

## Final Submission
Predictions generated using the **Linear Regression** pipeline for submission:
```python
submission = pd.DataFrame({
    'id': X_test['id'],
    'price': predictions
})
submission.to_csv('submission.csv', index=False)