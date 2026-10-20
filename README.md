# CodeAlpha_CarPricePrediction

**Data Science Internship — Task 3: Car Price Prediction with Machine Learning**

## Overview
This project predicts the selling price of used cars (in lakhs) from features such as showroom price, age, kilometers driven, fuel type, seller type, transmission, and number of previous owners.

## Workflow
1. **Load & inspect** the dataset (301 listings; 2 duplicates, no missing values)
2. **Clean**: remove duplicates, drop `Car_Name` (too many unique values), engineer `Car_Age` from `Year`
3. **Exploratory Data Analysis**: price distribution, correlations, price by fuel/seller/transmission, outlier check (IQR)
4. **Preprocess**: one-hot encoding, 80/20 train/test split, feature scaling
5. **Train & compare** Linear Regression, Decision Tree, and Random Forest
6. **Validate** with shuffled 5-fold cross-validation, and test the effect of removing price outliers
7. **Evaluate** the final model: test metrics, predicted vs actual plot, feature importance

## Results
| Model | Single split R² | 5-fold CV R² (mean) |
|---|---|---|
| Decision Tree | 0.828 | **0.923** |
| Random Forest | 0.522 | 0.859 |
| Linear Regression | 0.753 | 0.852 |

The final **Decision Tree** reaches R² = 0.828 on the held-out test set, with an average error (MAE) of about 1.1 lakhs. `Present_Price` accounts for about 91% of the model's feature importance, followed by `Car_Age` (about 8%).

## Key Takeaways
- A single train/test split on ~300 rows can be misleading, so cross-validation was used. The rows in this file are ordered, so the folds must be shuffled.
- 16 high-priced cars (5.4%) are outliers but genuine listings, so they were kept.
- Predictions are less reliable for the most expensive cars, and the data has no information on condition or brand.

## Tech Stack
Python, pandas, numpy, scikit-learn, matplotlib, seaborn

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook CarPricePrediction.ipynb
```

## Dataset
`car data.csv`, provided by CodeAlpha.

## Author
Akor Emmanuel — CodeAlpha Data Science Intern (Student ID: CA/DF1/305488)