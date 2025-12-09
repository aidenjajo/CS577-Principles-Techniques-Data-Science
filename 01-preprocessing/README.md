# CS577-Principles-Techniques-Data-Science
Projects and assignments from CS577: Principles and Techniques of Data Science at SDSU

# Data Preprocessing

A comprehensive data preprocessing pipeline implementing fundamental techniques for preparing raw data for machine learning models.

## Overview

This project demonstrates core preprocessing steps including missing value imputation, categorical encoding, train-test splitting, and feature scaling using Python and scikit-learn.

## Technologies

- Python 3.x
- pandas
- scikit-learn (LabelEncoder, StandardScaler, train_test_split)

## Implementation

**Key Steps:**
1. **Missing Value Handling** - Mean imputation for numerical features
2. **Categorical Encoding** - Label encoding for object-type columns
3. **Train-Test Split** - 80/20 split with random_state=42
4. **Feature Scaling** - StandardScaler for normalization

## Usage

```bash
python preprocessing_assignment.py
```

## Output

The script displays:
- First 5 rows of scaled training data
- Training and testing set dimensions
- List of encoded categorical columns

## Key Concepts

- Proper train-test splitting to prevent overfitting
- Feature scaling fit on training data only (prevents data leakage)
- Reproducible results through random state seeding

---

**CS577 - Principles and Techniques of Data Science**  
San Diego State University | Spring 2025
