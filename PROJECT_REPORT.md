# Project Report: Student Performance Analysis and Prediction System

**Student:** Swayam Kumar  
**Roll No.:** AIML-A6/SEP-11833  
**Project Type:** Minor Project

## Abstract
This project demonstrates an end-to-end workflow for exploring student academic indicators and estimating final examination marks using Linear Regression.

## Objectives
- Inspect and clean tabular academic data.
- Explore relationships between study hours, attendance, prior marks, assignments, and final marks.
- Train a Linear Regression model and evaluate it using MAE, RMSE, and R².
- Generate exploratory visualizations and cross-validation results.

## Tools
Python, pandas, NumPy, scikit-learn, Matplotlib, and Seaborn.

## Workflow
1. Load the CSV and inspect missing values and duplicate rows.
2. Convert numeric columns and remove records without a target value.
3. Generate descriptive statistics and exploratory plots.
4. Split the data into training and test sets.
5. Fit a median-imputation and Linear Regression pipeline.
6. Calculate evaluation metrics and save output files.

## Dataset limitation
The repository's starter data is a small illustrative sample and is not suitable for drawing real-world conclusions. Use a larger, properly licensed dataset and rerun the analysis before presenting final empirical findings.

## Conclusion
The repository provides a reproducible starter workflow for regression-based student performance analysis. Model scores and conclusions must be based on a documented, appropriate dataset.