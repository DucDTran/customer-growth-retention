# Customer Growth and Retention Analysis

This project implements a comprehensive customer retention strategy using machine learning models to predict churn, estimate customer lifetime value (CLV), and prioritize retention efforts.

It features a detailed analysis notebook and a production-ready API for real-time inference.

## Business Context

The goal is to reduce churn, maximize CLV, and avoid over-treatment in a fintech/subscription/e-commerce setting with high Customer Acquisition Cost (CAC).

**Key Question:** If budget allows retention of only 20% of customers, which 20% should be prioritized to minimize revenue loss?

## Project Structure

```text
├── customer_retention_analysis.ipynb  # Primary analysis, training, and model evaluation
├── api.py                             # FastAPI application for inference
├── models/                            # Saved model artifacts (XGBoost, BG-NBD, etc.)
├── data/                              # Dataset files (customers.csv, transactions.csv)
└── README.md                          # Project documentation

