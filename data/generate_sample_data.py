"""
Sample Data Generator for AI CEO Project
Generates realistic corporate datasets for sales, HR, and business operations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sales_data(n_records=1000):
    """Generate sales dataset"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    dates = np.random.choice(dates, n_records)
    
    data = {
        'date': dates,
        'product_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], n_records),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_records),
        'sales_rep_id': np.random.choice([f'REP{i:03d}' for i in range(1, 51)], n_records),
        'units_sold': np.random.poisson(lam=50, size=n_records),
        'unit_price': np.random.normal(100, 20, n_records).round(2),
        'discount': np.random.uniform(0, 0.2, n_records).round(3),
        'marketing_spend': np.random.uniform(500, 5000, n_records).round(2),
    }
    
    df = pd.DataFrame(data)
    df['revenue'] = df['units_sold'] * df['unit_price'] * (1 - df['discount'])
    df['cost'] = df['units_sold'] * df['unit_price'] * 0.6  # 60% cost
    df['profit'] = df['revenue'] - df['cost']
    
    return df

def generate_hr_data(n_records=500):
    """Generate HR dataset"""
    np.random.seed(42)
    
    departments = ['Sales', 'Marketing', 'Operations', 'IT', 'Finance', 'HR']
    positions = ['Junior', 'Mid', 'Senior', 'Manager', 'Director']
    
    data = {
        'employee_id': [f'EMP{i:04d}' for i in range(1, n_records + 1)],
        'department': np.random.choice(departments, n_records),
        'position': np.random.choice(positions, n_records),
        'salary': np.random.normal(75000, 30000, n_records).round(2),
        'years_experience': np.random.exponential(5, n_records).round(1),
        'performance_score': np.random.uniform(60, 100, n_records).round(1),
        'hiring_date': pd.date_range(start='2015-01-01', end='2024-12-31', periods=n_records),
        'attrition_risk': np.random.uniform(0, 1, n_records).round(3),
    }
    
    df = pd.DataFrame(data)
    df['salary'] = np.maximum(df['salary'], 40000)  # Minimum salary
    df.loc[df['attrition_risk'] > 0.7, 'status'] = 'High Risk'
    df.loc[df['attrition_risk'] <= 0.7, 'status'] = 'Stable'
    
    return df

def generate_business_data(n_records=200):
    """Generate business operations dataset"""
    np.random.seed(42)
    
    quarters = pd.date_range(start='2020-01-01', end='2024-12-31', freq='Q')
    dates = np.random.choice(quarters, n_records)
    
    data = {
        'quarter': dates,
        'operational_cost': np.random.uniform(500000, 2000000, n_records).round(2),
        'marketing_budget': np.random.uniform(100000, 500000, n_records).round(2),
        'rd_investment': np.random.uniform(50000, 300000, n_records).round(2),
        'customer_satisfaction': np.random.uniform(70, 95, n_records).round(1),
        'market_share': np.random.uniform(10, 35, n_records).round(2),
        'competitor_count': np.random.poisson(lam=5, size=n_records),
        'economic_index': np.random.uniform(0.8, 1.2, n_records).round(3),
    }
    
    df = pd.DataFrame(data)
    return df

def generate_all_data():
    """Generate all datasets and save to CSV"""
    print("Generating sample datasets...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate datasets
    sales_df = generate_sales_data(1000)
    hr_df = generate_hr_data(500)
    business_df = generate_business_data(200)
    
    # Save to CSV
    sales_df.to_csv('data/sales_data.csv', index=False)
    hr_df.to_csv('data/hr_data.csv', index=False)
    business_df.to_csv('data/business_data.csv', index=False)
    
    print(f"✓ Generated sales_data.csv: {len(sales_df)} records")
    print(f"✓ Generated hr_data.csv: {len(hr_df)} records")
    print(f"✓ Generated business_data.csv: {len(business_df)} records")
    
    return sales_df, hr_df, business_df

if __name__ == '__main__':
    generate_all_data()
