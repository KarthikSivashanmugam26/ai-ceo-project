"""
Data Pipeline for AI CEO Project
Loads, cleans, preprocesses data and engineers KPIs
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataPipeline:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.sales_df = None
        self.hr_df = None
        self.business_df = None
        self.master_df = None
        
    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")
        
        try:
            self.sales_df = pd.read_csv(f'{self.data_dir}/sales_data.csv')
            self.hr_df = pd.read_csv(f'{self.data_dir}/hr_data.csv')
            self.business_df = pd.read_csv(f'{self.data_dir}/business_data.csv')
            
            # Convert date columns
            self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
            self.hr_df['hiring_date'] = pd.to_datetime(self.hr_df['hiring_date'])
            self.business_df['quarter'] = pd.to_datetime(self.business_df['quarter'])
            
            print(f"✓ Loaded sales data: {len(self.sales_df)} records")
            print(f"✓ Loaded HR data: {len(self.hr_df)} records")
            print(f"✓ Loaded business data: {len(self.business_df)} records")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please run data/generate_sample_data.py first")
            raise
            
    def clean_data(self):
        """Clean and preprocess data"""
        print("Cleaning data...")
        
        # Clean sales data
        self.sales_df = self.sales_df.dropna()
        self.sales_df = self.sales_df[self.sales_df['revenue'] > 0]
        self.sales_df = self.sales_df[self.sales_df['profit'] > -10000]  # Remove extreme outliers
        
        # Clean HR data
        self.hr_df = self.hr_df.dropna()
        self.hr_df = self.hr_df[self.hr_df['salary'] > 0]
        
        # Clean business data
        self.business_df = self.business_df.dropna()
        
        print("✓ Data cleaning completed")
        
    def engineer_kpis(self):
        """Engineer key performance indicators"""
        print("Engineering KPIs...")
        
        # Sales KPIs (aggregated by month)
        sales_monthly = self.sales_df.copy()
        sales_monthly['year_month'] = sales_monthly['date'].dt.to_period('M')
        
        sales_kpis = sales_monthly.groupby('year_month').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'units_sold': 'sum',
            'marketing_spend': 'sum',
            'discount': 'mean'
        }).reset_index()
        
        sales_kpis['profit_margin'] = (sales_kpis['profit'] / sales_kpis['revenue'] * 100).round(2)
        sales_kpis['revenue_per_unit'] = (sales_kpis['revenue'] / sales_kpis['units_sold']).round(2)
        sales_kpis['marketing_roi'] = ((sales_kpis['revenue'] - sales_kpis['marketing_spend']) / sales_kpis['marketing_spend'] * 100).round(2)
        
        # HR KPIs (aggregated by month)
        hr_kpis = self.hr_df.copy()
        hr_kpis['year_month'] = hr_kpis['hiring_date'].dt.to_period('M')
        
        hr_monthly = hr_kpis.groupby('year_month').agg({
            'employee_id': 'count',
            'salary': ['sum', 'mean'],
            'attrition_risk': 'mean',
            'performance_score': 'mean'
        }).reset_index()
        
        hr_monthly.columns = ['year_month', 'employee_count', 'total_payroll', 'avg_salary', 'avg_attrition_risk', 'avg_performance']
        hr_monthly['attrition_rate'] = (hr_monthly['avg_attrition_risk'] * 100).round(2)
        
        # Business KPIs (already quarterly)
        business_kpis = self.business_df.copy()
        business_kpis['year_month'] = business_kpis['quarter'].dt.to_period('M')
        
        # Merge all KPIs
        master_df = sales_kpis.merge(hr_monthly, on='year_month', how='outer')
        master_df = master_df.merge(business_kpis, on='year_month', how='outer')
        
        # Fill missing values
        master_df = master_df.ffill().bfill()
        
        # Calculate growth rates
        master_df['revenue_growth'] = master_df['revenue'].pct_change().fillna(0) * 100
        master_df['profit_growth'] = master_df['profit'].pct_change().fillna(0) * 100
        
        # Operational efficiency
        master_df['operational_efficiency'] = (master_df['profit'] / master_df['operational_cost'] * 100).round(2)
        
        # Convert year_month to datetime
        master_df['date'] = master_df['year_month'].astype(str) + '-01'
        master_df['date'] = pd.to_datetime(master_df['date'])
        
        self.master_df = master_df.sort_values('date').reset_index(drop=True)
        
        print(f"✓ Engineered KPIs: {len(self.master_df)} records")
        print(f"✓ Features: {list(self.master_df.columns)}")
        
        return self.master_df
    
    def get_master_dataset(self):
        """Get the final master dataset"""
        if self.master_df is None:
            self.load_data()
            self.clean_data()
            self.engineer_kpis()
        
        return self.master_df
    
    def save_master_dataset(self, output_path='data/master_dataset.csv'):
        """Save master dataset to CSV"""
        if self.master_df is None:
            self.get_master_dataset()
        
        self.master_df.to_csv(output_path, index=False)
        print(f"✓ Saved master dataset to {output_path}")

if __name__ == '__main__':
    pipeline = DataPipeline()
    pipeline.load_data()
    pipeline.clean_data()
    master_df = pipeline.engineer_kpis()
    pipeline.save_master_dataset()
