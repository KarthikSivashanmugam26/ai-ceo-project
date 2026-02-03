"""
ML Model Training for AI CEO Project
Trains models to predict profit and revenue
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class ModelTrainer:
    def __init__(self, data_path='data/master_dataset.csv'):
        self.data_path = data_path
        self.df = None
        self.profit_model = None
        self.revenue_model = None
        self.feature_importance = {}
        
    def load_data(self):
        """Load master dataset"""
        print("Loading master dataset...")
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f"✓ Loaded {len(self.df)} records")
        
    def prepare_features(self):
        """Prepare features for training"""
        print("Preparing features...")
        
        # Select features
        feature_cols = [
            'units_sold', 'marketing_spend', 'discount',
            'employee_count', 'total_payroll', 'avg_salary',
            'attrition_rate', 'avg_performance',
            'operational_cost', 'marketing_budget', 'rd_investment',
            'customer_satisfaction', 'market_share', 'competitor_count',
            'economic_index', 'revenue_growth', 'profit_growth'
        ]
        
        # Create feature matrix
        X = self.df[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Targets
        y_profit = self.df['profit'].values
        y_revenue = self.df['revenue'].values
        
        return X, y_profit, y_revenue, feature_cols
    
    def train_profit_model(self, X, y):
        """Train profit prediction model"""
        print("\nTraining Profit Prediction Model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train RandomForest
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"  Train RMSE: {train_rmse:.2f}")
        print(f"  Test RMSE: {test_rmse:.2f}")
        print(f"  Train MAE: {train_mae:.2f}")
        print(f"  Test MAE: {test_mae:.2f}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        
        self.profit_model = model
        self.profit_metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        return model
    
    def train_revenue_model(self, X, y):
        """Train revenue prediction model"""
        print("\nTraining Revenue Prediction Model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train GradientBoosting
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"  Train RMSE: {train_rmse:.2f}")
        print(f"  Test RMSE: {test_rmse:.2f}")
        print(f"  Train MAE: {train_mae:.2f}")
        print(f"  Test MAE: {test_mae:.2f}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        
        self.revenue_model = model
        self.revenue_metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        return model
    
    def get_feature_importance(self):
        """Get feature importance from models"""
        if self.profit_model and self.revenue_model:
            profit_importance = dict(zip(
                self.feature_cols,
                self.profit_model.feature_importances_
            ))
            revenue_importance = dict(zip(
                self.feature_cols,
                self.revenue_model.feature_importances_
            ))
            
            self.feature_importance = {
                'profit': profit_importance,
                'revenue': revenue_importance
            }
            
            return self.feature_importance
    
    def save_models(self, model_dir='models'):
        """Save trained models"""
        os.makedirs(model_dir, exist_ok=True)
        
        if self.profit_model:
            joblib.dump(self.profit_model, f'{model_dir}/profit_model.pkl')
            print(f"\n✓ Saved profit model to {model_dir}/profit_model.pkl")
        
        if self.revenue_model:
            joblib.dump(self.revenue_model, f'{model_dir}/revenue_model.pkl')
            print(f"✓ Saved revenue model to {model_dir}/revenue_model.pkl")
    
    def train_all(self):
        """Train all models"""
        self.load_data()
        X, y_profit, y_revenue, feature_cols = self.prepare_features()
        self.feature_cols = feature_cols
        
        self.train_profit_model(X, y_profit)
        self.train_revenue_model(X, y_revenue)
        self.get_feature_importance()
        self.save_models()
        
        return {
            'profit_model': self.profit_model,
            'revenue_model': self.revenue_model,
            'feature_importance': self.feature_importance,
            'profit_metrics': self.profit_metrics,
            'revenue_metrics': self.revenue_metrics
        }

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.train_all()
