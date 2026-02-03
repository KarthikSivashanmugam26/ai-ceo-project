"""
Model Loader for AI CEO Project
Utility to load trained models
"""

import joblib
import os

class ModelLoader:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.profit_model = None
        self.revenue_model = None
        
    def load_models(self):
        """Load trained models"""
        profit_path = f'{self.model_dir}/profit_model.pkl'
        revenue_path = f'{self.model_dir}/revenue_model.pkl'
        
        if os.path.exists(profit_path):
            self.profit_model = joblib.load(profit_path)
            print(f"✓ Loaded profit model from {profit_path}")
        else:
            print(f"⚠ Profit model not found at {profit_path}")
            
        if os.path.exists(revenue_path):
            self.revenue_model = joblib.load(revenue_path)
            print(f"✓ Loaded revenue model from {revenue_path}")
        else:
            print(f"⚠ Revenue model not found at {revenue_path}")
            
        return self.profit_model, self.revenue_model
    
    def predict_profit(self, X):
        """Predict profit"""
        if self.profit_model is None:
            self.load_models()
        return self.profit_model.predict(X)
    
    def predict_revenue(self, X):
        """Predict revenue"""
        if self.revenue_model is None:
            self.load_models()
        return self.revenue_model.predict(X)
