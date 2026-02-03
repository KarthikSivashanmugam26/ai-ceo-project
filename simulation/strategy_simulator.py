"""
Strategy Simulation Engine for AI CEO Project
Simulates business decisions and predicts outcomes
"""

import pandas as pd
import numpy as np
import sys
import os

# Handle imports
try:
    from models.model_loader import ModelLoader
except ImportError:
    # Add parent directory to path if needed
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from models.model_loader import ModelLoader

class StrategySimulator:
    def __init__(self, master_data_path='data/master_dataset.csv'):
        self.master_data_path = master_data_path
        self.model_loader = ModelLoader()
        try:
            self.model_loader.load_models()
        except Exception as e:
            print(f"Warning: Could not load models: {e}")
            print("Please run 'python main.py setup' first")
        self.current_state = None
        try:
            self.load_current_state()
        except Exception as e:
            print(f"Warning: Could not load current state: {e}")
        
    def load_current_state(self):
        """Load current business state"""
        if not os.path.exists(self.master_data_path):
            raise FileNotFoundError(f"Master dataset not found at {self.master_data_path}")
        df = pd.read_csv(self.master_data_path)
        if len(df) == 0:
            raise ValueError("Master dataset is empty")
        self.current_state = df.iloc[-1].to_dict()
        
    def prepare_features(self, state_dict):
        """Prepare feature vector from state dictionary"""
        feature_cols = [
            'units_sold', 'marketing_spend', 'discount',
            'employee_count', 'total_payroll', 'avg_salary',
            'attrition_rate', 'avg_performance',
            'operational_cost', 'marketing_budget', 'rd_investment',
            'customer_satisfaction', 'market_share', 'competitor_count',
            'economic_index', 'revenue_growth', 'profit_growth'
        ]
        
        # Handle missing values with defaults
        defaults = {
            'units_sold': 1000,
            'marketing_spend': 10000,
            'discount': 0.1,
            'employee_count': 100,
            'total_payroll': 5000000,
            'avg_salary': 50000,
            'attrition_rate': 15,
            'avg_performance': 80,
            'operational_cost': 1000000,
            'marketing_budget': 200000,
            'rd_investment': 100000,
            'customer_satisfaction': 85,
            'market_share': 20,
            'competitor_count': 5,
            'economic_index': 1.0,
            'revenue_growth': 5,
            'profit_growth': 3
        }
        
        features = []
        for col in feature_cols:
            val = state_dict.get(col, defaults.get(col, 0))
            if pd.isna(val):
                val = defaults.get(col, 0)
            features.append(float(val))
        
        return np.array(features).reshape(1, -1)
    
    def simulate_pricing_strategy(self, price_change_pct=0):
        """Simulate pricing change strategy"""
        state = self.current_state.copy()
        
        # Adjust unit price (affects revenue directly)
        current_revenue = state.get('revenue', 1000000)
        current_units = state.get('units_sold', 1000)
        
        # Price change affects units sold (elasticity)
        price_elasticity = -1.5  # 1% price increase = 1.5% unit decrease
        units_change = -price_change_pct * price_elasticity
        
        state['units_sold'] = current_units * (1 + units_change / 100)
        state['revenue'] = current_revenue * (1 + price_change_pct / 100) * (1 + units_change / 100)
        
        # Predict outcomes
        X = self.prepare_features(state)
        predicted_profit = self.model_loader.predict_profit(X)[0]
        predicted_revenue = self.model_loader.predict_revenue(X)[0]
        
        return {
            'strategy': 'Pricing Change',
            'action': f'{price_change_pct:+.1f}% price change',
            'predicted_revenue': predicted_revenue,
            'predicted_profit': predicted_profit,
            'profit_margin': (predicted_profit / predicted_revenue * 100) if predicted_revenue > 0 else 0,
            'state_changes': {
                'units_sold': state['units_sold'],
                'revenue': state['revenue']
            }
        }
    
    def simulate_marketing_strategy(self, marketing_change_pct=0):
        """Simulate marketing spend change strategy"""
        state = self.current_state.copy()
        
        current_marketing = state.get('marketing_spend', 10000)
        current_budget = state.get('marketing_budget', 200000)
        
        # Increase marketing spend
        new_marketing = current_marketing * (1 + marketing_change_pct / 100)
        new_budget = current_budget * (1 + marketing_change_pct / 100)
        
        state['marketing_spend'] = new_marketing
        state['marketing_budget'] = new_budget
        
        # Marketing affects revenue (ROI ~2:1)
        marketing_roi = 2.0
        revenue_boost = (new_marketing - current_marketing) * marketing_roi
        state['revenue'] = state.get('revenue', 1000000) + revenue_boost
        
        # Predict outcomes
        X = self.prepare_features(state)
        predicted_profit = self.model_loader.predict_profit(X)[0]
        predicted_revenue = self.model_loader.predict_revenue(X)[0]
        
        return {
            'strategy': 'Marketing Investment',
            'action': f'{marketing_change_pct:+.1f}% marketing spend change',
            'predicted_revenue': predicted_revenue,
            'predicted_profit': predicted_profit,
            'profit_margin': (predicted_profit / predicted_revenue * 100) if predicted_revenue > 0 else 0,
            'state_changes': {
                'marketing_spend': new_marketing,
                'marketing_budget': new_budget
            }
        }
    
    def simulate_hr_strategy(self, hiring_change_pct=0):
        """Simulate hiring/firing strategy"""
        state = self.current_state.copy()
        
        current_employees = state.get('employee_count', 100)
        current_payroll = state.get('total_payroll', 5000000)
        current_attrition = state.get('attrition_rate', 15)
        
        # Change employee count
        new_employees = current_employees * (1 + hiring_change_pct / 100)
        avg_salary = current_payroll / current_employees if current_employees > 0 else 50000
        new_payroll = new_employees * avg_salary
        
        state['employee_count'] = new_employees
        state['total_payroll'] = new_payroll
        state['avg_salary'] = avg_salary
        
        # More employees can improve operations but increase costs
        if hiring_change_pct > 0:
            # Hiring reduces attrition risk
            state['attrition_rate'] = max(5, current_attrition - hiring_change_pct * 0.5)
            state['avg_performance'] = min(100, state.get('avg_performance', 80) + hiring_change_pct * 0.2)
        else:
            # Firing increases attrition risk
            state['attrition_rate'] = min(50, current_attrition + abs(hiring_change_pct) * 0.3)
        
        # Predict outcomes
        X = self.prepare_features(state)
        predicted_profit = self.model_loader.predict_profit(X)[0]
        predicted_revenue = self.model_loader.predict_revenue(X)[0]
        
        return {
            'strategy': 'HR Strategy',
            'action': f'{hiring_change_pct:+.1f}% workforce change',
            'predicted_revenue': predicted_revenue,
            'predicted_profit': predicted_profit,
            'profit_margin': (predicted_profit / predicted_revenue * 100) if predicted_revenue > 0 else 0,
            'state_changes': {
                'employee_count': new_employees,
                'total_payroll': new_payroll,
                'attrition_rate': state['attrition_rate']
            }
        }
    
    def simulate_cost_optimization(self, cost_reduction_pct=0):
        """Simulate cost optimization strategy"""
        state = self.current_state.copy()
        
        current_op_cost = state.get('operational_cost', 1000000)
        new_op_cost = current_op_cost * (1 - cost_reduction_pct / 100)
        
        state['operational_cost'] = new_op_cost
        
        # Cost reduction might affect quality/customer satisfaction
        if cost_reduction_pct > 10:
            state['customer_satisfaction'] = max(70, state.get('customer_satisfaction', 85) - cost_reduction_pct * 0.3)
        
        # Predict outcomes
        X = self.prepare_features(state)
        predicted_profit = self.model_loader.predict_profit(X)[0]
        predicted_revenue = self.model_loader.predict_revenue(X)[0]
        
        return {
            'strategy': 'Cost Optimization',
            'action': f'{cost_reduction_pct:.1f}% operational cost reduction',
            'predicted_revenue': predicted_revenue,
            'predicted_profit': predicted_profit,
            'profit_margin': (predicted_profit / predicted_revenue * 100) if predicted_revenue > 0 else 0,
            'state_changes': {
                'operational_cost': new_op_cost,
                'customer_satisfaction': state.get('customer_satisfaction', 85)
            }
        }
    
    def simulate_combined_strategy(self, **kwargs):
        """Simulate combined strategy with multiple changes"""
        state = self.current_state.copy()
        
        # Apply all changes
        if 'price_change' in kwargs:
            price_change = kwargs['price_change']
            current_revenue = state.get('revenue', 1000000)
            current_units = state.get('units_sold', 1000)
            units_change = -price_change * 1.5
            state['units_sold'] = current_units * (1 + units_change / 100)
            state['revenue'] = current_revenue * (1 + price_change / 100) * (1 + units_change / 100)
        
        if 'marketing_change' in kwargs:
            marketing_change = kwargs['marketing_change']
            current_marketing = state.get('marketing_spend', 10000)
            current_budget = state.get('marketing_budget', 200000)
            state['marketing_spend'] = current_marketing * (1 + marketing_change / 100)
            state['marketing_budget'] = current_budget * (1 + marketing_change / 100)
        
        if 'hr_change' in kwargs:
            hr_change = kwargs['hr_change']
            current_employees = state.get('employee_count', 100)
            current_payroll = state.get('total_payroll', 5000000)
            avg_salary = current_payroll / current_employees if current_employees > 0 else 50000
            state['employee_count'] = current_employees * (1 + hr_change / 100)
            state['total_payroll'] = state['employee_count'] * avg_salary
        
        if 'cost_reduction' in kwargs:
            cost_reduction = kwargs['cost_reduction']
            current_op_cost = state.get('operational_cost', 1000000)
            state['operational_cost'] = current_op_cost * (1 - cost_reduction / 100)
        
        # Predict outcomes
        X = self.prepare_features(state)
        predicted_profit = self.model_loader.predict_profit(X)[0]
        predicted_revenue = self.model_loader.predict_revenue(X)[0]
        
        actions = []
        if 'price_change' in kwargs:
            actions.append(f"Price: {kwargs['price_change']:+.1f}%")
        if 'marketing_change' in kwargs:
            actions.append(f"Marketing: {kwargs['marketing_change']:+.1f}%")
        if 'hr_change' in kwargs:
            actions.append(f"HR: {kwargs['hr_change']:+.1f}%")
        if 'cost_reduction' in kwargs:
            actions.append(f"Cost: -{kwargs['cost_reduction']:.1f}%")
        
        return {
            'strategy': 'Combined Strategy',
            'action': ' | '.join(actions),
            'predicted_revenue': predicted_revenue,
            'predicted_profit': predicted_profit,
            'profit_margin': (predicted_profit / predicted_revenue * 100) if predicted_revenue > 0 else 0,
            'state_changes': state
        }
    
    def generate_scenarios(self):
        """Generate multiple strategy scenarios"""
        scenarios = []
        
        # Pricing scenarios
        for change in [-10, -5, 0, 5, 10]:
            scenarios.append(self.simulate_pricing_strategy(change))
        
        # Marketing scenarios
        for change in [-20, -10, 0, 10, 20, 30]:
            scenarios.append(self.simulate_marketing_strategy(change))
        
        # HR scenarios
        for change in [-20, -10, 0, 10, 20]:
            scenarios.append(self.simulate_hr_strategy(change))
        
        # Cost optimization scenarios
        for reduction in [0, 5, 10, 15, 20]:
            scenarios.append(self.simulate_cost_optimization(reduction))
        
        return scenarios
