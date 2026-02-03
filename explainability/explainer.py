"""
Explainable AI Module for AI CEO Project
Provides feature importance and decision explanations
"""

import numpy as np
from typing import Dict, List
import sys
import os

# Handle imports
try:
    from models.model_loader import ModelLoader
except ImportError:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from models.model_loader import ModelLoader

class AIExplainer:
    """Explainable AI for strategy decisions"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.model_loader.load_models()
        
        # Feature names
        self.feature_names = [
            'units_sold', 'marketing_spend', 'discount',
            'employee_count', 'total_payroll', 'avg_salary',
            'attrition_rate', 'avg_performance',
            'operational_cost', 'marketing_budget', 'rd_investment',
            'customer_satisfaction', 'market_share', 'competitor_count',
            'economic_index', 'revenue_growth', 'profit_growth'
        ]
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained models"""
        importance = {}
        
        if self.model_loader.profit_model:
            profit_importance = self.model_loader.profit_model.feature_importances_
            importance['profit'] = dict(zip(self.feature_names, profit_importance))
        
        if self.model_loader.revenue_model:
            revenue_importance = self.model_loader.revenue_model.feature_importances_
            importance['revenue'] = dict(zip(self.feature_names, revenue_importance))
        
        return importance
    
    def get_top_features(self, n=5, target='profit') -> List[Dict]:
        """Get top N most important features"""
        importance = self.get_feature_importance()
        
        if target not in importance:
            return []
        
        features = importance[target]
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'feature': name, 'importance': float(imp), 'rank': i+1}
            for i, (name, imp) in enumerate(sorted_features[:n])
        ]
    
    def explain_prediction(self, scenario: Dict, state_dict: Dict) -> Dict:
        """Explain why a prediction was made"""
        try:
            from simulation.strategy_simulator import StrategySimulator
        except ImportError:
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from simulation.strategy_simulator import StrategySimulator
        
        simulator = StrategySimulator()
        X = simulator.prepare_features(state_dict)
        
        # Get feature contributions (simplified)
        if self.model_loader.profit_model:
            # Use tree-based feature contributions
            tree = self.model_loader.profit_model.estimators_[0]
            contributions = {}
            
            # Get feature importances
            importances = self.model_loader.profit_model.feature_importances_
            
            # Scale by actual feature values
            for i, (name, value) in enumerate(zip(self.feature_names, X[0])):
                contributions[name] = {
                    'value': float(value),
                    'importance': float(importances[i]),
                    'contribution': float(value * importances[i])
                }
        
        # Identify key drivers
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]['contribution']),
            reverse=True
        )
        
        key_drivers = [
            {
                'feature': name,
                'value': info['value'],
                'impact': 'positive' if info['contribution'] > 0 else 'negative',
                'magnitude': abs(info['contribution'])
            }
            for name, info in sorted_contributions[:5]
        ]
        
        return {
            'key_drivers': key_drivers,
            'explanation': self._generate_explanation(key_drivers, scenario)
        }
    
    def _generate_explanation(self, key_drivers: List[Dict], scenario: Dict) -> str:
        """Generate human-readable explanation"""
        explanation_parts = []
        
        explanation_parts.append(f"The strategy '{scenario.get('strategy', 'Unknown')}' is predicted to:")
        explanation_parts.append(f"- Generate profit: ${scenario.get('predicted_profit', 0):,.2f}")
        explanation_parts.append(f"- Generate revenue: ${scenario.get('predicted_revenue', 0):,.2f}")
        explanation_parts.append(f"- Achieve profit margin: {scenario.get('profit_margin', 0):.2f}%")
        explanation_parts.append("")
        explanation_parts.append("Key factors driving this prediction:")
        
        for driver in key_drivers[:3]:
            impact_desc = "positively" if driver['impact'] == 'positive' else "negatively"
            explanation_parts.append(
                f"- {driver['feature'].replace('_', ' ').title()}: "
                f"{impact_desc} impacts prediction (value: {driver['value']:.2f})"
            )
        
        return "\n".join(explanation_parts)
    
    def explain_strategy_comparison(self, scenarios: List[Dict]) -> Dict:
        """Explain why one strategy is better than others"""
        if not scenarios:
            return {}
        
        # Find best strategy
        best = max(scenarios, key=lambda x: x.get('predicted_profit', 0))
        
        explanation = {
            'best_strategy': best.get('strategy', 'Unknown'),
            'best_profit': best.get('predicted_profit', 0),
            'comparison': []
        }
        
        for scenario in scenarios:
            if scenario != best:
                comparison = {
                    'strategy': scenario.get('strategy', 'Unknown'),
                    'profit_diff': best.get('predicted_profit', 0) - scenario.get('predicted_profit', 0),
                    'revenue_diff': best.get('predicted_revenue', 0) - scenario.get('predicted_revenue', 0),
                    'margin_diff': best.get('profit_margin', 0) - scenario.get('profit_margin', 0)
                }
                explanation['comparison'].append(comparison)
        
        return explanation
    
    def generate_executive_summary(self, scenario: Dict, analysis: Dict) -> str:
        """Generate executive summary of decision"""
        summary_parts = []
        
        summary_parts.append("=" * 60)
        summary_parts.append("EXECUTIVE DECISION SUMMARY")
        summary_parts.append("=" * 60)
        summary_parts.append("")
        
        summary_parts.append(f"Strategy: {scenario.get('strategy', 'Unknown')}")
        summary_parts.append(f"Action: {scenario.get('action', 'N/A')}")
        summary_parts.append("")
        
        summary_parts.append("Financial Projections:")
        summary_parts.append(f"  Predicted Revenue: ${scenario.get('predicted_revenue', 0):,.2f}")
        summary_parts.append(f"  Predicted Profit: ${scenario.get('predicted_profit', 0):,.2f}")
        summary_parts.append(f"  Profit Margin: {scenario.get('profit_margin', 0):.2f}%")
        summary_parts.append("")
        
        if 'ceo_analysis' in scenario:
            ceo = scenario['ceo_analysis']
            summary_parts.append(f"CEO Decision: {ceo.get('decision', 'Pending')}")
            summary_parts.append(f"Confidence: {ceo.get('confidence', 0)*100:.0f}%")
            summary_parts.append("")
            
            summary_parts.append("Executive Team Analysis:")
            if 'cfo_analysis' in ceo:
                summary_parts.append(f"  CFO: {ceo['cfo_analysis'].get('recommendation', 'N/A')} - {ceo['cfo_analysis'].get('reasoning', '')}")
            if 'cmo_analysis' in ceo:
                summary_parts.append(f"  CMO: {ceo['cmo_analysis'].get('recommendation', 'N/A')} - {ceo['cmo_analysis'].get('reasoning', '')}")
            if 'coo_analysis' in ceo:
                summary_parts.append(f"  COO: {ceo['coo_analysis'].get('recommendation', 'N/A')} - {ceo['coo_analysis'].get('reasoning', '')}")
        
        summary_parts.append("")
        summary_parts.append("=" * 60)
        
        return "\n".join(summary_parts)
