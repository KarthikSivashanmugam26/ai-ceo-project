"""
Strategy Optimization for AI CEO Project
Ranks strategies based on profit, risk, and stability
"""

import numpy as np
from typing import List, Dict

class StrategyOptimizer:
    def __init__(self):
        self.scenarios = []
        
    def calculate_risk_score(self, scenario: Dict) -> float:
        """Calculate risk score for a scenario (0-100, lower is better)"""
        risk = 0
        
        # Revenue volatility risk
        revenue = scenario.get('predicted_revenue', 0)
        if revenue < 0:
            risk += 50
        elif revenue < scenario.get('current_revenue', 1000000) * 0.8:
            risk += 30
        
        # Profit margin risk
        profit_margin = scenario.get('profit_margin', 0)
        if profit_margin < 5:
            risk += 30
        elif profit_margin < 10:
            risk += 15
        
        # Negative profit risk
        profit = scenario.get('predicted_profit', 0)
        if profit < 0:
            risk += 40
        
        # State change volatility
        state_changes = scenario.get('state_changes', {})
        if 'attrition_rate' in state_changes:
            if state_changes['attrition_rate'] > 30:
                risk += 20
        
        if 'customer_satisfaction' in state_changes:
            if state_changes['customer_satisfaction'] < 75:
                risk += 25
        
        return min(100, risk)
    
    def calculate_stability_score(self, scenario: Dict) -> float:
        """Calculate stability score (0-100, higher is better)"""
        stability = 100
        
        # Reduce stability for extreme changes
        action = scenario.get('action', '')
        if '%' in action:
            try:
                changes = [float(x) for x in action.split() if '%' in x]
                max_change = max([abs(c) for c in changes])
                if max_change > 30:
                    stability -= 30
                elif max_change > 20:
                    stability -= 15
                elif max_change > 10:
                    stability -= 5
            except:
                pass
        
        # Reduce stability for negative outcomes
        profit = scenario.get('predicted_profit', 0)
        if profit < 0:
            stability -= 40
        
        # Reduce stability for high risk
        risk = self.calculate_risk_score(scenario)
        stability -= risk * 0.3
        
        return max(0, stability)
    
    def calculate_composite_score(self, scenario: Dict, 
                                  profit_weight=0.5, 
                                  risk_weight=0.3, 
                                  stability_weight=0.2) -> float:
        """Calculate composite score for ranking"""
        profit = scenario.get('predicted_profit', 0)
        risk = self.calculate_risk_score(scenario)
        stability = self.calculate_stability_score(scenario)
        
        # Normalize profit (assume max profit around 2M)
        normalized_profit = min(100, (profit / 2000000) * 100) if profit > 0 else 0
        
        # Composite score (higher is better)
        score = (normalized_profit * profit_weight + 
                (100 - risk) * risk_weight + 
                stability * stability_weight)
        
        return score
    
    def rank_strategies(self, scenarios: List[Dict]) -> List[Dict]:
        """Rank strategies by composite score"""
        # Add scores to scenarios
        for scenario in scenarios:
            scenario['risk_score'] = self.calculate_risk_score(scenario)
            scenario['stability_score'] = self.calculate_stability_score(scenario)
            scenario['composite_score'] = self.calculate_composite_score(scenario)
        
        # Sort by composite score (descending)
        ranked = sorted(scenarios, key=lambda x: x['composite_score'], reverse=True)
        
        return ranked
    
    def get_best_strategy(self, scenarios: List[Dict]) -> Dict:
        """Get the best strategy"""
        ranked = self.rank_strategies(scenarios)
        if ranked:
            return ranked[0]
        return None
    
    def get_top_n_strategies(self, scenarios: List[Dict], n=5) -> List[Dict]:
        """Get top N strategies"""
        ranked = self.rank_strategies(scenarios)
        return ranked[:n]
    
    def filter_strategies(self, scenarios: List[Dict], 
                         min_profit=None, 
                         max_risk=None,
                         min_stability=None) -> List[Dict]:
        """Filter strategies by criteria"""
        filtered = scenarios.copy()
        
        if min_profit is not None:
            filtered = [s for s in filtered if s.get('predicted_profit', 0) >= min_profit]
        
        if max_risk is not None:
            filtered = [s for s in filtered if self.calculate_risk_score(s) <= max_risk]
        
        if min_stability is not None:
            filtered = [s for s in filtered if self.calculate_stability_score(s) >= min_stability]
        
        return filtered
