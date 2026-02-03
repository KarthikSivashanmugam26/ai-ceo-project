"""
Multi-Agent Executive AI System for AI CEO Project
CEO, CFO, CMO, COO agents that analyze strategies
"""

from typing import Dict, List
import numpy as np

class CFOAgent:
    """CFO Agent - Focuses on ROI and financial risk"""
    
    def analyze(self, scenario: Dict) -> Dict:
        """Analyze scenario from CFO perspective"""
        profit = scenario.get('predicted_profit', 0)
        revenue = scenario.get('predicted_revenue', 0)
        profit_margin = scenario.get('profit_margin', 0)
        
        # Calculate ROI
        state_changes = scenario.get('state_changes', {})
        investment = 0
        
        if 'marketing_spend' in state_changes:
            investment += state_changes.get('marketing_spend', 0) - scenario.get('current_marketing', 0)
        if 'total_payroll' in state_changes:
            investment += state_changes.get('total_payroll', 0) - scenario.get('current_payroll', 0)
        
        roi = ((profit - scenario.get('current_profit', 0)) / investment * 100) if investment > 0 else 0
        
        # Financial risk assessment
        risk_level = 'Low'
        if profit < 0:
            risk_level = 'Critical'
        elif profit_margin < 5:
            risk_level = 'High'
        elif profit_margin < 10:
            risk_level = 'Medium'
        
        # Recommendation
        recommendation = 'Approve'
        if risk_level == 'Critical':
            recommendation = 'Reject'
        elif risk_level == 'High':
            recommendation = 'Review'
        
        return {
            'agent': 'CFO',
            'roi': roi,
            'profit_margin': profit_margin,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'reasoning': f"ROI: {roi:.1f}%, Profit Margin: {profit_margin:.1f}%, Risk: {risk_level}"
        }

class CMOAgent:
    """CMO Agent - Focuses on growth and marketing"""
    
    def analyze(self, scenario: Dict) -> Dict:
        """Analyze scenario from CMO perspective"""
        revenue = scenario.get('predicted_revenue', 0)
        current_revenue = scenario.get('current_revenue', 1000000)
        revenue_growth = ((revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
        
        state_changes = scenario.get('state_changes', {})
        marketing_spend = state_changes.get('marketing_spend', scenario.get('current_marketing', 10000))
        current_marketing = scenario.get('current_marketing', 10000)
        marketing_change = ((marketing_spend - current_marketing) / current_marketing * 100) if current_marketing > 0 else 0
        
        # Marketing ROI
        marketing_roi = ((revenue - current_revenue) / (marketing_spend - current_marketing)) if (marketing_spend - current_marketing) != 0 else 0
        
        # Growth assessment
        growth_level = 'Stagnant'
        if revenue_growth > 15:
            growth_level = 'High Growth'
        elif revenue_growth > 5:
            growth_level = 'Moderate Growth'
        elif revenue_growth > 0:
            growth_level = 'Slow Growth'
        elif revenue_growth < -5:
            growth_level = 'Declining'
        
        # Recommendation
        recommendation = 'Approve'
        if growth_level == 'Declining':
            recommendation = 'Reject'
        elif growth_level == 'Stagnant' and marketing_change > 0:
            recommendation = 'Review'
        
        return {
            'agent': 'CMO',
            'revenue_growth': revenue_growth,
            'marketing_change': marketing_change,
            'marketing_roi': marketing_roi,
            'growth_level': growth_level,
            'recommendation': recommendation,
            'reasoning': f"Revenue Growth: {revenue_growth:.1f}%, Marketing Change: {marketing_change:.1f}%, Growth: {growth_level}"
        }

class COOAgent:
    """COO Agent - Focuses on operations and efficiency"""
    
    def analyze(self, scenario: Dict) -> Dict:
        """Analyze scenario from COO perspective"""
        state_changes = scenario.get('state_changes', {})
        
        # Operational metrics
        employee_count = state_changes.get('employee_count', scenario.get('current_employees', 100))
        operational_cost = state_changes.get('operational_cost', scenario.get('current_op_cost', 1000000))
        attrition_rate = state_changes.get('attrition_rate', scenario.get('current_attrition', 15))
        customer_satisfaction = state_changes.get('customer_satisfaction', scenario.get('current_satisfaction', 85))
        
        # Efficiency metrics
        revenue = scenario.get('predicted_revenue', 0)
        revenue_per_employee = revenue / employee_count if employee_count > 0 else 0
        cost_efficiency = (revenue / operational_cost) if operational_cost > 0 else 0
        
        # Operational risk
        risk_factors = []
        if attrition_rate > 25:
            risk_factors.append('High Attrition')
        if customer_satisfaction < 75:
            risk_factors.append('Low Satisfaction')
        if revenue_per_employee < 50000:
            risk_factors.append('Low Productivity')
        
        operational_risk = 'Low'
        if len(risk_factors) >= 2:
            operational_risk = 'High'
        elif len(risk_factors) == 1:
            operational_risk = 'Medium'
        
        # Recommendation
        recommendation = 'Approve'
        if operational_risk == 'High':
            recommendation = 'Reject'
        elif operational_risk == 'Medium':
            recommendation = 'Review'
        
        return {
            'agent': 'COO',
            'revenue_per_employee': revenue_per_employee,
            'cost_efficiency': cost_efficiency,
            'attrition_rate': attrition_rate,
            'customer_satisfaction': customer_satisfaction,
            'operational_risk': operational_risk,
            'risk_factors': risk_factors,
            'recommendation': recommendation,
            'reasoning': f"Efficiency: {cost_efficiency:.2f}, Attrition: {attrition_rate:.1f}%, Risk: {operational_risk}"
        }

class CEOAgent:
    """CEO Agent - Final decision maker, synthesizes all inputs"""
    
    def __init__(self):
        self.cfo = CFOAgent()
        self.cmo = CMOAgent()
        self.coo = COOAgent()
    
    def analyze(self, scenario: Dict, current_state: Dict = None) -> Dict:
        """Analyze scenario from CEO perspective, synthesizing all agent inputs"""
        
        # Add current state for comparison
        if current_state:
            scenario['current_revenue'] = current_state.get('revenue', 1000000)
            scenario['current_profit'] = current_state.get('profit', 200000)
            scenario['current_marketing'] = current_state.get('marketing_spend', 10000)
            scenario['current_payroll'] = current_state.get('total_payroll', 5000000)
            scenario['current_employees'] = current_state.get('employee_count', 100)
            scenario['current_op_cost'] = current_state.get('operational_cost', 1000000)
            scenario['current_attrition'] = current_state.get('attrition_rate', 15)
            scenario['current_satisfaction'] = current_state.get('customer_satisfaction', 85)
        
        # Get analyses from all agents
        cfo_analysis = self.cfo.analyze(scenario)
        cmo_analysis = self.cmo.analyze(scenario)
        coo_analysis = self.coo.analyze(scenario)
        
        # Synthesize recommendations
        recommendations = {
            cfo_analysis['recommendation']: cfo_analysis,
            cmo_analysis['recommendation']: cmo_analysis,
            coo_analysis['recommendation']: coo_analysis
        }
        
        # Decision logic
        if 'Reject' in recommendations:
            decision = 'Reject'
            confidence = 0.7
        elif recommendations.get('Review'):
            decision = 'Review'
            confidence = 0.5
        else:
            decision = 'Approve'
            confidence = 0.8
        
        # Calculate overall score
        profit = scenario.get('predicted_profit', 0)
        revenue = scenario.get('predicted_revenue', 0)
        
        # Weighted decision score
        score = 0
        if profit > 0:
            score += 40
        if revenue > scenario.get('current_revenue', 1000000):
            score += 30
        if cfo_analysis['risk_level'] == 'Low':
            score += 20
        if cmo_analysis['growth_level'] in ['Moderate Growth', 'High Growth']:
            score += 10
        
        return {
            'agent': 'CEO',
            'decision': decision,
            'confidence': confidence,
            'score': score,
            'cfo_analysis': cfo_analysis,
            'cmo_analysis': cmo_analysis,
            'coo_analysis': coo_analysis,
            'reasoning': f"Final Decision: {decision} (Confidence: {confidence*100:.0f}%). "
                        f"CFO: {cfo_analysis['recommendation']}, "
                        f"CMO: {cmo_analysis['recommendation']}, "
                        f"COO: {coo_analysis['recommendation']}"
        }
    
    def make_final_decision(self, scenarios: List[Dict], current_state: Dict = None) -> Dict:
        """Make final decision on best strategy"""
        if not scenarios:
            return None
        
        # Analyze all scenarios
        analyses = []
        for scenario in scenarios:
            analysis = self.analyze(scenario, current_state)
            scenario['ceo_analysis'] = analysis
            analyses.append((scenario, analysis))
        
        # Filter approved scenarios
        approved = [(s, a) for s, a in analyses if a['decision'] == 'Approve']
        
        if approved:
            # Sort by CEO score
            approved.sort(key=lambda x: x[1]['score'], reverse=True)
            best_scenario, best_analysis = approved[0]
            return {
                'scenario': best_scenario,
                'analysis': best_analysis,
                'status': 'approved'
            }
        else:
            # If no approved, return highest scoring review
            analyses.sort(key=lambda x: x[1]['score'], reverse=True)
            best_scenario, best_analysis = analyses[0]
            return {
                'scenario': best_scenario,
                'analysis': best_analysis,
                'status': 'needs_review'
            }
