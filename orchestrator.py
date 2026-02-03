"""
Main Orchestrator for AI CEO Project
Coordinates all components and provides unified API
"""

import pandas as pd
from preprocessing.data_pipeline import DataPipeline
from models.train_models import ModelTrainer
from simulation.strategy_simulator import StrategySimulator
from optimization.strategy_optimizer import StrategyOptimizer
from multi_agents.executive_agents import CEOAgent
from rl_agent.rl_agent import RLAgent
from explainability.explainer import AIExplainer

class AIOrchestrator:
    """Main orchestrator for AI CEO system"""
    
    def __init__(self):
        self.pipeline = DataPipeline()
        self.simulator = None
        self.optimizer = StrategyOptimizer()
        self.ceo_agent = CEOAgent()
        self.rl_agent = RLAgent()
        self.explainer = AIExplainer()
        self.master_df = None
        self.current_state = None
        
    def initialize(self):
        """Initialize all components"""
        print("Initializing AI CEO System...")
        
        # Load and prepare data
        print("\n1. Loading data pipeline...")
        self.pipeline.load_data()
        self.pipeline.clean_data()
        self.master_df = self.pipeline.engineer_kpis()
        self.current_state = self.master_df.iloc[-1].to_dict()
        
        # Train models if not already trained
        print("\n2. Checking models...")
        import os
        if not os.path.exists('models/profit_model.pkl') or not os.path.exists('models/revenue_model.pkl'):
            print("   Training models...")
            trainer = ModelTrainer()
            trainer.train_all()
        else:
            print("   Models already trained")
        
        # Initialize simulator
        print("\n3. Initializing strategy simulator...")
        self.simulator = StrategySimulator()
        
        print("\n✓ System initialized successfully")
        
    def run_strategy_analysis(self, **strategy_params):
        """Run complete strategy analysis"""
        if self.simulator is None:
            self.initialize()
        
        # Generate scenarios
        if strategy_params:
            scenario = self.simulator.simulate_combined_strategy(**strategy_params)
            scenarios = [scenario]
        else:
            scenarios = self.simulator.generate_scenarios()
        
        # Add current state for comparison
        for s in scenarios:
            s.update(self.current_state)
        
        # Optimize and rank
        ranked_scenarios = self.optimizer.rank_strategies(scenarios)
        
        # CEO analysis
        final_decision = self.ceo_agent.make_final_decision(ranked_scenarios, self.current_state)
        
        # RL agent learning
        self.rl_agent.learn_from_scenarios(scenarios, self.current_state)
        
        # Explainability
        explanation = self.explainer.explain_prediction(
            final_decision['scenario'],
            self.current_state
        )
        
        return {
            'scenarios': ranked_scenarios,
            'best_strategy': self.optimizer.get_best_strategy(scenarios),
            'final_decision': final_decision,
            'explanation': explanation,
            'rl_recommendation': self.rl_agent.get_best_learned_action(scenarios, self.current_state)
        }
    
    def get_current_state(self):
        """Get current business state"""
        if self.master_df is None:
            self.initialize()
        return self.current_state
    
    def get_insights(self):
        """Get business insights"""
        if self.master_df is None:
            self.initialize()
        
        insights = {
            'current_revenue': self.current_state.get('revenue', 0),
            'current_profit': self.current_state.get('profit', 0),
            'profit_margin': self.current_state.get('profit_margin', 0),
            'revenue_growth': self.current_state.get('revenue_growth', 0),
            'profit_growth': self.current_state.get('profit_growth', 0),
            'attrition_rate': self.current_state.get('attrition_rate', 0),
            'feature_importance': self.explainer.get_top_features(n=5)
        }
        
        return insights

if __name__ == '__main__':
    orchestrator = AIOrchestrator()
    orchestrator.initialize()
    
    print("\n" + "="*60)
    print("Running Strategy Analysis...")
    print("="*60)
    
    results = orchestrator.run_strategy_analysis(
        price_change=5,
        marketing_change=10,
        hr_change=5,
        cost_reduction=5
    )
    
    print("\nBest Strategy:", results['best_strategy']['strategy'])
    print("Predicted Profit:", f"${results['best_strategy']['predicted_profit']:,.2f}")
    print("CEO Decision:", results['final_decision']['analysis']['decision'])
