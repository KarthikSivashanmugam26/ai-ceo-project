"""
Reinforcement Learning Agent for AI CEO Project
Learns optimal strategy selection over time
"""

import numpy as np
from typing import List, Dict
import json
import os

class RLAgent:
    """Reinforcement Learning Agent that learns from strategy outcomes"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}  # State-action value table
        self.history = []
        self.policy_file = 'rl_agent/policy.json'
        
        # Load existing policy if available
        self.load_policy()
    
    def state_to_key(self, state: Dict) -> str:
        """Convert state dictionary to hashable key"""
        # Discretize continuous values for Q-learning
        revenue_bucket = int(state.get('revenue', 1000000) / 200000)
        profit_bucket = int(state.get('profit', 200000) / 50000)
        margin_bucket = int(state.get('profit_margin', 10))
        
        return f"{revenue_bucket}_{profit_bucket}_{margin_bucket}"
    
    def action_to_key(self, action: Dict) -> str:
        """Convert action to hashable key"""
        strategy = action.get('strategy', 'Unknown')
        action_desc = action.get('action', '')
        
        # Extract key parameters
        if 'price' in action_desc.lower():
            try:
                val = float([x for x in action_desc.split() if '%' in x][0].replace('%', ''))
                return f"price_{int(val)}"
            except:
                pass
        elif 'marketing' in action_desc.lower():
            try:
                val = float([x for x in action_desc.split() if '%' in x][0].replace('%', ''))
                return f"marketing_{int(val)}"
            except:
                pass
        elif 'hr' in action_desc.lower() or 'workforce' in action_desc.lower():
            try:
                val = float([x for x in action_desc.split() if '%' in x][0].replace('%', ''))
                return f"hr_{int(val)}"
            except:
                pass
        elif 'cost' in action_desc.lower():
            try:
                val = float([x for x in action_desc.split() if '%' in x][0].replace('%', ''))
                return f"cost_{int(val)}"
            except:
                pass
        
        return f"{strategy}_{hash(action_desc) % 1000}"
    
    def get_reward(self, scenario: Dict) -> float:
        """Calculate reward for a scenario"""
        profit = scenario.get('predicted_profit', 0)
        revenue = scenario.get('predicted_revenue', 0)
        profit_margin = scenario.get('profit_margin', 0)
        
        # Base reward from profit
        reward = profit / 10000  # Scale down
        
        # Bonus for good margins
        if profit_margin > 15:
            reward += 10
        elif profit_margin > 10:
            reward += 5
        
        # Penalty for negative outcomes
        if profit < 0:
            reward -= 20
        if revenue < scenario.get('current_revenue', 1000000) * 0.8:
            reward -= 10
        
        return reward
    
    def select_action(self, scenarios: List[Dict], current_state: Dict = None) -> Dict:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.exploration_rate:
            # Explore: random selection
            return np.random.choice(scenarios)
        
        # Exploit: select best known action
        best_scenario = None
        best_value = float('-inf')
        
        state_key = self.state_to_key(current_state) if current_state else "default"
        
        for scenario in scenarios:
            action_key = self.action_to_key(scenario)
            q_key = f"{state_key}_{action_key}"
            
            # Get Q-value (default to reward if not in table)
            q_value = self.q_table.get(q_key, self.get_reward(scenario))
            
            if q_value > best_value:
                best_value = q_value
                best_scenario = scenario
        
        return best_scenario if best_scenario else scenarios[0]
    
    def update_q_value(self, state: Dict, action: Dict, reward: float, next_state: Dict = None):
        """Update Q-value using Q-learning"""
        state_key = self.state_to_key(state)
        action_key = self.action_to_key(action)
        q_key = f"{state_key}_{action_key}"
        
        # Current Q-value
        current_q = self.q_table.get(q_key, 0)
        
        # Max future Q-value
        if next_state:
            next_state_key = self.state_to_key(next_state)
            max_future_q = max([
                self.q_table.get(f"{next_state_key}_{a}", 0)
                for a in ['price_5', 'marketing_10', 'hr_10', 'cost_10']  # Sample actions
            ], default=0)
        else:
            max_future_q = 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )
        
        self.q_table[q_key] = new_q
        
        # Store in history
        self.history.append({
            'state': state_key,
            'action': action_key,
            'reward': reward,
            'q_value': new_q
        })
    
    def learn_from_scenarios(self, scenarios: List[Dict], current_state: Dict):
        """Learn from a set of scenarios"""
        for scenario in scenarios:
            reward = self.get_reward(scenario)
            # Use scenario as next state approximation
            next_state = {
                'revenue': scenario.get('predicted_revenue', 0),
                'profit': scenario.get('predicted_profit', 0),
                'profit_margin': scenario.get('profit_margin', 0)
            }
            self.update_q_value(current_state, scenario, reward, next_state)
    
    def get_learned_policy(self) -> Dict:
        """Get the learned policy"""
        return self.q_table.copy()
    
    def save_policy(self):
        """Save learned policy to file"""
        os.makedirs('rl_agent', exist_ok=True)
        with open(self.policy_file, 'w') as f:
            json.dump({
                'q_table': self.q_table,
                'history_count': len(self.history)
            }, f, indent=2)
    
    def load_policy(self):
        """Load learned policy from file"""
        if os.path.exists(self.policy_file):
            try:
                with open(self.policy_file, 'r') as f:
                    data = json.load(f)
                    self.q_table = data.get('q_table', {})
                    print(f"✓ Loaded RL policy with {len(self.q_table)} Q-values")
            except Exception as e:
                print(f"⚠ Could not load policy: {e}")
                self.q_table = {}
    
    def get_best_learned_action(self, scenarios: List[Dict], current_state: Dict) -> Dict:
        """Get best action based on learned policy"""
        state_key = self.state_to_key(current_state)
        best_scenario = None
        best_q = float('-inf')
        
        for scenario in scenarios:
            action_key = self.action_to_key(scenario)
            q_key = f"{state_key}_{action_key}"
            q_value = self.q_table.get(q_key, float('-inf'))
            
            if q_value > best_q:
                best_q = q_value
                best_scenario = scenario
        
        return best_scenario if best_scenario else scenarios[0]
