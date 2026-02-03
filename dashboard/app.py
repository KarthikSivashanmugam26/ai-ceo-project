"""
Streamlit Dashboard for AI CEO Project
Interactive web application for recruiters to view AI CEO decisions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from preprocessing.data_pipeline import DataPipeline
from simulation.strategy_simulator import StrategySimulator
from optimization.strategy_optimizer import StrategyOptimizer
from multi_agents.executive_agents import CEOAgent
from rl_agent.rl_agent import RLAgent
from explainability.explainer import AIExplainer

# Page config
st.set_page_config(
    page_title="AI CEO - Corporate Strategy Decision System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .decision-approved {
        color: #28a745;
        font-weight: bold;
    }
    .decision-rejected {
        color: #dc3545;
        font-weight: bold;
    }
    .decision-review {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare data"""
    try:
        # Check if data exists, if not, try to generate it
        if not os.path.exists('data/master_dataset.csv'):
            with st.spinner("Initializing project - this may take a minute..."):
                try:
                    # Try to run setup
                    from data.generate_sample_data import generate_all_data
                    generate_all_data()
                    
                    pipeline = DataPipeline()
                    pipeline.load_data()
                    pipeline.clean_data()
                    master_df = pipeline.engineer_kpis()
                    pipeline.save_master_dataset()
                    
                    # Try to train models if they don't exist
                    if not os.path.exists('models/profit_model.pkl'):
                        from models.train_models import ModelTrainer
                        trainer = ModelTrainer()
                        trainer.train_all()
                    
                    return master_df, pipeline.current_state
                except Exception as setup_error:
                    st.error(f"Auto-setup failed: {setup_error}")
                    st.info("Please run 'python main.py setup' in the terminal")
                    return None, None
        
        pipeline = DataPipeline()
        pipeline.load_data()
        pipeline.clean_data()
        master_df = pipeline.engineer_kpis()
        return master_df, pipeline.current_state
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run 'python main.py setup' first")
        return None, None

@st.cache_resource
def initialize_components():
    """Initialize AI components"""
    simulator = StrategySimulator()
    optimizer = StrategyOptimizer()
    ceo_agent = CEOAgent()
    rl_agent = RLAgent()
    explainer = AIExplainer()
    return simulator, optimizer, ceo_agent, rl_agent, explainer

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<div class="main-header">🎯 AI CEO - Corporate Strategy Decision System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        
        st.subheader("Strategy Simulation")
        price_change = st.slider("Price Change (%)", -20, 20, 0, 1)
        marketing_change = st.slider("Marketing Change (%)", -30, 50, 0, 5)
        hr_change = st.slider("Workforce Change (%)", -30, 30, 0, 5)
        cost_reduction = st.slider("Cost Reduction (%)", 0, 30, 0, 1)
        
        st.markdown("---")
        
        if st.button("🚀 Run Strategy Simulation", type="primary", use_container_width=True):
            st.session_state.run_simulation = True
        
        if st.button("🔄 Generate All Scenarios", use_container_width=True):
            st.session_state.generate_all = True
        
        st.markdown("---")
        st.info("💡 This AI system simulates business strategies and provides executive-level decisions.")
    
    # Load data
    master_df, current_state = load_data()
    
    if master_df is None:
        st.stop()
    
    # Initialize components
    simulator, optimizer, ceo_agent, rl_agent, explainer = initialize_components()
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Revenue", f"${master_df['revenue'].iloc[-1]:,.0f}")
    with col2:
        st.metric("Current Profit", f"${master_df['profit'].iloc[-1]:,.0f}")
    with col3:
        st.metric("Profit Margin", f"{master_df['profit_margin'].iloc[-1]:.2f}%")
    with col4:
        st.metric("Employees", f"{int(master_df['employee_count'].iloc[-1])}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "🎯 Strategy Simulation", "🤖 AI Agents", "📈 Analytics", "🔍 Explainability"
    ])
    
    with tab1:
        st.header("Business Overview")
        
        # Time series charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_revenue = px.line(
                master_df, x='date', y='revenue',
                title='Revenue Over Time',
                labels={'revenue': 'Revenue ($)', 'date': 'Date'}
            )
            fig_revenue.update_layout(height=300)
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            fig_profit = px.line(
                master_df, x='date', y='profit',
                title='Profit Over Time',
                labels={'profit': 'Profit ($)', 'date': 'Date'}
            )
            fig_profit.update_layout(height=300)
            st.plotly_chart(fig_profit, use_container_width=True)
        
        # KPI charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_margin = px.line(
                master_df, x='date', y='profit_margin',
                title='Profit Margin (%)',
                labels={'profit_margin': 'Margin (%)', 'date': 'Date'}
            )
            fig_margin.update_layout(height=300)
            st.plotly_chart(fig_margin, use_container_width=True)
        
        with col2:
            fig_attrition = px.line(
                master_df, x='date', y='attrition_rate',
                title='Attrition Rate (%)',
                labels={'attrition_rate': 'Rate (%)', 'date': 'Date'}
            )
            fig_attrition.update_layout(height=300)
            st.plotly_chart(fig_attrition, use_container_width=True)
    
    with tab2:
        st.header("Strategy Simulation")
        
        if st.session_state.get('run_simulation', False) or st.session_state.get('generate_all', False):
            with st.spinner("Running simulation..."):
                if st.session_state.get('generate_all', False):
                    # Generate all scenarios
                    scenarios = simulator.generate_scenarios()
                    st.session_state.scenarios = scenarios
                    st.session_state.generate_all = False
                else:
                    # Run custom simulation
                    scenario = simulator.simulate_combined_strategy(
                        price_change=price_change,
                        marketing_change=marketing_change,
                        hr_change=hr_change,
                        cost_reduction=cost_reduction
                    )
                    scenarios = [scenario]
                    st.session_state.scenarios = scenarios
                
                # Optimize and rank
                ranked_scenarios = optimizer.rank_strategies(scenarios)
                best_strategy = optimizer.get_best_strategy(scenarios)
                
                # CEO analysis
                current_state_dict = master_df.iloc[-1].to_dict()
                for s in ranked_scenarios:
                    s.update(current_state_dict)
                
                final_decision = ceo_agent.make_final_decision(ranked_scenarios, current_state_dict)
                
                st.session_state.ranked_scenarios = ranked_scenarios
                st.session_state.best_strategy = best_strategy
                st.session_state.final_decision = final_decision
        
        if 'final_decision' in st.session_state:
            decision = st.session_state.final_decision
            
            st.subheader("🎯 AI CEO Decision")
            
            decision_status = decision['status']
            decision_class = {
                'approved': 'decision-approved',
                'needs_review': 'decision-review',
                'rejected': 'decision-rejected'
            }.get(decision_status, '')
            
            st.markdown(f'<div class="{decision_class}">Status: {decision_status.upper()}</div>', unsafe_allow_html=True)
            
            scenario = decision['scenario']
            analysis = decision['analysis']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Revenue", f"${scenario.get('predicted_revenue', 0):,.0f}")
            with col2:
                st.metric("Predicted Profit", f"${scenario.get('predicted_profit', 0):,.0f}")
            with col3:
                st.metric("Profit Margin", f"{scenario.get('profit_margin', 0):.2f}%")
            
            st.markdown("---")
            
            st.subheader("Strategy Details")
            st.write(f"**Strategy:** {scenario.get('strategy', 'Unknown')}")
            st.write(f"**Action:** {scenario.get('action', 'N/A')}")
            st.write(f"**CEO Confidence:** {analysis.get('confidence', 0)*100:.0f}%")
            
            st.markdown("---")
            
            # Executive team analysis
            st.subheader("Executive Team Analysis")
            exec_cols = st.columns(3)
            
            with exec_cols[0]:
                cfo = analysis.get('cfo_analysis', {})
                st.write("**CFO Analysis**")
                st.write(f"ROI: {cfo.get('roi', 0):.1f}%")
                st.write(f"Risk: {cfo.get('risk_level', 'N/A')}")
                st.write(f"Recommendation: **{cfo.get('recommendation', 'N/A')}**")
            
            with exec_cols[1]:
                cmo = analysis.get('cmo_analysis', {})
                st.write("**CMO Analysis**")
                st.write(f"Growth: {cmo.get('revenue_growth', 0):.1f}%")
                st.write(f"Level: {cmo.get('growth_level', 'N/A')}")
                st.write(f"Recommendation: **{cmo.get('recommendation', 'N/A')}**")
            
            with exec_cols[2]:
                coo = analysis.get('coo_analysis', {})
                st.write("**COO Analysis**")
                st.write(f"Efficiency: {coo.get('cost_efficiency', 0):.2f}")
                st.write(f"Risk: {coo.get('operational_risk', 'N/A')}")
                st.write(f"Recommendation: **{coo.get('recommendation', 'N/A')}**")
            
            st.markdown("---")
            
            # Strategy comparison
            if 'ranked_scenarios' in st.session_state:
                st.subheader("Top 5 Strategies")
                top_5 = st.session_state.ranked_scenarios[:5]
                
                comparison_data = []
                for i, s in enumerate(top_5, 1):
                    comparison_data.append({
                        'Rank': i,
                        'Strategy': s.get('strategy', 'Unknown'),
                        'Action': s.get('action', 'N/A'),
                        'Revenue': s.get('predicted_revenue', 0),
                        'Profit': s.get('predicted_profit', 0),
                        'Margin': s.get('profit_margin', 0),
                        'Score': s.get('composite_score', 0)
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Visualization
                fig_comparison = go.Figure()
                fig_comparison.add_trace(go.Bar(
                    x=[s.get('strategy', 'Unknown') for s in top_5],
                    y=[s.get('predicted_profit', 0) for s in top_5],
                    name='Predicted Profit',
                    marker_color='lightblue'
                ))
                fig_comparison.update_layout(
                    title='Top 5 Strategies - Profit Comparison',
                    xaxis_title='Strategy',
                    yaxis_title='Profit ($)',
                    height=400
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
        else:
            st.info("👆 Use the sidebar to configure and run a strategy simulation")
    
    with tab3:
        st.header("Multi-Agent Executive AI")
        
        if 'ranked_scenarios' in st.session_state:
            st.subheader("Agent Recommendations")
            
            for i, scenario in enumerate(st.session_state.ranked_scenarios[:3], 1):
                with st.expander(f"Strategy {i}: {scenario.get('strategy', 'Unknown')}"):
                    if 'ceo_analysis' in scenario:
                        ceo_analysis = scenario['ceo_analysis']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Decision:** {ceo_analysis.get('decision', 'N/A')}")
                            st.write(f"**Confidence:** {ceo_analysis.get('confidence', 0)*100:.0f}%")
                            st.write(f"**Score:** {ceo_analysis.get('score', 0):.1f}")
                        
                        with col2:
                            st.write("**Agent Recommendations:**")
                            st.write(f"- CFO: {ceo_analysis.get('cfo_analysis', {}).get('recommendation', 'N/A')}")
                            st.write(f"- CMO: {ceo_analysis.get('cmo_analysis', {}).get('recommendation', 'N/A')}")
                            st.write(f"- COO: {ceo_analysis.get('coo_analysis', {}).get('recommendation', 'N/A')}")
        else:
            st.info("Run a strategy simulation first to see agent recommendations")
    
    with tab4:
        st.header("Analytics & Insights")
        
        if 'ranked_scenarios' in st.session_state:
            scenarios_df = pd.DataFrame(st.session_state.ranked_scenarios)
            
            # Risk vs Profit scatter
            fig_risk_profit = px.scatter(
                scenarios_df,
                x='risk_score',
                y='predicted_profit',
                color='composite_score',
                size='predicted_revenue',
                hover_data=['strategy', 'action'],
                title='Risk vs Profit Analysis',
                labels={'risk_score': 'Risk Score', 'predicted_profit': 'Predicted Profit ($)'}
            )
            st.plotly_chart(fig_risk_profit, use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = explainer.get_feature_importance()
            
            if 'profit' in feature_importance:
                top_features = explainer.get_top_features(n=10, target='profit')
                features_df = pd.DataFrame(top_features)
                
                fig_features = px.bar(
                    features_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 10 Features for Profit Prediction',
                    labels={'importance': 'Importance', 'feature': 'Feature'}
                )
                fig_features.update_layout(height=500)
                st.plotly_chart(fig_features, use_container_width=True)
        else:
            st.info("Run a strategy simulation first to see analytics")
    
    with tab5:
        st.header("Explainable AI")
        
        if 'final_decision' in st.session_state:
            scenario = st.session_state.final_decision['scenario']
            current_state_dict = master_df.iloc[-1].to_dict()
            
            explanation = explainer.explain_prediction(scenario, current_state_dict)
            
            st.subheader("Decision Explanation")
            st.markdown(f"```\n{explanation.get('explanation', 'N/A')}\n```")
            
            st.subheader("Key Drivers")
            key_drivers = explanation.get('key_drivers', [])
            drivers_df = pd.DataFrame(key_drivers)
            if not drivers_df.empty:
                st.dataframe(drivers_df, use_container_width=True, hide_index=True)
            
            st.subheader("Executive Summary")
            summary = explainer.generate_executive_summary(
                scenario,
                st.session_state.final_decision['analysis']
            )
            st.markdown(f"```\n{summary}\n```")
        else:
            st.info("Run a strategy simulation first to see explanations")

if __name__ == '__main__':
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    if 'generate_all' not in st.session_state:
        st.session_state.generate_all = False
    
    main()
