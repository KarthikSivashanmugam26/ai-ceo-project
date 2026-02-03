# 🎯 AI CEO - Corporate Strategy Decision System

A comprehensive, enterprise-grade AI platform that simulates business strategies, forecasts profit/revenue, and makes executive-level decisions using multi-agent AI systems, reinforcement learning, and explainable AI.

## 🌟 Features

### Core Capabilities
- **Data Pipeline**: Automated data loading, cleaning, preprocessing, and KPI engineering
- **ML Models**: Trained RandomForest and GradientBoosting models for profit/revenue prediction
- **Strategy Simulation**: Simulate pricing, marketing, HR, and cost optimization strategies
- **Decision Optimization**: Rank strategies based on profit, risk, and stability scores
- **Multi-Agent Executive AI**: CEO, CFO, CMO, and COO agents that analyze strategies
- **Reinforcement Learning**: AI agent that learns optimal strategies over time
- **Explainable AI**: Feature importance and human-readable decision explanations
- **Live Dashboard**: Interactive Streamlit web application

## 📁 Project Structure

```
ai_ceo_project/
├── data/                          # Data directory
│   ├── generate_sample_data.py   # Sample data generator
│   ├── sales_data.csv            # Sales dataset
│   ├── hr_data.csv               # HR dataset
│   ├── business_data.csv         # Business operations dataset
│   └── master_dataset.csv        # Master merged dataset
│
├── preprocessing/                 # Data processing
│   └── data_pipeline.py          # Data pipeline and KPI engineering
│
├── models/                        # ML models
│   ├── train_models.py           # Model training
│   └── model_loader.py           # Model loading utilities
│
├── simulation/                    # Strategy simulation
│   └── strategy_simulator.py     # Business strategy simulator
│
├── optimization/                  # Strategy optimization
│   └── strategy_optimizer.py     # Strategy ranking and optimization
│
├── multi_agents/                  # Multi-agent system
│   └── executive_agents.py       # CEO, CFO, CMO, COO agents
│
├── rl_agent/                      # Reinforcement learning
│   ├── rl_agent.py               # RL agent implementation
│   └── policy.json               # Learned policy (generated)
│
├── explainability/                # Explainable AI
│   └── explainer.py              # Feature importance and explanations
│
├── dashboard/                     # Streamlit dashboard
│   └── app.py                    # Main dashboard application
│
├── orchestrator.py                # Main orchestrator
├── main.py                        # Entry point
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd ai_ceo_project

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Project

```bash
# Generate sample data, run pipeline, and train models
python main.py setup
```

This will:
- Generate sample sales, HR, and business datasets
- Run the data pipeline to create master dataset
- Train ML models for profit/revenue prediction

### 3. Run Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 🌐 Deployment to Streamlit Cloud (Free Hosting)

### Step 1: Prepare Repository

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - AI CEO Project"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/ai-ceo-project.git
   git push -u origin main
   ```

2. **Create `.streamlit/config.toml`** (optional, for custom config):
   ```toml
   [theme]
   primaryColor = "#1f77b4"
   backgroundColor = "#ffffff"
   secondaryBackgroundColor = "#f0f2f6"
   ```

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/ai-ceo-project`
5. Set **Main file path**: `dashboard/app.py`
6. Click "Deploy"

### Step 3: Post-Deployment Setup

After deployment, you may need to run the setup once:

1. Open the deployed app
2. Use Streamlit Cloud's terminal (if available) or add a setup button in the dashboard
3. Run: `python main.py setup`

**Alternative**: Add automatic setup check in `dashboard/app.py`:

```python
# Add this at the start of main() function
if not os.path.exists('data/master_dataset.csv'):
    st.warning("Initializing project...")
    from main import setup_project
    setup_project()
```

### Your Public URL

Once deployed, your app will be available at:
```
https://YOUR_USERNAME-ai-ceo-project.streamlit.app
```

Share this URL with recruiters!

## 📊 Using the Dashboard

### Dashboard Tab
- View revenue, profit, and KPI trends over time
- Monitor business metrics

### Strategy Simulation Tab
- Configure strategy parameters using sidebar sliders:
  - Price Change (%)
  - Marketing Change (%)
  - Workforce Change (%)
  - Cost Reduction (%)
- Click "Run Strategy Simulation" to test a custom strategy
- Click "Generate All Scenarios" to see multiple strategy options
- View AI CEO decision, predictions, and executive team analysis

### AI Agents Tab
- See recommendations from each executive agent (CFO, CMO, COO)
- View CEO's final decision synthesis

### Analytics Tab
- Risk vs Profit analysis
- Feature importance visualization
- Strategy comparison charts

### Explainability Tab
- Understand why decisions were made
- View key drivers of predictions
- Read executive summaries

## 🔧 API Usage

You can also use the orchestrator programmatically:

```python
from orchestrator import AIOrchestrator

# Initialize
orchestrator = AIOrchestrator()
orchestrator.initialize()

# Run strategy analysis
results = orchestrator.run_strategy_analysis(
    price_change=5,
    marketing_change=10,
    hr_change=5,
    cost_reduction=5
)

# Access results
print(f"Best Strategy: {results['best_strategy']['strategy']}")
print(f"Predicted Profit: ${results['best_strategy']['predicted_profit']:,.2f}")
print(f"CEO Decision: {results['final_decision']['analysis']['decision']}")
```

## 🎓 System Architecture

### Data Flow
1. **Raw Data** → Data Pipeline → **Master Dataset**
2. **Master Dataset** → ML Training → **Trained Models**
3. **Strategy Input** → Simulator → **Scenarios**
4. **Scenarios** → Optimizer → **Ranked Strategies**
5. **Ranked Strategies** → Multi-Agents → **Executive Analysis**
6. **Analysis** → RL Agent → **Learned Policy**
7. **Final Decision** → Explainer → **Human-Readable Explanation**

### AI Components
- **ML Models**: RandomForest (profit), GradientBoosting (revenue)
- **Multi-Agent System**: Rule-based agents with domain expertise
- **RL Agent**: Q-learning for strategy selection
- **Explainability**: Feature importance and decision trees

## 📈 Model Performance

Models are evaluated using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

Check model metrics during training or in the dashboard.

## 🛠️ Customization

### Adding New Strategies
Edit `simulation/strategy_simulator.py` to add new strategy types.

### Modifying Agents
Edit `multi_agents/executive_agents.py` to adjust agent logic.

### Changing Models
Edit `models/train_models.py` to use different algorithms.

## 📝 Requirements

- **Python 3.8+** (Python 3.10+ recommended)
- See `requirements.txt` for package versions

### Installation
```bash
pip install -r requirements.txt
```

**Note**: For Streamlit Cloud deployment, Python 3.10+ is recommended.

## 🤝 Contributing

This is a demonstration project. Feel free to extend:
- Add more data sources
- Implement additional ML models
- Enhance agent reasoning
- Add more visualization types

## 📄 License

This project is provided as-is for demonstration purposes.

## 🎯 For Recruiters

This project demonstrates:
- ✅ End-to-end ML pipeline development
- ✅ Production-ready code structure
- ✅ Multi-agent AI systems
- ✅ Reinforcement learning implementation
- ✅ Explainable AI
- ✅ Web application development
- ✅ Deployment to cloud platforms

**Live Demo**: [Your Streamlit Cloud URL]

---

Built with ❤️ using Python, Streamlit, scikit-learn, and Plotly
