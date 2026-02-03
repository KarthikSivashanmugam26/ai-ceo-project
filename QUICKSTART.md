# ⚡ Quick Start Guide

Get the AI CEO Project running in 5 minutes!

## 🎯 For Local Development

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize Project
```bash
python main.py setup
```

This will:
- ✅ Generate sample datasets
- ✅ Run data pipeline
- ✅ Train ML models

### 3. Start Dashboard
```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser!

## 🌐 For Deployment (Streamlit Cloud)

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "AI CEO Project"
git remote add origin https://github.com/YOUR_USERNAME/ai-ceo-project.git
git push -u origin main
```

### 2. Deploy
1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your repository
4. Set main file: `dashboard/app.py`
5. Click "Deploy"

### 3. First Run
The dashboard will auto-detect missing data and guide you through setup.

**OR** pre-generate data:
```bash
python main.py setup
git add data/ models/
git commit -m "Add data and models"
git push
```

## 📊 Using the Dashboard

1. **Dashboard Tab**: View business metrics and trends
2. **Strategy Simulation**: 
   - Adjust sliders in sidebar
   - Click "Run Strategy Simulation"
   - View AI CEO decision
3. **AI Agents**: See executive team recommendations
4. **Analytics**: Explore feature importance and risk analysis
5. **Explainability**: Understand decision reasoning

## 🐛 Troubleshooting

**Problem**: "Models not found"
- **Solution**: Run `python main.py setup`

**Problem**: "Data not found"
- **Solution**: Run `python main.py setup`

**Problem**: Import errors
- **Solution**: Ensure you're in the project root directory

**Problem**: Streamlit won't start
- **Solution**: Check `streamlit --version` and reinstall if needed

## ✅ Verification

After setup, verify everything works:
```bash
python orchestrator.py
```

You should see:
- ✓ System initialized successfully
- ✓ Strategy analysis results
- ✓ CEO decision output

## 🎓 Next Steps

- Explore different strategy combinations
- Review executive agent recommendations
- Check feature importance in Analytics tab
- Read decision explanations in Explainability tab

## 📞 Need Help?

Check the main README.md for detailed documentation.

---

**Ready to go!** 🚀
