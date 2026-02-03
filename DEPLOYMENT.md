# 🚀 Deployment Guide - AI CEO Project

## Quick Deployment to Streamlit Cloud (Free)

### Prerequisites
- GitHub account
- Python project ready to push

### Step-by-Step Instructions

#### 1. Prepare Your Repository

```bash
# Initialize git (if not already done)
cd ai_ceo_project
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - AI CEO Project"

# Create GitHub repository and push
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-ceo-project.git
git push -u origin main
```

#### 2. Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Configure**:
   - **Repository**: Select `YOUR_USERNAME/ai-ceo-project`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
   - **App URL** (optional): Customize if desired
5. **Click "Deploy"**

#### 3. Post-Deployment Setup

After deployment, the app will try to load data. If data doesn't exist:

**Option A: Use Streamlit Cloud Terminal** (if available)
```bash
python main.py setup
```

**Option B: Add Auto-Setup** (Recommended)

The dashboard already includes auto-setup detection. If data is missing, it will show a warning.

**Option C: Pre-generate Data Locally**

Before pushing to GitHub, run setup locally and commit the data files:
```bash
python main.py setup
git add data/*.csv models/*.pkl
git commit -m "Add pre-generated data and models"
git push
```

#### 4. Your Live URL

Once deployed, your app will be available at:
```
https://YOUR_USERNAME-ai-ceo-project.streamlit.app
```

Or if you customized:
```
https://YOUR_CUSTOM_NAME.streamlit.app
```

## Alternative Deployment Options

### Heroku

1. Create `Procfile`:
```
web: streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Create `runtime.txt`:
```
python-3.11.0
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Docker

1. Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and run:
```bash
docker build -t ai-ceo .
docker run -p 8501:8501 ai-ceo
```

### Local Network Access

Run locally and share on your network:
```bash
streamlit run dashboard/app.py --server.address=0.0.0.0
```

Access from other devices: `http://YOUR_IP:8501`

## Troubleshooting

### Issue: Models not found
**Solution**: Run `python main.py setup` before deploying or include models in repository

### Issue: Import errors
**Solution**: Ensure all `__init__.py` files are present and paths are correct

### Issue: Data not loading
**Solution**: Check file paths are relative (not absolute) and data files exist

### Issue: Streamlit Cloud timeout
**Solution**: Pre-generate data and models, commit them to repository

## Environment Variables (Optional)

Create `.streamlit/secrets.toml` for sensitive data:
```toml
[secrets]
API_KEY = "your-api-key"
```

**Note**: Never commit secrets.toml to public repositories!

## Performance Optimization

1. **Cache data loading** in Streamlit (already implemented)
2. **Pre-generate models** and commit to repo
3. **Use smaller datasets** for faster loading
4. **Enable Streamlit's caching** for expensive operations

## Monitoring

- Streamlit Cloud provides basic analytics
- Check app logs in Streamlit Cloud dashboard
- Monitor resource usage

## Updates

To update your deployed app:
```bash
git add .
git commit -m "Update features"
git push
```

Streamlit Cloud will automatically redeploy!
